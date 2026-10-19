import csv
import gc
import itertools
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "camembert-base"
TRAIN_FILE = Path("src/NER_training_files/train.tsv")
DEV_FILE = Path("src/NER_training_files/dev.tsv")

MAX_LENGTH = 64         
SEED = 42
NUM_EPOCHS = 5
WARMUP_RATIO = 0.0
WEIGHT_DECAY = 0.0
EVAL_BATCH_SIZE = 128

LEARNING_RATES = [2e-5, 3e-5, 4e-5]
BATCH_SIZES = [16, 32]

OUT_DIR = Path("grid_search")
OUT_DIR.mkdir(exist_ok=True)
RESULTS_CSV = OUT_DIR / "grid_results.csv"
TMP_DIR = Path("./results_grid")

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16

NON_METRIC_KEYS = {"eval_runtime", "eval_samples_per_second", "eval_steps_per_second"}


# ─────────────────────────────────────────────────────────────────────────────
# Données
# ─────────────────────────────────────────────────────────────────────────────

def read_tsv_file(file_path: Path) -> List[Tuple[List[str], List[str]]]:
    sentences, tokens, tags = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    tokens.append(parts[0])
                    tags.append(parts[1].strip())
        if tokens:
            sentences.append((tokens, tags))
    return sentences


def create_label_mappings(sentences) -> Tuple[Dict, Dict]:
    labels = sorted({t for _, tags in sentences for t in tags})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def prepare_dataset(sentences, label2id: Dict) -> Dataset:
    return Dataset.from_dict({
        "tokens": [t for t, _ in sentences],
        "ner_tags": [[label2id[x] for x in tags] for _, tags in sentences],
    })


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    all_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous, label_ids = None, []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous = word_idx
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


# ─────────────────────────────────────────────────────────────────────────────
# Métriques (seqeval strict, IOB2)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred, id2label):
    predictions, labels = eval_pred

    true_labels, true_predictions = [], []
    for prediction, label in zip(predictions, labels):
        t, p = [], []
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                t.append(id2label[label_id])
                p.append(id2label[pred_id])
        if t:
            true_labels.append(t)
            true_predictions.append(p)

    kw = dict(mode="strict", scheme=IOB2)
    report = classification_report(
        true_labels, true_predictions, output_dict=True, zero_division=0, **kw
    )

    results = {
        "precision": precision_score(true_labels, true_predictions, **kw),
        "recall": recall_score(true_labels, true_predictions, **kw),
        "f1_micro": f1_score(true_labels, true_predictions, **kw),
        "f1_macro": report["macro avg"]["f1-score"],
        "accuracy": accuracy_score(true_labels, true_predictions),
    }
    for ent, scores in report.items():
        if ent not in ("micro avg", "macro avg", "weighted avg"):
            results[f"f1_{ent.lower()}"] = scores["f1-score"]
            results[f"precision_{ent.lower()}"] = scores["precision"]
            results[f"recall_{ent.lower()}"] = scores["recall"]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_done_runs() -> set:
    """Runs déjà présents dans le CSV (pour reprendre après une interruption)."""
    done = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((float(row["learning_rate"]), int(row["batch_size"])))
    return done


def append_row(row: Dict):
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def best_eval_entry(log_history: List[Dict]) -> Dict:
    """Entrée d'évaluation du meilleur epoch (= checkpoint rechargé en fin
    d'entraînement), pour éviter de refaire une évaluation complète."""
    evals = [e for e in log_history if "eval_f1_macro" in e]
    return max(evals, key=lambda e: e["eval_f1_macro"])


# ─────────────────────────────────────────────────────────────────────────────
# Un run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(lr, batch_size, tokenized_train, tokenized_dev, data_collator,
            label2id, id2label):
    run_dir = TMP_DIR / f"lr{lr:.0e}_bs{batch_size}"
    model, trainer = None, None
    try:
        set_seed(SEED)
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )

        args = TrainingArguments(
            output_dir=str(run_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=100_000,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            seed=SEED,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            fp16=USE_FP16,
            bf16=USE_BF16,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, id2label),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        start = time.time()
        trainer.train()
        minutes = (time.time() - start) / 60

        best = best_eval_entry(trainer.state.log_history)

        row = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "best_epoch": best["epoch"],
            "train_minutes": round(minutes, 1),
        }
        for key, value in best.items():
            if key.startswith("eval_") and key not in NON_METRIC_KEYS:
                row[key[len("eval_"):]] = float(value)
        return row

    finally:
        trainer = None
        model = None
        shutil.rmtree(run_dir, ignore_errors=True)
        free_gpu_memory()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"GPU disponible : {torch.cuda.is_available()}")

    train_sentences = read_tsv_file(TRAIN_FILE)
    dev_sentences = read_tsv_file(DEV_FILE)
    print(f"Train: {len(train_sentences)} phrases | Dev: {len(dev_sentences)} phrases")

    label2id, id2label = create_label_mappings(train_sentences + dev_sentences)
    print(f"Labels ({len(label2id)}): {label2id}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = prepare_dataset(train_sentences, label2id)
    dev_ds = prepare_dataset(dev_sentences, label2id)

    tokenized_train = train_ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True, remove_columns=train_ds.column_names,
    )
    tokenized_dev = dev_ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True, remove_columns=dev_ds.column_names,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    grid = list(itertools.product(LEARNING_RATES, BATCH_SIZES))
    done = load_done_runs()
    print(f"\nGrille : {len(grid)} configurations, {len(done)} déjà terminées.\n")

    for i, (lr, bs) in enumerate(grid, start=1):
        if (lr, bs) in done:
            print(f"[{i}/{len(grid)}] lr={lr:.0e} bs={bs} : déjà fait, ignoré")
            continue

        print("=" * 80)
        print(f"[{i}/{len(grid)}] lr={lr:.0e} | batch_size={bs}")
        print("=" * 80)

        row = run_one(lr, bs, tokenized_train, tokenized_dev, data_collator,
                      label2id, id2label)
        append_row(row)

        print(f"\n→ meilleur epoch {row['best_epoch']:.0f} | "
              f"F1-macro {row['f1_macro']:.4f} | F1-micro {row['f1_micro']:.4f} | "
              f"{row['train_minutes']} min\n")

    print(f"\nTerminé. Résultats dans {RESULTS_CSV}")


if __name__ == "__main__":
    main()