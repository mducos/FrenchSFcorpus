import torch
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import numpy as np
import torch.nn as nn

class WeightedTrainer(Trainer):

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Loss avec poids de classes
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                ignore_index=-100
            )
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

class FocalLoss(torch.nn.Module):

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        targets = targets.view(-1)
        mask = targets != self.ignore_index

        logits = logits[mask]
        targets = targets[mask]
        log_probs = log_probs[mask]
        probs = probs[mask]

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0

        loss = -alpha_t * (1-target_probs)**self.gamma*target_log_probs
        return loss.mean()

class FocalTrainer(Trainer):

    def __init__(self, *args, alpha=None, gamma=2.0, **kwargs):
        super().__init__(*args,**kwargs)
        self.focal_loss = FocalLoss(
                alpha = alpha.to(self.model.device) if alpha is not None else None,
                gamma = gamma)

    def comput_loss(self, model, inputs, num_items_in_batch=None,return_outputs=False):
        labels=inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.focal_loss(
                logits.view(-1, logits.size(-1)),
                labels.view(-1))

        return (loss, outputs) if return_outputs else loss

alpha = torch.tensor([20,40,480,150,10,15,65,440,90,15,1], dtype=torch.float)
gamma = 2.15

def read_tsv_file(file_path: Path) -> List[Tuple[List[str], List[str]]]:
    """
    Reads a TSV file and returns a list of sentences.
    Each sentence is a tuple (tokens, tags).
    """
    sentences = []
    current_tokens = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:  # empty line = end of sentence
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, tag = parts
                    current_tokens.append(token)
                    current_tags.append(tag.strip())

        if current_tokens:
            sentences.append((current_tokens, current_tags))

    return sentences

def create_label_mappings(sentences: List[Tuple[List[str], List[str]]]) -> Tuple[Dict, Dict]:

    unique_labels = set()
    for _, tags in sentences:
        unique_labels.update(tags)

    sorted_labels = sorted(list(unique_labels))

    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    return label2id, id2label

def prepare_dataset(sentences: List[Tuple[List[str], List[str]]], label2id: Dict) -> Dataset:

    tokens_list = []
    tags_list = []

    for tokens, tags in sentences:
        tokens_list.append(tokens)
        tags_list.append([label2id[tag] for tag in tags])

    dataset = Dataset.from_dict({
        'tokens': tokens_list,
        'ner_tags': tags_list
    })

    return dataset

def tokenize_and_align_labels(examples, tokenizer, label2id):

    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=64
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                # special token (CLS, SEP, PAD)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def compute_metrics(eval_pred, id2label):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        true_label = []
        true_prediction = []

        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                true_label.append(id2label[label_id])
                true_prediction.append(id2label[pred_id])

        if true_label:
            true_labels.append(true_label)
            true_predictions.append(true_prediction)

    results = {
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'f1': f1_score(true_labels, true_predictions),
        'accuracy': accuracy_score(true_labels, true_predictions),
    }

    return results

def evaluate_and_print_results(trainer, dataset, id2label, dataset_name="Test"):

    print(f"{'='*80}")
    print(f"Évaluation sur {dataset_name}")
    print(f"{'='*80}\n")
    results = f"\n{'='*80}\nÉvaluation sur {dataset_name}\n{'='*80}\n\n"

    # Prédictions
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=2)
    labels = predictions.label_ids

    # Convertir en format seqeval
    true_labels = []
    true_predictions = []

    for prediction, label in zip(preds, labels):
        true_label = []
        true_prediction = []

        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                true_label.append(id2label[label_id])
                true_prediction.append(id2label[pred_id])

        if true_label:
            true_labels.append(true_label)
            true_predictions.append(true_prediction)

    # Classification report détaillé
    print(classification_report(true_labels, true_predictions, digits=4))
    results = results + classification_report(true_labels, true_predictions, digits=4)

    # Métriques globales
    print(f"\n{'='*80}")
    print("MÉTRIQUES GLOBALES")
    print(f"{'='*80}")
    print(f"Accuracy:        {accuracy_score(true_labels, true_predictions):.4f}")
    print(f"Precision:       {precision_score(true_labels, true_predictions):.4f}")
    print(f"Recall:          {recall_score(true_labels, true_predictions):.4f}")
    print(f"F1-Score:        {f1_score(true_labels, true_predictions):.4f}")
    results = results + f"\n{'='*80}\nMÉTRIQUES GLOBALES\n{'='*80}\nAccuracy:        {accuracy_score(true_labels, true_predictions):.4f}\nPrecision:       {precision_score(true_labels, true_predictions):.4f}\nRecall:          {recall_score(true_labels, true_predictions):.4f}\nF1-Score:        {f1_score(true_labels, true_predictions):.4f}"

    # F1 micro et macro
    f1_micro = f1_score(true_labels, true_predictions, average='micro')
    f1_macro = f1_score(true_labels, true_predictions, average='macro')

    print(f"F1-Score (micro): {f1_micro:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    print(f"{'='*80}\n")
    results = results + f"\nF1-Score (micro): {f1_micro:.4f}\nF1-Score (macro): {f1_macro:.4f}\n{'='*80}\n\n"

    with open("results.txt", "a+", encoding='utf-8') as f:
        f.write(results)

def main():

    print(f"GPU disponible : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU utilisé : {torch.cuda.get_device_name(0)}")
        print(f"VRAM totale : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    train_file = Path("src/train.tsv")
    dev_file = Path("src/dev.tsv")
    test_file = Path("src/test.tsv")

    print("Chargement des données...")
    train_sentences = read_tsv_file(train_file)
    dev_sentences = read_tsv_file(dev_file)
    test_sentences = read_tsv_file(test_file)

    print(f"Train: {len(train_sentences)} phrases")
    print(f"Dev: {len(dev_sentences)} phrases")
    print(f"Test: {len(test_sentences)} phrases")

    all_sentences = train_sentences + dev_sentences + test_sentences
    label2id, id2label = create_label_mappings(all_sentences)

    print(f"\nNombre de labels: {len(label2id)}")
    print(label2id)

    train_dataset = prepare_dataset(train_sentences, label2id)
    dev_dataset = prepare_dataset(dev_sentences, label2id)
    test_dataset = prepare_dataset(test_sentences, label2id)

    model_name = "camembert-base"
    print(f"\nChargement du modèle {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    print("Tokenization des données...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_dev = dev_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
        remove_columns=dev_dataset.column_names
    )

    tokenized_test = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
        remove_columns=test_dataset.column_names
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="src/camembert_ner",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=20,
        warmup_ratio=0,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )

    trainer = FocalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        alpha=alpha,
        gamma=gamma,
        compute_metrics=lambda x: compute_metrics(x, id2label),
    )

    print("\nDébut de l'entraînement...")
    trainer.train()

    evaluate_and_print_results(trainer, tokenized_dev, id2label, "Dev Set")

    output_dir = Path("src/camembert_ner_final")
    print(f"\nSauvegarde du modèle dans {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✓ Entraînement terminé!")

if __name__ == "__main__":
    main()
