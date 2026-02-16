from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pathlib import Path
from typing import List, Tuple
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from datasets import Dataset
import numpy as np

def prepare_dataset(sentences, label2id):

    tokens_list = [tokens for tokens, _ in sentences]
    tags_list = [[label2id[tag] for tag in tags] for _, tags in sentences]
    return Dataset.from_dict({"tokens": tokens_list, "ner_tags": tags_list})

def read_tsv_file(file_path: Path) -> List[Tuple[List[str], List[str]]]:

    sentences = []
    current_tokens = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:  # Empty line = end of sentence
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

model_checkpoint = AutoModelForTokenClassification.from_pretrained("src/camembert_ner_final")
tokenizer_checkpoint = AutoTokenizer.from_pretrained("src/camembert_ner_final")

label2id = model_checkpoint.config.label2id
id2label = model_checkpoint.config.id2label

# to reproduce the results on the test set
test_sentences = read_tsv_file(Path("src/test.tsv"))
# to compare the results with a book
#test_sentences = read_tsv_file(Path("data/NerSFcorpus/JehinPrume_LesAventuresExtraordinairesDeDeuxCanayens_1918/JehinPrume_LesAventuresExtraordinairesDeDeuxCanayens_1918.tsv"))
test_dataset = prepare_dataset(test_sentences, label2id)

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=64,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenize_fn = lambda x: tokenize_and_align_labels(x, tokenizer_checkpoint, label2id)
tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=test_dataset.column_names)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer_checkpoint)

dummy_args = TrainingArguments(
    output_dir="./tmp",
    per_device_eval_batch_size=32,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model_checkpoint,
    args=dummy_args,
    data_collator=data_collator,
)

print("="*80)
print("REPRODUCTION EXACTE DU CALLBACK sur TEST SET")
print("="*80)

out = trainer.predict(tokenized_test)
preds = np.argmax(out.predictions, axis=2)
labels = out.label_ids

true_labels = []
true_preds = []

for pred_seq, label_seq in zip(preds, labels):
    tl, tp = [], []
    for pid, lid in zip(pred_seq, label_seq):
        if lid != -100:
            tl.append(id2label[lid])
            tp.append(id2label[pid])
    if tl:
        true_labels.append(tl)
        true_preds.append(tp)

print(classification_report(true_labels, true_preds, digits=4))
print(f"\nF1-micro: {f1_score(true_labels, true_preds, average='micro'):.4f}")
print(f"F1-macro: {f1_score(true_labels, true_preds, average='macro'):.4f}")
print(f"Precision: {precision_score(true_labels, true_preds):.4f}")
print(f"Recall: {recall_score(true_labels, true_preds):.4f}")