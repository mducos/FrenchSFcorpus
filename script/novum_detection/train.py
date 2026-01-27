import torch
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict
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
    """Trainer avec pondération des classes."""
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

def read_tsv_file(file_path: Path) -> List[Tuple[List[str], List[str]]]:
    """
    Lit un fichier TSV et retourne une liste de phrases.
    Chaque phrase est un tuple (tokens, tags).
    """
    sentences = []
    current_tokens = []
    current_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:  # Ligne vide = fin de phrase
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
        
        # Ajouter la dernière phrase si elle existe
        if current_tokens:
            sentences.append((current_tokens, current_tags))
    
    return sentences

def create_label_mappings(sentences: List[Tuple[List[str], List[str]]]) -> Tuple[Dict, Dict]:
    """Crée les mappings label <-> id."""
    unique_labels = set()
    for _, tags in sentences:
        unique_labels.update(tags)
    
    # Trier pour avoir un ordre cohérent
    sorted_labels = sorted(list(unique_labels))
    
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label

def compute_class_weights(sentences: List[Tuple[List[str], List[str]]], label2id: Dict) -> torch.Tensor:
    """Calcule les poids de classes pour gérer le déséquilibre."""
    from collections import Counter
    
    all_tags = []
    for _, tags in sentences:
        all_tags.extend([label2id[tag] for tag in tags])
    
    counts = Counter(all_tags)
    total = sum(counts.values())
    
    # Calculer les poids inversement proportionnels à la fréquence
    weights = torch.ones(len(label2id))
    for label_id, count in counts.items():
        weights[label_id] = total / (len(label2id) * count)
    
    return weights

def prepare_dataset(sentences: List[Tuple[List[str], List[str]]], label2id: Dict) -> Dataset:
    """Convertit les phrases en format Dataset HuggingFace."""
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
    """Tokenize et aligne les labels avec les sous-tokens."""
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
                # Token spécial (CLS, SEP, PAD)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Premier sous-token d'un mot
                label_ids.append(label[word_idx])
            #else:
                # Sous-tokens suivants : on met -100 pour les ignorer
                #label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def compute_metrics(eval_pred, id2label):
    """Calcule les métriques avec seqeval."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Convertir les prédictions et labels en format seqeval
    true_labels = []
    true_predictions = []
    
    for prediction, label in zip(predictions, labels):
        true_label = []
        true_prediction = []
        
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                true_label.append(id2label[label_id])
                true_prediction.append(id2label[pred_id])
        
        if true_label:  # Ignorer les séquences vides
            true_labels.append(true_label)
            true_predictions.append(true_prediction)
    
    # Calculer les métriques
    results = {
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'f1': f1_score(true_labels, true_predictions),
        'accuracy': accuracy_score(true_labels, true_predictions),
    }
    
    return results

def evaluate_and_print_results(trainer, dataset, id2label, dataset_name="Test"):
    """Évalue sur un dataset et affiche les résultats détaillés."""

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

    # Vérifier le GPU
    print(f"GPU disponible : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU utilisé : {torch.cuda.get_device_name(0)}")
        print(f"VRAM totale : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
   
    # Chemins des fichiers
    train_file = Path("script/novum_detection/train.tsv")
    dev_file = Path("script/novum_detection/dev.tsv")
    test_file = Path("script/novum_detection/test.tsv")
    
    # Charger les données
    print("Chargement des données...")
    train_sentences = read_tsv_file(train_file)
    dev_sentences = read_tsv_file(dev_file)
    test_sentences = read_tsv_file(test_file)
    
    print(f"Train: {len(train_sentences)} phrases")
    print(f"Dev: {len(dev_sentences)} phrases")
    print(f"Test: {len(test_sentences)} phrases")
    
    # Créer les mappings de labels
    all_sentences = train_sentences + dev_sentences + test_sentences
    label2id, id2label = create_label_mappings(all_sentences)
    
    print(f"\nNombre de labels: {len(label2id)}")
    print(label2id)
    
    # Préparer les datasets
    train_dataset = prepare_dataset(train_sentences, label2id)
    dev_dataset = prepare_dataset(dev_sentences, label2id)
    test_dataset = prepare_dataset(test_sentences, label2id)
    
    # Charger le tokenizer et le modèle
    model_name = "camembert-base"
    print(f"\nChargement du modèle {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Calculer les poids de classes
    '''
    class_weights = compute_class_weights(train_sentences, label2id)
    print(f"\nPoids des classes:")
    for label, weight in zip(sorted(label2id.keys()), class_weights):
        print(f"  {label}: {weight:.4f}")
    '''
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenizer les datasets
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

    print("\nVérification présence NOV dans train tokenisé:")
    nov_count = 0
    for example in tokenized_train:
        for label in example['labels']:
            if label == label2id.get('B-NOV'):
                nov_count += 1
    print(f"Nombre de tokens NOV dans train tokenisé: {nov_count}")

    print("\nVérification présence ORG dans train tokenisé:")
    nov_count = 0
    for example in tokenized_train:
        for label in example['labels']:
            if label == label2id.get('B-ORG'):
                nov_count += 1
    print(f"Nombre de tokens ORG dans train tokenisé: {nov_count}")
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir="src/camembert-ner",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=1,
        #warmup_ratio=0.1,  # 10% du training pour warmup
        #lr_scheduler_type="cosine",  # Décroissance progressive
        logging_dir='./logs',
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )
    
    # Trainer
    trainer = Trainer( # Pas WeightedTrainer
        #class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, id2label)
    )
    
    # Entraînement
    print("\nDébut de l'entraînement...")
    trainer.train()
    
    # Évaluation sur dev
    evaluate_and_print_results(trainer, tokenized_dev, id2label, "Dev Set")
    
    # Évaluation sur test
    evaluate_and_print_results(trainer, tokenized_test, id2label, "Test Set")
    
    # Sauvegarder le modèle final
    output_dir = Path("src/camembert-ner-final")
    print(f"\nSauvegarde du modèle dans {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✓ Entraînement terminé!")

if __name__ == "__main__":
    main()