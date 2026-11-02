from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pathlib import Path
from typing import List, Tuple
import re


def predict_labels(tokens: List[str], model, tokenizer, id2label) -> List[Tuple[str, str]]:

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in encoding.items() if k in ["input_ids", "attention_mask"]}

    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    
    word_ids = encoding.word_ids(batch_index=0)
    aligned_labels = []
    last_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != last_word_idx:
            aligned_labels.append(id2label[predictions[idx]])
            last_word_idx = word_idx

    assert len(aligned_labels) == len(tokens), (len(aligned_labels), len(tokens))
    return list(zip(tokens, aligned_labels))

def predict_labels_sentence(sentence: str, model, tokenizer, id2label):

    tokens = []
    for word in sentence.strip().split():
        if "'" in word:
            parts = word.split("'")
            for i, p in enumerate(parts):
                if i > 0:
                    tokens.append("'")
                if p:
                    tokens.append(p)
        else:
            tokens.append(word)
    
    return predict_labels(tokens, model, tokenizer, id2label)

def annotate_file_tsv(input_file: Path, output_file: Path, model, tokenizer, id2label):

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        current_tokens = []
        for line in f_in:
            line = line.strip()
            if not line:
                if current_tokens:
                    preds = predict_labels(current_tokens, model, tokenizer, id2label)
                    for token, label in preds:
                        f_out.write(f"{token}\t{label}\n")
                    f_out.write("\n")
                    current_tokens = []
                continue
            
            parts = line.split("\t")
            token = parts[0]
            current_tokens.append(token)
        
        if current_tokens:
            preds = predict_labels(current_tokens, model, tokenizer, id2label)
            for token, label in preds:
                f_out.write(f"{token}\t{label}\n")

# NerSFmodel
model = AutoModelForTokenClassification.from_pretrained("MathildeDucos/NerSFmodel")
tokenizer = AutoTokenizer.from_pretrained("MathildeDucos/NerSFmodel")

id2label = model.config.id2label

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# file to annotate
input_file = Path("data/SFcorpus/Capus_LHommeBicycle_1893/Capus_LHommeBicycle_1893_sent.txt")
output_file = Path("Capus_LHommeBicycle_1893.tsv")

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        line = re.sub(r"([?.!,';:()])", r" \1 ", line).strip()

        result = predict_labels_sentence(line, model, tokenizer, id2label)
        for token, label in result:
            f_out.write(f"{token}\t{label}\n")
        
        f_out.write("\n")
