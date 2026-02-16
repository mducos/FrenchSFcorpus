import os
import random
from pathlib import Path

def write_tsv(sentences, path):

    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for line in sent:
                f.write(line + "\n")
            f.write("\n") # empty line = end of sentence

def oversample_nov(sentences, factor=5):
    """
    Duplicate sentences containing B-NOV a certain number of times.
    
    Args:
        sentences: list of sentences
        factor: total number of copies (5 = 1 original + 4 duplicates)
    
    Returns:
        List of sentences with duplication of sentences containing NOV
    """
    result = []
    nov_count = 0
    
    for sent in sentences:
        has_nov = False
        for line in sent:
            parts = line.split('\t')
            if len(parts) >= 2 and parts[1].strip() == 'B-NOV':
                has_nov = True
                break
        
        if has_nov:
            for _ in range(factor):
                result.append(sent)
            nov_count += 1
        else:
            result.append(sent)
    
    print(f"  → {nov_count} phrases avec NOV dupliquées {factor} fois")
    return result

NER_DIR = Path("data/NerSFcorpus")

all_sentences = []
for subdir in NER_DIR.iterdir():
    if not subdir.is_dir():
        continue
    for tsv_file in subdir.glob("*.tsv"):
        with open(tsv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        phrase = []
        for line in lines:
            if line.strip() == "":
                if phrase: # end of sentence
                    all_sentences.append(phrase)
                    phrase = []
            else:
                phrase.append(line.rstrip("\n"))
        if phrase:
            all_sentences.append(phrase)

random.seed(42)
random.shuffle(all_sentences)

# split 80/10/10
n = len(all_sentences)
n_train = int(0.8 * n)
n_dev = int(0.1 * n)

train_sents = all_sentences[:n_train]
dev_sents = all_sentences[n_train:n_train + n_dev]
test_sents = all_sentences[n_train + n_dev:]

print(f"Nombre total de phrases : {n}")
print(f"Train : {len(train_sents)} phrases")
print(f"Dev   : {len(dev_sents)} phrases")
print(f"Test  : {len(test_sents)} phrases")

# oversampling
print("\nSuréchantillonnage des phrases avec NOV dans train:")
train_subset = train_sents
train_oversampled = oversample_nov(train_subset, factor=10)
print(f"Train avant oversampling: {len(train_subset)} phrases")
print(f"Train après oversampling: {len(train_oversampled)} phrases")

output_dir = Path("src")
output_dir.mkdir(parents=True, exist_ok=True)

write_tsv(train_oversampled, output_dir / "train.tsv")
write_tsv(dev_sents, output_dir / "dev.tsv")
write_tsv(test_sents, output_dir / "test.tsv")

print(f"\nFichiers créés dans {output_dir}")