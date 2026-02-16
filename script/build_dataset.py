import os
import random
from pathlib import Path

def write_tsv(sentences, path):
    """
    sentences : liste de phrases
    chaque phrase = liste de lignes TSV complètes (token + annotations)
    """
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for line in sent:
                f.write(line + "\n")  # ligne complète avec toutes les colonnes
            f.write("\n")  # saut de ligne entre phrases

def oversample_nov(sentences, factor=5):
    """
    Duplique les phrases contenant B-NOV un certain nombre de fois.
    
    Args:
        sentences: liste de phrases
        factor: nombre total de copies (5 = 1 originale + 4 duplicatas)
    
    Returns:
        Liste de phrases avec duplication des phrases contenant NOV
    """
    result = []
    nov_count = 0
    
    for sent in sentences:
        # Vérifier si la phrase contient B-NOV
        has_nov = False
        for line in sent:
            parts = line.split('\t')
            if len(parts) >= 2 and parts[1].strip() == 'B-NOV':
                has_nov = True
                break
        
        if has_nov:
            # Dupliquer cette phrase 'factor' fois
            for _ in range(factor):
                result.append(sent)
            nov_count += 1
        else:
            # Garder telle quelle
            result.append(sent)
    
    print(f"  → {nov_count} phrases avec NOV dupliquées {factor} fois")
    return result

def oversample_org(sentences, factor=5):
    """
    Duplique les phrases contenant B-NOV un certain nombre de fois.
    
    Args:
        sentences: liste de phrases
        factor: nombre total de copies (5 = 1 originale + 4 duplicatas)
    
    Returns:
        Liste de phrases avec duplication des phrases contenant NOV
    """
    result = []
    nov_count = 0
    
    for sent in sentences:
        # Vérifier si la phrase contient B-NOV
        has_nov = False
        for line in sent:
            parts = line.split('\t')
            if len(parts) >= 2 and parts[1].strip() == 'B-ORG':
                has_nov = True
                break
        
        if has_nov:
            # Dupliquer cette phrase 'factor' fois
            for _ in range(factor):
                result.append(sent)
            nov_count += 1
        else:
            # Garder telle quelle
            result.append(sent)
    
    print(f"  → {nov_count} phrases avec ORG dupliquées {factor} fois")
    return result

# Répertoire de base
NER_DIR = Path("data/NerSFcorpus")

# Liste pour toutes les phrases (chaque phrase = liste de lignes TSV)
all_sentences = []

# Parcours tous les sous-dossiers
for subdir in NER_DIR.iterdir():
    if not subdir.is_dir():
        continue
    for tsv_file in subdir.glob("*.tsv"):
        with open(tsv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        phrase = []
        for line in lines:
            if line.strip() == "":
                if phrase:  # fin de phrase
                    all_sentences.append(phrase)
                    phrase = []
            else:
                phrase.append(line.rstrip("\n"))  # ligne complète
        if phrase:  # dernière phrase du fichier
            all_sentences.append(phrase)

# Shuffle pour split aléatoire
random.seed(42)
random.shuffle(all_sentences)

# ----------------------------
# Split 80/10/10
# ----------------------------
n = len(all_sentences)
n_train = int(0.8 * n)
n_dev = int(0.1 * n)

train_sents = all_sentences[:n_train]
dev_sents = all_sentences[n_train:n_train + n_dev]
test_sents = all_sentences[n_train + n_dev:]

# ----------------------------
# Affichage du nombre de phrases
# ----------------------------
print(f"Nombre total de phrases : {n}")
print(f"Train : {len(train_sents)} phrases")
print(f"Dev   : {len(dev_sents)} phrases")
print(f"Test  : {len(test_sents)} phrases")

# ----------------------------
# Suréchantillonnage des NOV dans train
# ----------------------------
print("\nSuréchantillonnage des phrases avec NOV dans train:")
train_subset = train_sents#[:240500]
train_oversampled = oversample_nov(train_subset, factor=10)
print(f"Train avant oversampling: {len(train_subset)} phrases")
print(f"Train après oversampling: {len(train_oversampled)} phrases")

# ----------------------------
# Écriture des fichiers TSV
# ----------------------------
output_dir = Path("src")
output_dir.mkdir(parents=True, exist_ok=True)

write_tsv(train_oversampled, output_dir / "train.tsv")
write_tsv(dev_sents, output_dir / "dev.tsv")
write_tsv(test_sents, output_dir / "test.tsv")

print(f"\nFichiers créés dans {output_dir}")