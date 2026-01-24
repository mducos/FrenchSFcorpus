import os
import spacy
import re
from pathlib import Path

# charger le modèle spaCy
nlp = spacy.load("fr_dep_news_trf")

corpus2 = "data/SFcorpus"

def clean_text(text):
    # remplacement des retours ligne par un espace
    text = re.sub(r"\n", " ", text)

    # normalisation des espaces
    text = re.sub(r"[ \t]+", " ", text).strip()

    # suppression de la mise en forme italique ou gras
    text = re.sub(r"\*", "", text)
    text = re.sub(r"\_","",text)

    # normalisation des caractères spéciaux
    text = re.sub(r"’", "'", text)
    text = re.sub(r"´","",text)
    text = re.sub(r"œ","oe",text)

    # séparation des mots avec tirets
    text = re.sub(r"\-"," - ",text)

    text = re.sub(r"'","' ",text)

    return text

def merge_lines(file_path: str):
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    merged = []

    for line in lines:
        stripped = line.lstrip()

        # commence par une minuscule
        starts_lower = stripped[:1].islower()

        # commence par de la ponctuation (guillemets, …, ponctuation Unicode incluse)
        starts_punct = bool(re.match(r"[\"'«»“”‘’.,;:!?…()\[\]{}-]", stripped))

        # la ligne précédente ne se termine pas par une ponctuation forte
        prev_no_strong_punct = (
            merged
            and not re.search(r"[.!?…]$", merged[-1].rstrip())
        )

        if merged and (starts_punct or (starts_lower and prev_no_strong_punct)):
            merged[-1] = merged[-1].rstrip() + " " + stripped
        else:
            merged.append(line)

    return "\n".join(merged)

for folder_name in os.listdir(corpus2):
    folder_path = os.path.join(corpus2, folder_name)

    if not os.path.isdir(folder_path):
        continue

    # récupérer le fichier txt source
    txt_files = [f for f in os.listdir(folder_path)
                 if f.endswith(".txt") and not f.endswith("_sent.txt")]

    if not txt_files:
        continue

    txt_path = os.path.join(folder_path, txt_files[0])

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_text(text)
    text = merge_lines(text)

    # segmentation en phrases avec spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # fichier de sortie
    out_path = os.path.join(folder_path, f"{folder_name}_sent.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent + "\n")

    print(f"Fichier enregistré : {out_path}")
    
