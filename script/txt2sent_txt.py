import os
import spacy
import re

# charger le modèle spaCy
nlp = spacy.load("fr_dep_news_trf")

corpus2 = "NovSFcorpus"

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

    return text

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

    # segmentation en phrases avec spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # fichier de sortie
    out_path = os.path.join(folder_path, f"{folder_name}_sent.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent + "\n")

    print(f"Fichier enregistré : {out_path}")
    
