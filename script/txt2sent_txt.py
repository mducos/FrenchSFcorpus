import os
import spacy

# charger le modèle spaCy
nlp = spacy.load("fr_dep_news_trf")

corpus2 = "NovSFcorpus"

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
    text = text.replace("\n", " ")

    # segmentation en phrases avec spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # fichier de sortie
    out_path = os.path.join(folder_path, f"{folder_name}_sent.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent + "\n")

    print(f"Fichier enregistré : {out_path}")
    
