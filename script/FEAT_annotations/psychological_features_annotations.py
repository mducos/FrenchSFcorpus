
import json
import csv
import spacy
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

GAZETTEER_PATH = Path("src/Features_lexicons/traits_2facet_valence__FR.json")
INPUT_DIR = Path("data/CorefSFcorpus")
OUTPUT_DIR = Path("data/FeatSFcorpus")

# Modèle français spaCy
nlp = spacy.load("fr_core_news_lg")


# ============================================================
# 1. LEMMATISATION DU GAZETTEER
# ============================================================

def load_gazetteer(path):
    with open(path, "r", encoding="utf-8") as f:
        gazetteer = json.load(f)

    lemmas = set()

    # On lemmatise toutes les clés du JSON
    terms = list(gazetteer.keys())

    for doc in nlp.pipe(terms, batch_size=500):
        lemma = " ".join(
            token.lemma_.lower()
            for token in doc
            if not token.is_space
        )
        lemmas.add(lemma)

    print(f"Gazetteer chargé : {len(gazetteer)} termes")
    print(f"Nombre de lemmes uniques : {len(lemmas)}")

    return lemmas


# ============================================================
# 2. ANNOTATION DES FICHIERS TSV
# ============================================================

def annotate_tsv(input_path, output_path, gazetteer_lemmas):

    # Lecture du TSV
    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue

            token = row[0]
            ner = row[1] if "PER" in row[1] else "O"
            coref = row[2]
            rows.append((token, ner, coref))

    # Lemmatisation des tokens
    tokens = [row[0] for row in rows]

    docs = list(nlp.pipe(tokens, batch_size=1000))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        for (token, ner, coref), doc in zip(rows, docs):

            # Lemme du token
            lemma = doc[0].lemma_.lower() if len(doc) > 0 else token.lower()
            pos = doc[0].pos_ if len(doc) > 0 else "X"

            # Annotation PSY_FEAT si le lemme est dans le gazetteer
            label = "PSY_FEAT" if lemma in gazetteer_lemmas and pos == "ADJ" else "O"

            writer.writerow([
                token,
                lemma,
                pos,
                ner,
                coref,
                label
            ])


# ============================================================
# 3. TRAITEMENT DE TOUT LE CORPUS
# ============================================================

def main():

    gazetteer_lemmas = load_gazetteer(GAZETTEER_PATH)

    tsv_files = list(INPUT_DIR.rglob("*.tsv"))

    print(f"\nNombre de fichiers TSV à traiter : {len(tsv_files)}")

    for i, input_path in enumerate(tsv_files, 1):

        # Conservation de l'arborescence et du nom du fichier
        relative_path = input_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path

        annotate_tsv(
            input_path,
            output_path,
            gazetteer_lemmas
        )

        if i % 100 == 0:
            print(f"{i}/{len(tsv_files)} fichiers traités")

    print("\nTraitement terminé.")
    print(f"Résultats enregistrés dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
