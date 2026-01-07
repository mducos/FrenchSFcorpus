import re
import spacy
from pathlib import Path
import json
import os


# -----------------------------------------------------------
# 1. Nettoyage du texte
# -----------------------------------------------------------

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

# -----------------------------------------------------------
# 2. Tokenisation + lemmatisation
# -----------------------------------------------------------

def tokenize(text):
    """
    Retourne :
    - tokens : liste des tokens avec leurs caractéristiques spacy
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        tokens.append({
            "text": token.text,
            "pos": token.pos_,
            "lemma": token.lemma_.lower(),
            "start": token.idx,
            "end": token.idx + len(token.text)
        })
    return tokens


# -----------------------------------------------------------
# 3. Détection des novums lemmatisés dans le texte
# -----------------------------------------------------------

def find_novum_spans(tokens, novums_lemma, max_gap=4):
    spans = []

    # mettre les novums sous forme de listes de lemmes
    novum_lemmas = [novum.lower().split() for novum in novums_lemma]

    for target in novum_lemmas:
        target_len = len(target)
        i = 0
        while i < len(tokens):
            t_idx = 0  # index dans le novum
            span_start = None
            span_end = None
            gap_count = 0
            j = i
            while j < len(tokens) and t_idx < target_len:
                if tokens[j]["lemma"].lower() == target[t_idx]:
                    if span_start is None:
                        span_start = tokens[j]["start"]
                    span_end = tokens[j]["end"]
                    t_idx += 1
                    gap_count = 0
                else:
                    gap_count += 1
                    if gap_count > max_gap:
                        break
                j += 1

            if t_idx == target_len:
                # on a trouvé le novum
                spans.append((span_start, span_end, " ".join(target)))
                i = j  # avancer après la fin du novum trouvé
            else:
                i += 1

    return spans


# -----------------------------------------------------------
# 4. Génération du fichier .bio
# -----------------------------------------------------------

def write_bio(tokens, spans, bio_path, n_cols=3):
    """
    spans = liste (start, end, text)
    tokens = tokens spaCy enrichis (text, lemma, pos, start, end)
    Génère n colonnes BIO pour gérer les chevauchements.
    """

    # initialisation des colonnes BIO
    bio_labels = [["O"] * len(tokens) for _ in range(n_cols)]

    for start, end, novum_text in spans:
        novum_lemmas = [tok.lemma_.lower() for tok in nlp(novum_text)]

        matched_indices = []
        t_idx = 0

        for j, tok in enumerate(tokens):
            if t_idx >= len(novum_lemmas):
                break
            if (
                tok["lemma"].lower() == novum_lemmas[t_idx]
                and tok["start"] >= start
                and tok["end"] <= end
            ):
                matched_indices.append(j)
                t_idx += 1

        if not matched_indices:
            continue

        # recherche d'une colonne libre
        for col in range(n_cols):
            if all(bio_labels[col][i] == "O" for i in matched_indices):
                bio_labels[col][matched_indices[0]] = "B-NOV"
                for idx in matched_indices[1:]:
                    bio_labels[col][idx] = "I-NOV"
                break
        # sinon : novum ignoré (plus de colonnes disponibles mais ce cas n'arrive pas dans le corpus)

    # écriture du fichier
    with open(bio_path, "w", encoding="utf-8") as f:
        for i, tok in enumerate(tokens):
            labels = "\t".join(bio_labels[col][i] for col in range(n_cols))
            f.write(
                f"{tok['text']}\t{tok['lemma']}\t{tok['pos']}\t{labels}\n"
            )
            if tok["text"] in {".", "!", "?"}:
                f.write("\n")
        f.write("\n")


# chargement de spaCy FR avec lemmatisation
nlp = spacy.load("fr_dep_news_trf")

# parcours ses fichiers du dossier
for dirname in os.listdir("NovSFcorpus"):
    for filename in os.listdir("NovSFcorpus\\"+dirname):
        if not(filename.lower().endswith("_sent.txt")) and filename.lower().endswith(".txt"):
            title = filename[:-4]

            # -----------------------------------------------------------
            # CONFIGURATION
            # -----------------------------------------------------------

            TEXT_FILE = "NovSFcorpus\\" + dirname + "\\" + filename
            BIO_FILE = "NovSFcorpus\\" + dirname + "\\" + title + ".tsv"
            TITLE2NOVUM_FILE = "src\\title2novum.json"
            with open(TITLE2NOVUM_FILE, "r", encoding="utf-8") as f:
                title2novum = json.load(f)

            novums = title2novum[title]
            novums = [x[0] for x in novums]
            NOVUMS_LEMMA = []
            NOVUMS_POS = []

            # liste des novums lemmatisés (multi-mots possibles)
            for novum in novums:
                doc = nlp(novum)
                tmp_lemma = []
                tmp_pos = []
                for token in doc:
                    tmp_lemma.append(token.lemma_.lower())
                    tmp_pos.append(token.pos_)
                NOVUMS_LEMMA.append(" ".join(tmp_lemma))
                NOVUMS_POS.append(" ".join(tmp_pos))

            # -----------------------------------------------------------
            # PIPELINE
            # -----------------------------------------------------------

            # chargement du texte original
            raw_text = Path(TEXT_FILE).read_text(encoding="utf-8")

            # nettoyage
            cleaned_text = clean_text(raw_text)

            # tokenisation + lemmatisation
            tokens = tokenize(cleaned_text)

            # recherche des novums par lemmes
            spans = find_novum_spans(tokens, NOVUMS_LEMMA)

            # génération du fichier .bio
            write_bio(tokens, spans, BIO_FILE)

            print(f"Génération du fichier : {title}.tsv")