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
# 4. Génération du fichier .ann
# -----------------------------------------------------------

def write_ann(spans, ann_path):
    with open(ann_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(spans, start=1):
            f.write(f"T{i}\tNOVUM {start} {end}\t{text}\n")


# chargement de spaCy FR avec lemmatisation
nlp = spacy.load("fr_core_news_sm")

# parcours ses fichiers du dossier
for dirname in os.listdir("NovSFcorpus"):
    for filename in os.listdir("NovSFcorpus\\"+dirname):
        if not(filename.lower().endswith("_sent.txt")) and filename.lower().endswith(".txt"):
            title = filename[:-4]

            # -----------------------------------------------------------
            # CONFIGURATION
            # -----------------------------------------------------------

            TEXT_FILE = "NovSFcorpus\\" + dirname + "\\" + filename
            ANN_FILE = "NovSFcorpus\\" + dirname + "\\" + title + ".ann"
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

            # génération du fichier .ann
            write_ann(spans, ANN_FILE)

            print(f"Génération du fichier : {title}.ann")