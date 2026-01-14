import re
import spacy
from pathlib import Path
import json
import os


# -----------------------------------------------------------
# 1. Tokenisation + lemmatisation
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
# 2. Détection des novums lemmatisés dans le texte
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
# 3. Génération du fichier .bio
# -----------------------------------------------------------

def write_bio_sentence(tokens, spans, f, n_cols=3):
    """
    spans = liste (start, end, text)
    tokens = tokens spaCy enrichis (text, lemma, pos, start, end)
    Génère n colonnes BIO pour gérer les chevauchements.
    """

    bio_labels = [["O"] * len(tokens) for _ in range(n_cols)]

    for start, end, novum_text in spans:
        novum_lemmas = [tok.lemma_.lower() for tok in nlp(novum_text)]

        matched = []
        t_idx = 0

        for i, tok in enumerate(tokens):
            if t_idx >= len(novum_lemmas):
                break
            if tok["lemma"] == novum_lemmas[t_idx]:
                matched.append(i)
                t_idx += 1

        if not matched:
            continue

        for col in range(n_cols):
            if all(bio_labels[col][i] == "O" for i in matched):
                bio_labels[col][matched[0]] = "B-NOV"
                for i in matched[1:]:
                    bio_labels[col][i] = "I-NOV"
                break

    for i, tok in enumerate(tokens):
        labels = "\t".join(bio_labels[col][i] for col in range(n_cols))
        f.write(f"{tok['text']}\t{tok['lemma']}\t{tok['pos']}\t{labels}\n")

    f.write("\n")


# chargement de spaCy FR avec lemmatisation
nlp = spacy.load("fr_dep_news_trf")

# parcours ses fichiers du dossier
for dirname in os.listdir("NovSFcorpus"):
    for filename in os.listdir("NovSFcorpus\\"+dirname):
        if filename.lower().endswith("_sent.txt"):
            title = filename[:-9]

            # -----------------------------------------------------------
            # CONFIGURATION
            # -----------------------------------------------------------

            TEXT_FILE = "NovSFcorpus\\" + dirname + "\\" + filename
            BIO_FILE = "NovSFcorpus\\" + dirname + "\\" + title + ".tsv"
            TITLE2NOVUM_FILE = "src\\title2novum.json"
            with open(TITLE2NOVUM_FILE, "r", encoding="utf-8") as f:
                title2novum = json.load(f)

            # récupération des novum en excluant les novum type 4
            novums = title2novum[title]
            novums = [x[0] for x in novums if x[1] != 4]
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

            sentences = raw_text.split("\n")

            with open(BIO_FILE, "w", encoding="utf-8") as f_out:
                for sentence in sentences:
                    if not sentence.strip():
                        f_out.write("\n")
                        continue

                    tokens = tokenize(sentence)
                    spans = find_novum_spans(tokens, NOVUMS_LEMMA)

                    write_bio_sentence(tokens, spans, f_out)

            print(f"Génération du fichier : {title}.tsv")