
import spacy
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================

FEAT_DIR = "data/FeatSFcorpus"
ANNOTATION_DIR = "data/NovSFcorpus"
TITLE2NOVUM_FILE = "src/title2novum.json"

MAX_GAP = 4
NUM_COLS = 3

nlp = spacy.load("fr_core_news_sm")


# ============================================================
# LECTURE DES TSV DE FEATSFCORPUS
# ============================================================

def load_tsv(tsv_path):
    """
    Récupère les deux premières colonnes :
    - Colonne 0 : token
    - Colonne 1 : lemme

    Conserve la segmentation et les fins de phrases.
    """

    tokens = []

    with open(tsv_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.rstrip("\n\r")

            if not line:
                if tokens:
                    tokens[-1]["is_sent_end"] = True
                continue

            parts = line.split("\t")

            tokens.append({
                "text": parts[0],
                "lemma": parts[1],
                "is_sent_end": False
            })

    return tokens


# ============================================================
# RECHERCHE DES NOVUMS
# ============================================================

def token_matches(tok, target_form):

    text_lower = tok["text"].lower()
    lemma_lower = tok["lemma"].lower()

    return (
        lemma_lower == target_form
        or text_lower == target_form
        or text_lower == target_form + "s"
        or text_lower == target_form + "es"
        or text_lower.rstrip("s") == target_form
        or text_lower.rstrip("es") == target_form
    )


def find_novum_spans(tokens, novums_lemma, novums_raw, max_gap=4):

    spans = []

    # Suppression des doublons dans les listes de novums
    seen_targets = set()
    all_targets = []

    for novum in novums_lemma + novums_raw:

        key = tuple(novum.lower().split())

        if key not in seen_targets:
            seen_targets.add(key)
            all_targets.append(list(key))

    # Tri par longueur croissante
    all_targets = sorted(all_targets, key=len)

    for target in all_targets:

        target_len = len(target)
        i = 0

        while i < len(tokens):

            t_idx = 0
            span_start = None
            span_end = None
            gap_count = 0
            j = i

            while j < len(tokens) and t_idx < target_len:

                if token_matches(tokens[j], target[t_idx]):

                    if span_start is None:
                        span_start = j

                    span_end = j
                    t_idx += 1
                    gap_count = 0

                else:

                    gap_count += 1

                    if gap_count > max_gap:
                        break

                j += 1

            if t_idx == target_len:

                spans.append((
                    span_start,
                    span_end,
                    " ".join(target)
                ))

                i += 1

            else:
                i += 1

    # Suppression des occurrences strictement dupliquées
    seen = set()
    unique_spans = []

    for span in spans:

        key = (span[0], span[1])

        if key not in seen:
            seen.add(key)
            unique_spans.append(span)

    return unique_spans


# ============================================================
# ATTRIBUTION AUX TROIS COLONNES
# ============================================================

def assign_columns(token_span_indices, num_cols=3):

    col_occupied = [set() for _ in range(num_cols)]
    assignments = []

    for span_idx, token_indices in enumerate(token_span_indices):

        placed = False

        for col_idx in range(num_cols):

            # Vérifie qu'aucun token du span n'est déjà utilisé
            if not col_occupied[col_idx].intersection(token_indices):

                col_occupied[col_idx].update(token_indices)

                assignments.append((span_idx, col_idx))
                placed = True
                break

        if not placed:

            print(
                f"  ⚠ Span {span_idx} non placé : "
                "toutes les colonnes occupées"
            )

    return assignments


# ============================================================
# ÉCRITURE DU TSV FINAL
# ============================================================

def write_bio(tokens, spans, bio_path, num_cols=3):

    token_span_indices = []
    token_span_labels = []

    # Conversion des spans en indices de tokens
    for start, end, novum_text in spans:

        matched_indices = list(range(start, end + 1))

        if matched_indices:

            labels = (
                ["B-NOV"]
                + ["I-NOV"] * (len(matched_indices) - 1)
            )

            token_span_indices.append(matched_indices)
            token_span_labels.append(labels)

    # Attribution des spans aux colonnes
    assignments = assign_columns(token_span_indices, num_cols)

    # Création de la grille BIO
    grille = [
        ["O"] * len(tokens)
        for _ in range(num_cols)
    ]

    for span_idx, col_idx in assignments:

        for tok_idx, label in zip(
            token_span_indices[span_idx],
            token_span_labels[span_idx]
        ):

            grille[col_idx][tok_idx] = label

    # Écriture du TSV
    with open(bio_path, "w", encoding="utf-8") as f:

        for i, tok in enumerate(tokens):

            nov_cols = [
                grille[col][i]
                for col in range(num_cols)
            ]

            # 5 colonnes :
            # Token | Lemme | NOV1 | NOV2 | NOV3
            f.write(
                f"{tok['text']}\t"
                f"{tok['lemma']}\t"
                f"{'\t'.join(nov_cols)}\n"
            )

            # Conservation des séparateurs de phrases
            if tok["is_sent_end"]:
                f.write("\n")

        f.write("\n")


# ============================================================
# TRAITEMENT DU CORPUS
# ============================================================

with open(TITLE2NOVUM_FILE, "r", encoding="utf-8") as f:
    title2novum = json.load(f)


for subdir in os.listdir(FEAT_DIR):

    subdir_path = os.path.join(FEAT_DIR, subdir)

    if not os.path.isdir(subdir_path):
        continue

    for filename in os.listdir(subdir_path):

        if not filename.lower().endswith(".tsv"):
            continue

        title = filename[:-4]

        # Vérification de la présence du titre dans le JSON
        if title not in title2novum:

            print(f"⚠ Titre absent du JSON : {title}")
            continue

        tsv_file = os.path.join(subdir_path, filename)

        out_dir = os.path.join(ANNOTATION_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)

        bio_file = os.path.join(out_dir, filename)

        # Lecture des tokens et lemmes existants
        tokens = load_tsv(tsv_file)

        # Récupération des novums du titre
        novums = [
            x[0]
            for x in title2novum[title]
        ]

        novums_lemma = []
        novums_raw = []

        for novum in novums:

            doc = nlp(novum)

            novums_lemma.append(
                " ".join(
                    token.lemma_.lower()
                    for token in doc
                )
            )

            novums_raw.append(novum.lower())

        # Recherche des occurrences dans le texte
        spans = find_novum_spans(
            tokens,
            novums_lemma,
            novums_raw,
            max_gap=MAX_GAP
        )

        # Génération du TSV final
        write_bio(
            tokens,
            spans,
            bio_file,
            num_cols=NUM_COLS
        )

        print(f"Terminé : {title} ")

        break

print("\nTraitement du corpus terminé.")
