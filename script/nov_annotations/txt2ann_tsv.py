import spacy
from pathlib import Path
import json
import os

def tokenize(text):
    all_tokens = []
    offset = 0
    for line in text.splitlines():
        if line.strip():
            doc = nlp(line)
            for token in doc:
                all_tokens.append({
                    "text": token.text,
                    "pos": token.pos_,
                    "lemma": token.lemma_.lower(),
                    "text_lower": token.text.lower(), 
                    "start": token.idx + offset,
                    "end": token.idx + len(token.text) + offset,
                    "is_sent_end": False
                })
            if all_tokens:
                all_tokens[-1]["is_sent_end"] = True
        offset += len(line) + 1
    return all_tokens

def find_novum_spans(tokens, novums_lemma, novums_raw, max_gap=4):
    spans = []

    def token_matches(tok, target_form):
        return (tok["lemma"].lower() == target_form              # lemma == lemma
                or tok["text_lower"] == target_form              # raw form == lemma
                or tok["text_lower"] == target_form + "s"        # singular == plural
                or tok["text_lower"] == target_form + "es"       # singular == plural
                or tok["text_lower"].rstrip("s") == target_form  # plural == singular
                or tok["text_lower"].rstrip("es") == target_form # plural == singular
                )

    seen_targets = set()
    all_targets = []
    for novum in novums_lemma + novums_raw:
        key = tuple(novum.lower().split())
        if key not in seen_targets:
            seen_targets.add(key)
            all_targets.append(list(key))

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
                spans.append((span_start, span_end, " ".join(target)))
                i += 1
            else:
                i += 1

    # Remove duplicates
    seen = set()
    unique_spans = []
    for span in spans:
        key = (span[0], span[1])
        if key not in seen:
            seen.add(key)
            unique_spans.append(span)

    return unique_spans

def write_ann(spans, ann_path, raw_text):
    with open(ann_path, "w", encoding="utf-8") as f:
        for i, (start, end, _) in enumerate(spans, start=1):
            surface_form = raw_text[start:end]
            f.write(f"T{i}\tNOV {start} {end}\t{surface_form}\n")

def assign_columns(token_span_indices, num_cols=3):

    # col_occupied[c] = set of token indices already taken in column c
    col_occupied = [set() for _ in range(num_cols)]
    assignments = []

    for span_idx, token_indices in enumerate(token_span_indices):
        placed = False
        for col_idx in range(num_cols):
            # Check that no token from the span has already been taken in this column
            if not col_occupied[col_idx].intersection(token_indices):
                for ti in token_indices:
                    col_occupied[col_idx].add(ti)
                assignments.append((span_idx, col_idx))
                placed = True
                break
        if not placed:
            # No more columns available: ignore
            print(f"  ⚠ Span {span_idx} non placé : toutes les colonnes occupées")

    return assignments

def write_bio(tokens, spans, bio_path, num_cols=3):
    token_span_indices = []
    token_span_labels = []

    for start, end, novum_text in spans:
        # Use offsets directly rather than re-lemmatizing
        matched_indices = [
            j for j, tok in enumerate(tokens)
            if tok["start"] >= start and tok["end"] <= end
        ]

        if matched_indices:
            labels = ["B-NOV"] + ["I-NOV"] * (len(matched_indices) - 1)
            token_span_indices.append(matched_indices)
            token_span_labels.append(labels)

    # Assign each span to a column
    assignments = assign_columns(token_span_indices, num_cols)

    # Building the label grid
    grille = [["O"] * len(tokens) for _ in range(num_cols)]
    for span_idx, col_idx in assignments:
        for tok_idx, label in zip(token_span_indices[span_idx],
                                  token_span_labels[span_idx]):
            grille[col_idx][tok_idx] = label

    with open(bio_path, "w", encoding="utf-8") as f:
        for i, tok in enumerate(tokens):
            cols = "\t".join(grille[col][i] for col in range(num_cols))
            f.write(f"{tok['text']}\t{tok['lemma']}\t{tok['pos']}\t{cols}\n")
            if tok.get("is_sent_end"):
                f.write("\n")
        f.write("\n")

nlp = spacy.load("fr_core_news_sm")

CORPUS_DIR = "data/SFcorpus"
ANNOTATION_DIR = "data/NovSFcorpus"
TITLE2ESI_ANNOTATED_FILE = "src/title2novum.json"

with open(TITLE2ESI_ANNOTATED_FILE, "r", encoding="utf-8") as f:
    title2esi_annotated = json.load(f)

for subdir in os.listdir(CORPUS_DIR):
    subdir_path = os.path.join(CORPUS_DIR, subdir)
    if not os.path.isdir(subdir_path):
        continue

    for filename in os.listdir(subdir_path):
        if not filename.lower().endswith("_sent.txt"):
            continue

        title = filename[:-9]  # name without an extension or _sent

        txt_file = os.path.join(subdir_path, filename)
        
        if title not in title2esi_annotated:
            print(f"⚠ Titre absent du JSON : {title}")
            continue

        out_dir = os.path.join(ANNOTATION_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)

        ANN_FILE = os.path.join(out_dir, title + ".ann")
        BIO_FILE = os.path.join(out_dir, title + ".tsv")

        novums = [x[0] for x in title2esi_annotated[title]]
                
        NOVUMS_LEMMA = []
        NOVUMS_RAW = []
        for novum in novums:
            doc = nlp(novum)
            NOVUMS_LEMMA.append(" ".join(token.lemma_.lower() for token in doc))
            NOVUMS_RAW.append(novum.lower())
        
        raw_text = Path(txt_file).read_text(encoding="utf-8")
        tokens = tokenize(raw_text)
        spans = find_novum_spans(tokens, NOVUMS_LEMMA, NOVUMS_RAW)
        write_ann(spans, ANN_FILE, raw_text)
        write_bio(tokens, spans, BIO_FILE)

        print(f"Terminé : {title}")
        
