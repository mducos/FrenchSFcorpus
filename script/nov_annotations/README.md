# NovSFcorpus — Novum Annotation Pipeline

## Description

This project provides a semi-automated annotation pipeline for novum in a French-language science fiction corpus. Using text files segmented into sentences and a list of manually annotated novum, the pipeline generates annotation files in Brat format (`.ann`) and BIO format (`.tsv`).

---

## Project structure
```
FrenchSFcorpus/
│
├── data/
│   ├── SFcorpus/                  # Raw corpus
│   │   └── <title>/
│   │       └── <title>_sent.txt   # Text file divided into sentences (one sentence per line)
│   │
│   └── NovSFcorpus/               # Annotation files producted
│       └── <title>/
│           ├── <title>.ann        # Annotations in Brat format
│           └── <title>.tsv        # Annotations in BIO format
│
├── script/
│   ├── txt2ann_tsv.py             # Main annotation pipeline
│   ├── distribution_types.py      # Distribution of novum Types
│   └── distribution_pos.py        # Distribution of morphosyntactic patterns
│
└── src/
    └── title2novum.json           # List of novum by story
```

---

## Prerequisites
```bash
pip install spacy stanza matplotlib
python -m spacy download fr_core_news_ms
python -c "import stanza; stanza.download('fr')"
```

---

## Input data format

### Text files (`<title>_sent.txt`)

Text files must be divided into sentences, with one sentence per line.

### Novum type file (`title2novum.json`)

This file lists the novum manually identified for each story, in the following format:
```json
{
    "Author_Title_PublicationYear": [
        ["novum", class, "PATTERN_POS"],
        ["penséographe", 2, "NOUN"],
        ["psychologie de l ' atome", 1, "NOUN ADP DET PUNCT NOUN"]
    ]
}
```

Each entry contains:
- the word in its lemmatized form
- its type (an integer from 1 to 4; see below)
- its morphosyntactic pattern (generated automatically by `distribution_pos.py`)

#### Types of novum

| Class | Description |
|------|-------------|
| 1 | The words exist separately, but the expression formed by combining them does not exist |
| 2 | Neither the words nor the expression exist |
| 3 | Only some of the words that make up the expression exist |
| 4 | Has entered common usage, but the expression did not exist at the time of writing |

---

## Annotation pipeline

### Launch
```bash
python src/annotate.py
```

### What the script does

1. Tokenization and lemmatization of each sentence using spaCy (`fr_core_news_ms`)
2. Detection of occurrences of each novum in the text, based on a match with the lemma and/or the raw form of the token (to account for lemmatization errors in neologisms)
3. Generation of the `.ann` file in Brat format with the offsets of each occurrence
4. Generation of the `.tsv` file in BIO format with three annotation columns to handle overlaps

### Configurable parameters

| Parameter | Default value | Description |
|-----------|------------------|-------------|
| `CORPUS_DIR` | `data/SFcorpus` | Folder containing the text files |
| `ANNOTATION_DIR` | `data/NovSFcorpus` | Output directory for annotations |
| `TITLE2ESI_ANNOTATED_FILE` | `src/title2novum.json` | JSON file of novum |
| `max_gap` | `4` | Maximum number of intervening tokens allowed during detection |
| `num_cols` | `3` | Number of BIO columns to handle overlaps |

---

## Format of output files

### `.ann` file (Brat format)

Each line corresponds to an occurrence of a novum in the raw text:
```
T279	NOV 307064 307076	penséographe
T189	NOV 112502 112525	psychologie de l' atome
```

The columns are: identifier, type, and offsets (start and end in characters), surface form in the text.

### `.tsv` file (BIO format)

Each line corresponds to a token, with the following columns:
```
token   lemma   POS   col1   col2   col3
```

The three BIO annotation columns allow you to manage overlaps between novum:
```tsv
homme   homme   NOUN   B-NOV   O       O
et      et      CCONJ  O       O       O
femme   femme   NOUN   O       B-NOV   O
lunaires lunaire ADJ   I-NOV   I-NOV   O
.       .       PUNCT  O       O       O
```

The labels used are:
- `B-NOV`: first token of a novum
- `I-NOV`: subsequent token within a novum
- `O`: unannotated token

Sentences are separated by a blank line.

---

## Distribution analysis

### Distribution of novum types (`distribution_types.py`)
```bash
python src/distributions_types.py
```

This script generates four histograms (one per type) showing the distribution of the number of novum per story for each type. The four graphs use the same scale to facilitate comparison.

### Distribution of morphosyntactic patterns (`distribution_pos.py`)
```bash
python src/distribution_pos.py
```

This script analyzes the morphosyntactic patterns (POS sequences) of all novum using Stanza (`fr`) and produces:

- a graph showing the 10 most frequent patterns (e.g., `NOUN ADJ`, `NOUN ADP NOUN`)
- a list of novum whose pattern matches a target value (which can be modified in the script via the `pattern_to_print` variable)

---

## Notes

- Novum are lemmatized in `title2novum.json` to enable the detection of all their inflected forms in the texts.
- Detection relies on both the lemma and the raw form of the token to compensate for spaCy’s lemmatization errors regarding neologisms specific to science fiction.
- If there is an overlap between novum spanning more than three levels, a warning is displayed and the span in question is not annotated.
