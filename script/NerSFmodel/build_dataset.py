from collections import Counter
from pathlib import Path
import random

CORPUS_DIR = Path("data/NerSFcorpus")
OUT_DIR = Path("src/NER_training_files")

RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
SEED = 42

REMOVED_LABELS = {
    "B-OBJ": "O",
    "I-OBJ": "O",
}

HAS_HEADER = False


def load_book(path: Path):
    """Returns a list of sentences.
    Each sentence is a list of (token, label).
    """

    sentences = []
    current = []

    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    if HAS_HEADER:
        lines = lines[1:]

    for line in lines:
        if not line.strip():
            if current:
                sentences.append(current)
                current = []
            continue

        cols = line.split("\t")

        if len(cols) < 2:
            print(f"[WARN] ligne ignorée dans {path.name} : {line!r}")
            continue

        token, label = cols[0], cols[-1].strip()
        label = REMOVED_LABELS.get(label, label)

        current.append((token, label))

    if current:
        sentences.append(current)

    return sentences


def write_split(path: Path, sentences):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
            f.write("\n")


def main():

    # ========================================================
    # 1. LOAD ALL SENTENCES
    # ========================================================

    tsv_files = sorted(CORPUS_DIR.glob("*/*.tsv"))

    if not tsv_files:
        raise SystemExit(
            f"No .tsv found in {CORPUS_DIR}/[title]/[title].tsv"
        )

    all_sentences = []
    sentence_to_book = []

    for path in tsv_files:
        book_sentences = load_book(path)

        all_sentences.extend(book_sentences)
        sentence_to_book.extend([path.parent.name] * len(book_sentences))

    print(
        f"{len(tsv_files)} books loaded, "
        f"{len(all_sentences)} total sentences\n"
    )

    # ========================================================
    # 2. RANDOM SENTENCE SPLIT
    # ========================================================

    combined = list(zip(all_sentences, sentence_to_book))

    random.seed(SEED)
    random.shuffle(combined)

    all_sentences, sentence_to_book = zip(*combined)

    all_sentences = list(all_sentences)
    sentence_to_book = list(sentence_to_book)

    n_total = len(all_sentences)

    n_train = int(n_total * RATIOS["train"])
    n_dev = int(n_total * RATIOS["dev"])

    splits = {
        "train": all_sentences[:n_train],
        "dev": all_sentences[n_train:n_train + n_dev],
        "test": all_sentences[n_train + n_dev:],
    }

    book_splits = {
        "train": sentence_to_book[:n_train],
        "dev": sentence_to_book[n_train:n_train + n_dev],
        "test": sentence_to_book[n_train + n_dev:],
    }

    # ========================================================
    # 3. WRITE TSV FILES
    # ========================================================

    for name, sentences in splits.items():
        write_split(OUT_DIR / f"{name}.tsv", sentences)

    # ========================================================
    # 4. STATISTICS
    # ========================================================

    label_counts = {
        k: Counter(
            label
            for sentence in splits[k]
            for _, label in sentence
        )
        for k in RATIOS
    }

    total_sentences = sum(len(s) for s in splits.values())
    total_tokens = sum(sum(label_counts[k].values()) for k in RATIOS)

    print("=== sentences / tokens / books per set ===")
    print(
        f"{'set':<8}{'books':>8}{'sentences':>12}"
        f"{'% sentences':>13}{'tokens':>12}{'% tokens':>11}"
    )

    for k in RATIOS:
        n_tokens = sum(label_counts[k].values())
        n_books = len(set(book_splits[k]))

        print(
            f"{k:<8}{n_books:>8}{len(splits[k]):>12}"
            f"{100 * len(splits[k]) / total_sentences:>12.2f}%"
            f"{n_tokens:>12}{100 * n_tokens / total_tokens:>10.2f}%"
        )

    # ========================================================
    # 5. LABEL DISTRIBUTION
    # ========================================================

    all_labels = sorted(
        set().union(*label_counts.values()),
        key=lambda label: (label == "O", label)
    )

    print("\n=== Occurrences per label (and portion of each set) ===")

    header = f"{'label':<10}" + "".join(
        f"{k:>12}{'%':>8}" for k in RATIOS
    ) + f"{'total':>12}"

    print(header)

    for label in all_labels:
        total = sum(label_counts[k][label] for k in RATIOS)
        row = f"{label:<10}"

        for k in RATIOS:
            count = label_counts[k][label]
            percentage = 100 * count / total if total else 0
            row += f"{count:>12}{percentage:>7.1f}%"

        row += f"{total:>12}"
        print(row)

    # ========================================================
    # 6. BOOK OVERLAP
    # ========================================================

    train_books = set(book_splits["train"])
    dev_books = set(book_splits["dev"])
    test_books = set(book_splits["test"])

    print("\n=== Book overlap ===")
    print(f"Train / Dev : {len(train_books & dev_books)} books")
    print(f"Train / Test: {len(train_books & test_books)} books")
    print(f"Dev / Test  : {len(dev_books & test_books)} books")

    print(f"\nFiles written in {OUT_DIR}/ (train.tsv, dev.tsv, test.tsv)")


if __name__ == "__main__":
    main()