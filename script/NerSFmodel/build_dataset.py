
import random
from collections import Counter
from pathlib import Path

import numpy as np

CORPUS_DIR = Path("data/NerSFcorpus")
OUT_DIR = Path("src/NER_training_files")
RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
SEED = 42
REMOVED_LABELS = {"B-OBJ": "O", "I-OBJ": "O"}
HAS_HEADER = False
N_RESTARTS = 30
N_STEPS = 5000


def load_book(path: Path):
    """Returns the list of sentences.
    sentence = a list of (token, label)"""

    sentences, current = [], []

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


def assign_books(books: dict):
    """Distribute the books to achieve an 80/10/10 sentence ratio"""

    titles = sorted(books)
    split_names = list(RATIOS)
    ratios = np.array([RATIOS[k] for k in split_names])

    # Nombre de phrases par livre
    sentence_counts = np.array([len(books[t]) for t in titles], dtype=float)
    total_sentences = sentence_counts.sum()
    targets = ratios * total_sentences

    def score(assignment):
        sums = np.zeros(len(split_names))
        np.add.at(sums, assignment, sentence_counts)
        return np.sum(((sums - targets) / targets) ** 2)

    rng = np.random.default_rng(SEED)
    best_assignment, best_score = None, float("inf")

    for _ in range(N_RESTARTS):
        assignment = rng.choice(
            len(split_names), size=len(titles), p=ratios
        )
        current_score = score(assignment)

        for _ in range(N_STEPS):
            candidate = assignment.copy()

            if rng.random() < 0.5:
                # Déplacer un livre
                i = rng.integers(len(titles))
                candidate[i] = rng.integers(len(split_names))
            else:
                # Échanger deux livres
                i, j = rng.integers(len(titles), size=2)
                candidate[i], candidate[j] = assignment[j], assignment[i]

            candidate_score = score(candidate)

            if candidate_score <= current_score:
                assignment = candidate
                current_score = candidate_score

        if current_score < best_score:
            best_assignment = assignment.copy()
            best_score = current_score

    result = {k: [] for k in split_names}

    for title, index in zip(titles, best_assignment):
        result[split_names[index]].append(title)

    print(f"Meilleur score : {best_score:.4f}\n")
    return result


def write_split(path: Path, sentences):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
            f.write("\n")


def main():
    tsv_files = sorted(CORPUS_DIR.glob("*/*.tsv"))

    if not tsv_files:
        raise SystemExit(
            f"No .tsv found in {CORPUS_DIR}/[title]/[title].tsv"
        )

    books = {p.parent.name: load_book(p) for p in tsv_files}

    print(
        f"{len(books)} books loaded, "
        f"{sum(len(s) for s in books.values())} total sentences\n"
    )

    assignment = assign_books(books)

    splits = {
        k: [sentence for title in assignment[k] for sentence in books[title]]
        for k in RATIOS
    }

    label_counts = {
        k: Counter(label for sentence in splits[k] for _, label in sentence)
        for k in RATIOS
    }

    for name, sentences in splits.items():
        write_split(OUT_DIR / f"{name}.tsv", sentences)


    # STATISTICS

    total_sentences = sum(len(s) for s in splits.values())
    total_tokens = sum(sum(label_counts[k].values()) for k in RATIOS)

    print("=== sentences / tokens / books per set ===")
    print(
        f"{'set':<8}{'books':>8}{'sentences':>12}"
        f"{'% \sentences':>12}{'tokens':>12}{'% tokens':>11}"
    )

    for k in RATIOS:
        n_tokens = sum(label_counts[k].values())
        print(
            f"{k:<8}{len(assignment[k]):>8}{len(splits[k]):>12}"
            f"{100 * len(splits[k]) / total_sentences:>11.2f}%"
            f"{n_tokens:>12}{100 * n_tokens / total_tokens:>10.2f}%"
        )

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

    print(f"\nFiles written in {OUT_DIR}/ (train.tsv, dev.tsv, test.tsv)")


if __name__ == "__main__":
    main()
