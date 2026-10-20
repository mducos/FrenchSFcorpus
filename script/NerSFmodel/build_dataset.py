import json
import random
from collections import Counter
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------- config ---
CORPUS_DIR = Path("data/NerSFcorpus")
OUT_DIR = Path("src/NER_training_files")
RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
SEED = 42
REMOVED_LABELS = {"B-OBJ": "O", "I-OBJ": "O"}
HAS_HEADER = False  # passer à True si les .tsv ont une ligne d'en-tête
BALANCED_LABELS = ["COMM"]   # classes à répartir 80/10/10 (en nb d'entités B-)
MIN_ENTITIES = {"COMM": 100}               # minimum dans dev ET test
N_RESTARTS = 30
N_STEPS = 5000
# ------------------------------------------------------------------------- #


def load_book(path: Path):
    """Retourne la liste des phrases ; une phrase = liste de (token, label)."""
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
    """
    Recherche locale avec redémarrages aléatoires.
    Objectif : phrases + entités des BALANCED_LABELS au plus près de 80/10/10,
    avec un minimum d'entités (MIN_ENTITIES) dans dev et test.
    """
    titles = sorted(books)
    split_names = list(RATIOS)                       # train, dev, test
    ratios = np.array([RATIOS[k] for k in split_names])

    # Matrice livres x features : [phrases, B-<label> pour chaque classe]
    features = ["sents"] + BALANCED_LABELS
    M = np.zeros((len(titles), len(features)))
    for i, t in enumerate(titles):
        M[i, 0] = len(books[t])
        c = Counter(l for s in books[t] for _, l in s)
        for j, lab in enumerate(BALANCED_LABELS, start=1):
            M[i, j] = c[f"B-{lab}"]
    totals = M.sum(axis=0)

    def score(a):
        sums = np.zeros((len(split_names), len(features)))
        np.add.at(sums, a, M)
        # écart relatif à la cible pour chaque set et chaque feature
        dev = (sums / (ratios[:, None] * totals) - 1.0) ** 2
        s = dev.sum()
        # pénalité forte si le minimum n'est pas atteint dans dev/test
        for lab, minimum in MIN_ENTITIES.items():
            j = features.index(lab)
            for k in (1, 2):                          # dev, test
                if sums[k, j] < minimum:
                    s += 100 + (minimum - sums[k, j])
        return s

    rng = np.random.default_rng(SEED)
    best_a, best_s = None, float("inf")

    for _ in range(N_RESTARTS):
        a = rng.choice(len(split_names), size=len(titles), p=ratios)
        s = score(a)
        for _ in range(N_STEPS):
            b = a.copy()
            if rng.random() < 0.5:                    # déplacer un livre
                b[rng.integers(len(titles))] = rng.integers(len(split_names))
            else:                                     # échanger deux livres
                i, j = rng.integers(len(titles), size=2)
                b[i], b[j] = a[j], a[i]
            sb = score(b)
            if sb <= s:
                a, s = b, sb
        if s < best_s:
            best_a, best_s = a.copy(), s

    assignment = {k: [] for k in split_names}
    for t, k in zip(titles, best_a):
        assignment[split_names[k]].append(t)

    ok = all(
        sum(M[titles.index(t), features.index(lab)] for t in assignment[k]) >= m
        for lab, m in MIN_ENTITIES.items() for k in ("dev", "test")
    )
    print(f"Meilleur score : {best_s:.4f} | contraintes MIN_ENTITIES respectées : {ok}\n")
    return assignment    


def write_split(path: Path, sentences):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for token, label in sent:
                f.write(f"{token}\t{label}\n")
            f.write("\n")


def main():
    tsv_files = sorted(CORPUS_DIR.glob("*/*.tsv"))
    if not tsv_files:
        raise SystemExit(f"Aucun .tsv trouvé dans {CORPUS_DIR}/[titre]/[titre].tsv")

    books = {}
    for p in tsv_files:
        books[p.parent.name] = load_book(p)
    print(f"{len(books)} livres chargés, "
          f"{sum(len(s) for s in books.values())} phrases au total\n")

    assignment = assign_books(books)

    splits = {k: [s for t in assignment[k] for s in books[t]] for k in RATIOS}
    label_counts = {k: Counter(l for s in splits[k] for _, l in s) for k in RATIOS}

    # --- écriture
    for name, sents in splits.items():
        write_split(OUT_DIR / f"{name}.tsv", sents)
    #with open(OUT_DIR / "split_books.json", "w", encoding="utf-8") as f:
    #    json.dump(assignment, f, ensure_ascii=False, indent=2)

    # --- stats : phrases
    total_sents = sum(len(s) for s in splits.values())
    total_tokens = sum(sum(label_counts[k].values()) for k in RATIOS)
    print("=== Phrases / tokens / livres par set ===")
    print(f"{'set':<8}{'livres':>8}{'phrases':>12}{'% phrases':>12}{'tokens':>12}{'% tokens':>11}")
    for k in RATIOS:
        n_tok = sum(label_counts[k].values())
        print(f"{k:<8}{len(assignment[k]):>8}{len(splits[k]):>12}"
              f"{100 * len(splits[k]) / total_sents:>11.2f}%"
              f"{n_tok:>12}{100 * n_tok / total_tokens:>10.2f}%")

    # --- stats : labels
    all_labels = sorted(set().union(*label_counts.values()), key=lambda l: (l == "O", l))
    print("\n=== Occurrences par label (et part de chaque set pour ce label) ===")
    header = f"{'label':<10}" + "".join(f"{k:>12}{'%':>8}" for k in RATIOS) + f"{'total':>12}"
    print(header)
    for lab in all_labels:
        tot = sum(label_counts[k][lab] for k in RATIOS)
        row = f"{lab:<10}"
        for k in RATIOS:
            c = label_counts[k][lab]
            row += f"{c:>12}{100 * c / tot if tot else 0:>7.1f}%"
        row += f"{tot:>12}"
        print(row)

    print(f"\nFichiers écrits dans {OUT_DIR}/ (train.tsv, dev.tsv, test.tsv)")


if __name__ == "__main__":
    main()