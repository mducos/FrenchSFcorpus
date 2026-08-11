import random
from pathlib import Path

SEED = 42


def write_tsv(sentences, path):
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for line in sent:
                f.write(line + "\n")
            f.write("\n")  # empty line = end of sentence


def oversample_nov(sentences, factor=5):
    """
    Duplicate sentences containing B-NOV a certain number of times.

    Args:
        sentences: list of sentences
        factor: total number of copies (5 = 1 original + 4 duplicates)

    Returns:
        List of sentences with duplication of sentences containing NOV
    """
    result = []
    nov_count = 0

    for sent in sentences:
        has_nov = False
        for line in sent:
            parts = line.split('\t')
            if len(parts) >= 2 and parts[1].strip() == 'B-NOV':
                has_nov = True
                break

        if has_nov:
            for _ in range(factor):
                result.append(sent)
            nov_count += 1
        else:
            result.append(sent)

    return result


def load_book_sentences(tsv_path):
    """Lit un fichier .tsv et retourne la liste de ses phrases (chaque phrase = liste de lignes)."""
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sentences = []
    phrase = []
    for line in lines:
        if line.strip() == "":
            if phrase:  # end of sentence
                sentences.append(phrase)
                phrase = []
        else:
            phrase.append(line.rstrip("\n"))
    if phrase:
        sentences.append(phrase)

    return sentences


def count_tokens(sentences):
    return sum(len(sent) for sent in sentences)


# =========================
# CHARGEMENT PAR LIVRE
# =========================

NER_DIR = Path("data/NerSFcorpus")

books = []  # liste de (nom_livre, [phrases])
for subdir in NER_DIR.iterdir():
    if not subdir.is_dir():
        continue
    for tsv_file in subdir.glob("*.tsv"):
        book_sentences = load_book_sentences(tsv_file)
        if book_sentences:
            books.append((tsv_file.stem, book_sentences))

print(f"Nombre de livres : {len(books)}")
print(f"Nombre total de phrases : {sum(len(s) for _, s in books)}")
print(f"Nombre total de tokens : {sum(count_tokens(s) for _, s in books)}")

# =========================
# SPLIT 80/10/10 AU NIVEAU DES LIVRES (reproductible)
# =========================

rng = random.Random(SEED)
books_shuffled = books.copy()
rng.shuffle(books_shuffled)

n_books = len(books_shuffled)
n_train_books = int(0.8 * n_books)
n_dev_books = int(0.1 * n_books)

train_books = books_shuffled[:n_train_books]
dev_books = books_shuffled[n_train_books:n_train_books + n_dev_books]
test_books = books_shuffled[n_train_books + n_dev_books:]

print(f"\nLivres -> Train: {len(train_books)} | Dev: {len(dev_books)} | Test: {len(test_books)}")

# Vérification de l'équilibre en TOKENS (pas juste en nombre de livres)
train_tokens = sum(count_tokens(s) for _, s in train_books)
dev_tokens = sum(count_tokens(s) for _, s in dev_books)
test_tokens = sum(count_tokens(s) for _, s in test_books)
total_tokens = train_tokens + dev_tokens + test_tokens

print("\nÉquilibre en tokens :")
print(f"  Train: {train_tokens:>8} ({train_tokens/total_tokens:.1%})")
print(f"  Dev  : {dev_tokens:>8} ({dev_tokens/total_tokens:.1%})")
print(f"  Test : {test_tokens:>8} ({test_tokens/total_tokens:.1%})")
print("  -> si un split s'écarte trop de sa cible (80/10/10), vérifiez qu'aucun")
print("     livre géant n'a été assigné seul à un petit split.")

# =========================
# APLATISSEMENT + SHUFFLE DES PHRASES DANS CHAQUE SET
# =========================

def flatten_and_shuffle(book_list, seed):
    sentences = [sent for _, book_sents in book_list for sent in book_sents]
    random.Random(seed).shuffle(sentences)
    return sentences

# seeds dérivées de SEED pour rester reproductibles mais différentes par split
train_sents = flatten_and_shuffle(train_books, seed=SEED + 1)
dev_sents = flatten_and_shuffle(dev_books, seed=SEED + 2)
test_sents = flatten_and_shuffle(test_books, seed=SEED + 3)

print(f"\nPhrases -> Train: {len(train_sents)} | Dev: {len(dev_sents)} | Test: {len(test_sents)}")

# =========================
# OVERSAMPLING NOV (train uniquement)
# =========================

train_oversampled = oversample_nov(train_sents, factor=10)
print(f"\nTrain avant oversampling : {len(train_sents)} phrases")
print(f"Train après oversampling : {len(train_oversampled)} phrases")

# =========================
# ÉCRITURE
# =========================

output_dir = Path("src/NER_training_files")
output_dir.mkdir(parents=True, exist_ok=True)

write_tsv(train_oversampled, output_dir / "train.tsv")
write_tsv(dev_sents, output_dir / "dev.tsv")
write_tsv(test_sents, output_dir / "test.tsv")

print(f"\n✔ Fichiers écrits dans {output_dir}")