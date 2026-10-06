import random
from pathlib import Path
from collections import defaultdict, Counter

def write_tsv(sentences, path):
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for line in sent:
                f.write(line + "\n")
            f.write("\n")

def get_nov_types(sent):
    """
    Récupère l'ensemble des formes de surface NOV présentes dans une phrase
    (reconstruit les entités B-NOV/I-NOV en une seule chaîne, pas juste le premier token).
    """
    nov_spans = set()
    current = []
    for line in sent:
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        token, tag = parts[0], parts[1].strip()
        if tag == 'B-NOV':
            if current:
                nov_spans.add(" ".join(current).lower())
            current = [token]
        elif tag == 'I-NOV' and current:
            current.append(token)
        else:
            if current:
                nov_spans.add(" ".join(current).lower())
                current = []
    if current:
        nov_spans.add(" ".join(current).lower())
    return nov_spans

def count_entities_per_class(sentences):
    """
    Compte le nombre d'entités (spans complets, pas de tokens) par classe
    dans une liste de phrases, en se basant sur les tags B-<CLASS>.
    """
    counts = Counter()
    for sent in sentences:
        for line in sent:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            tag = parts[1].strip()
            if tag.startswith('B-'):
                counts[tag[2:]] += 1
    return counts

def print_class_distribution(name, sentences):
    counts = count_entities_per_class(sentences)
    total = sum(counts.values())
    print(f"\nDistribution des entités - {name} ({total} entités au total)")
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / total if total else 0
        print(f"  {cls:6s} : {n:5d} ({pct:.1f}%)")

def oversample_nov(sentences, factor=5):
    result = []
    for sent in sentences:
        has_nov = any(
            len(line.split('\t')) >= 2 and line.split('\t')[1].strip() == 'B-NOV'
            for line in sent
        )
        if has_nov:
            result.extend([sent] * factor)
        else:
            result.append(sent)
    return result

NER_DIR = Path("data/NerSFcorpus")

all_sentences = []
sentence_to_book = []  # même index que all_sentences : nom du livre (sous-dossier) d'origine

for subdir in NER_DIR.iterdir():
    if not subdir.is_dir():
        continue
    for tsv_file in subdir.glob("*.tsv"):
        with open(tsv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        phrase = []
        for line in lines:
            if line.strip() == "":
                if phrase:
                    all_sentences.append(phrase)
                    sentence_to_book.append(subdir.name)
                    phrase = []
            else:
                phrase.append(line.rstrip("\n"))
        if phrase:
            all_sentences.append(phrase)
            sentence_to_book.append(subdir.name)

print(f"Phrases après filtrage  : {len(all_sentences)}")

random.seed(281)
# Shuffle conjoint pour garder all_sentences et sentence_to_book alignés
combined = list(zip(all_sentences, sentence_to_book))
random.shuffle(combined)
all_sentences, sentence_to_book = [list(t) for t in zip(*combined)]

# --- Étape 1 : regrouper les phrases par livre ET par novum (union-find) ---
# Deux contraintes de regroupement fusionnées dans le même union-find :
#   (a) toutes les phrases d'un même livre restent ensemble (contrainte principale demandée)
#   (b) toutes les phrases partageant un même novum restent ensemble
#       (utile si un même novum apparaît dans plusieurs livres différents ;
#        sans cette contrainte, un novum partagé entre deux livres pourrait
#        quand même se retrouver disjoint... mais comme deux livres ne
#        peuvent de toute façon plus être séparés par la contrainte (a) si
#        eux-mêmes partagent une phrase groupée, (b) sert surtout de garde-fou
#        explicite et de vérification a posteriori)

parent = list(range(len(all_sentences)))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry:
        parent[rx] = ry

# (a) Union par livre : toutes les phrases d'un même livre dans un seul groupe
book_to_first_idx = {}
for i, book in enumerate(sentence_to_book):
    if book not in book_to_first_idx:
        book_to_first_idx[book] = i
    else:
        union(book_to_first_idx[book], i)

# (b) Union par novum partagé (inter ou intra-livre)
nov_to_sentence_idx = defaultdict(list)
for i, sent in enumerate(all_sentences):
    for nov in get_nov_types(sent):
        nov_to_sentence_idx[nov].append(i)

for nov, idxs in nov_to_sentence_idx.items():
    for other in idxs[1:]:
        union(idxs[0], other)

groups = defaultdict(list)
for i in range(len(all_sentences)):
    groups[find(i)].append(i)

group_list = list(groups.values())
random.shuffle(group_list)

# --- Étape 2 : répartir les GROUPES (livres + novums fusionnés) en 80/10/10 ---
n_total = len(all_sentences)
target_train, target_dev = 0.8 * n_total, 0.1 * n_total

train_idx, dev_idx, test_idx = [], [], []
train_count = dev_count = 0

for group in group_list:
    if train_count < target_train:
        train_idx.extend(group)
        train_count += len(group)
    elif dev_count < target_dev:
        dev_idx.extend(group)
        dev_count += len(group)
    else:
        test_idx.extend(group)

train_sents = [all_sentences[i] for i in train_idx]
dev_sents = [all_sentences[i] for i in dev_idx]
test_sents = [all_sentences[i] for i in test_idx]

train_books = {sentence_to_book[i] for i in train_idx}
dev_books = {sentence_to_book[i] for i in dev_idx}
test_books = {sentence_to_book[i] for i in test_idx}

print(f"Total number of sentences: {n_total}")
print(f"Train : {len(train_sents)} sentences, {len(train_books)} livres")
print(f"Dev   : {len(dev_sents)} sentences, {len(dev_books)} livres")
print(f"Test  : {len(test_sents)} sentences, {len(test_books)} livres")

# --- Vérification de la disjonction des livres (attendu : ensembles vides) ---
print(f"Livres en commun train/dev   : {train_books & dev_books}")
print(f"Livres en commun train/test  : {train_books & test_books}")
print(f"Livres en commun dev/test    : {dev_books & test_books}")

# --- Vérification de la disjonction NOV ---
train_nov = set().union(*[get_nov_types(s) for s in train_sents]) if train_sents else set()
dev_nov = set().union(*[get_nov_types(s) for s in dev_sents]) if dev_sents else set()
test_nov = set().union(*[get_nov_types(s) for s in test_sents]) if test_sents else set()

print(f"NOV en commun train/dev   : {train_nov & dev_nov}")
print(f"NOV en commun train/test  : {train_nov & test_nov}")
print(f"NOV en commun dev/test    : {dev_nov & test_nov}")

# --- Distribution des classes par set (avant oversampling) ---
print_class_distribution("Train (avant oversampling)", train_sents)
print_class_distribution("Dev", dev_sents)
print_class_distribution("Test", test_sents)

# --- Oversampling (uniquement sur train, après split) ---
train_oversampled = oversample_nov(train_sents, factor=10)
print(f"\nTrain before oversampling : {len(train_sents)} sentences")
print(f"Train after oversampling  : {len(train_oversampled)} sentences")

# --- Distribution des classes dans train après oversampling ---
print_class_distribution("Train (après oversampling)", train_oversampled)

output_dir = Path("src/NER_training_files")
output_dir.mkdir(parents=True, exist_ok=True)

write_tsv(train_oversampled, output_dir / "train.tsv")
write_tsv(dev_sents, output_dir / "dev.tsv")
write_tsv(test_sents, output_dir / "test.tsv")