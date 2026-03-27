import stanza
import json
from collections import Counter
import matplotlib.pyplot as plt

stanza.download('fr')
nlp_stanza = stanza.Pipeline('fr', processors='tokenize,pos,lemma')

with open("src\\title2novum.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect unique novum
novums = set()
for doc_id, entries in data.items():
    for entry in entries:
        novums.add(entry[0])

# Analyze POS patterns
pattern_counter = Counter()
verb_novums = []

for novum in novums:
    doc = nlp_stanza(novum)
    pattern = " ".join(
        token.upos
        for sent in doc.sentences
        for token in sent.words
    )
    pattern_counter[pattern] += 1
    if pattern == "NOUN ADP NOUN":
        verb_novums.append(novum) # TODO

# Print novum with a specific pattern
print("=== Mots-fictions avec pattern VERB ===") # TODO
for n in sorted(verb_novums): # TODO
    print(f"  {n}")

# Graphic
top_patterns = pattern_counter.most_common(10)
patterns, counts = zip(*top_patterns)

plt.figure(figsize=(8, 6))
bars = plt.bar(patterns, counts)
plt.ylabel("Fréquence")
plt.xlabel("Distribution morpho-syntaxique")
plt.xticks(rotation=45, ha="right")

for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             str(count), ha="center", va="bottom")

plt.tight_layout()
plt.show()