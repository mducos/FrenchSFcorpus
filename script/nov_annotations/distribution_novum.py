import json
import matplotlib.pyplot as plt
from collections import Counter

file_path = r"src\\title2novum.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

esi_by_type = {1: [], 2: [], 3: [], 4: []}

for esi_list in data.values():
    counts = {1:0, 2:0, 3:0, 4:0}
    for esi, annotation in esi_list:
        if annotation in counts:
            counts[annotation] += 1
    for k in counts:
        if counts[k] > 0:
            esi_by_type[k].append(counts[k])

# Histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Find global limits
max_x = max(max(esi_by_type[k]) for k in esi_by_type if esi_by_type[k])
max_y = max(max(Counter(esi_by_type[k]).values()) for k in esi_by_type if esi_by_type[k])

x_range = range(1, max_x + 2)

for i, k in enumerate([1, 2, 3, 4]):
    counts = esi_by_type[k]
    counter = Counter(counts)
    y = [counter.get(v, 0) for v in x_range]

    axes[i].bar(x_range, y, width=0.8, edgecolor="black")
    axes[i].set_xlabel(f"Nombre de mots-fictions de classe {k}")
    axes[i].set_ylabel("Nombre de récits")
    axes[i].set_xticks(x_range)
    axes[i].set_xticklabels([v if j % 2 != 0 else "" for j, v in enumerate(x_range)])
    axes[i].set_xlim(0, max_x + 2) 
    axes[i].set_ylim(0, max_y + 1) 
    axes[i].set_axisbelow(True)
    axes[i].grid()

plt.tight_layout()
plt.show()
