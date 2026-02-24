from seqeval.metrics import classification_report


def read_predictions(filepath: str) -> tuple[list[list[str]], list[list[str]]]:

    all_gold, all_pred = [], []
    current_gold, current_pred = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if current_gold:
                    all_gold.append(current_gold)
                    all_pred.append(current_pred)
                    current_gold, current_pred = [], []
            else:
                parts = line.split("\t")
                if len(parts) >= 3:
                    _, gold, pred = parts[0], parts[1], parts[2]
                    current_gold.append(gold)
                    current_pred.append(pred)

    if current_gold:
        all_gold.append(current_gold)
        all_pred.append(current_pred)

    return all_gold, all_pred


def evaluate(filepath: str):
    gold, pred = read_predictions(filepath)

    print(classification_report(gold, pred, digits=4))


if __name__ == "__main__":
    evaluate("src/pred_by_universalNER_few_shot.tsv")