from seqeval.metrics import classification_report

def read_bio_tsv(filepath: str) -> tuple[list[list[str]], list[list[str]]]:
    sentences_tokens = []
    sentences_tags = []
    
    current_tokens = []
    current_tags = []
    
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            
            if line.strip() == "":
                if current_tokens:
                    sentences_tokens.append(current_tokens)
                    sentences_tags.append(current_tags)
                    current_tokens = []
                    current_tags = []
            else:
                parts = line.split("\t")
                if len(parts) < 2:
                    # Ligne mal formée, on ignore
                    continue
                token = parts[0]
                tag = parts[-1]  # dernière colonne = tag
                current_tokens.append(token)
                current_tags.append(tag)
    
    # Dernière phrase si pas de ligne vide finale
    if current_tokens:
        sentences_tokens.append(current_tokens)
        sentences_tags.append(current_tags)
    
    return sentences_tokens, sentences_tags

def align_predictions(
    gold_tokens: list[list[str]],
    gold_tags: list[list[str]],
    pred_tokens: list[list[str]],
    pred_tags: list[list[str]],
) -> tuple[list[list[str]], list[list[str]]]:

    if len(gold_tokens) != len(pred_tokens):
        raise ValueError(
            f"Nombre de phrases différent : gold={len(gold_tokens)}, pred={len(pred_tokens)}"
        )
    
    aligned_gold = []
    aligned_pred = []
    
    for i, (g_toks, p_toks) in enumerate(zip(gold_tokens, pred_tokens)):
        
        # On prend la longueur minimale pour éviter les IndexError
        min_len = min(len(g_toks), len(p_toks))
        aligned_gold.append(gold_tags[i][:min_len])
        aligned_pred.append(pred_tags[i][:min_len])
    
    return aligned_gold, aligned_pred

if __name__ == "__main__":
    
    gold_file = "src/NER_training_files/test.tsv"
    pred_file = "src/NER_training_files/pred_by_mistral.tsv"
    
    gold_tokens, gold_tags = read_bio_tsv(gold_file)
    pred_tokens, pred_tags = read_bio_tsv(pred_file)
    
    # Alignement
    aligned_gold, aligned_pred = align_predictions(
        gold_tokens, gold_tags, pred_tokens, pred_tags
    )

    print(classification_report(aligned_gold, aligned_pred, digits=4))
