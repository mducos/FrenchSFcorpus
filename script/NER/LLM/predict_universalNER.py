from vllm.logger import init_logger
import os
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["TQDM_DISABLE"] = "1"

import json
import re
from vllm import LLM, SamplingParams

import logging
logging.getLogger("vllm").setLevel(logging.WARNING)

MODEL_NAME  = "Universal-NER/UniNER-7B-all"
INPUT_FILE  = "src/NER_training_files/test.tsv"
OUTPUT_FILE = "src/NER_training_files/pred_by_universalNER_few_shot.tsv"

ENTITY_TYPES = [
    "a person identified by a proper name, title, or pseudonym like: 'Sir Bucephalus', 'le docteur Flax', 'Samuel Sun', 'Burton', 'M. Héricourt'",
    "a named geographical location, place or toponym, including parks, harbours, valleys and cities like: 'Londres', 'le National Gallery', 'vallées de la Somme', 'au jardin des Plantes', 'Bâbord-Harbour'",
    "a named institution, government, administration, military force or organization, including fictional ones like: 'la Sûreté', 'gouvernement suisse', 'l \' Empire de l \' Espace', 'le Secours universel', 'la marine royale'",
    "a people, nationality, ethnic group or inhabitants of a place referred to collectively or individually by their origin like: 'un Français', 'un Américain', 'des Arabes', 'les Martiens', 'le Suédois'",
]

CUSTOM_ENTITY_TYPES = [
    "a speculative or fictional object, technology, substance, creature or concept invented by the author that does not exist in reality or in known encyclopedic culture, such as an invented machine, a neologism, a fabricated material or a fictional species like: 'cinébouquin', 'anthroposaure', 'homme lunaire', 'napusifié', 'pain de houille'",
]

ENTITY_TYPE_TO_BIO_TAG = {
    "a person identified by a proper name, title, or pseudonym like: 'Sir Bucephalus', 'le docteur Flax', 'Samuel Sun', 'Burton', 'M. Héricourt'": "PER",
    "a named geographical location, place or toponym, including parks, harbours, valleys and cities like: 'Londres', 'le National Gallery', 'vallées de la Somme', 'au jardin des Plantes', 'Bâbord-Harbour'": "LOC",
    "a named institution, government, administration, military force or organization, including fictional ones like: 'la Sûreté', 'gouvernement suisse', 'l \' Empire de l \' Espace', 'le Secours universel', 'la marine royale'": "ORG",
    "a people, nationality, ethnic group or inhabitants of a place referred to collectively or individually by their origin like: 'un Français', 'un Américain', 'des Arabes', 'les Martiens', 'le Suédois'": "MISC",
    "a speculative or fictional object, technology, substance, creature or concept invented by the author that does not exist in reality or in known encyclopedic culture, such as an invented machine, a neologism, a fabricated material or a fictional species like: 'cinébouquin', 'anthroposaure', 'homme lunaire', 'napusifié', 'pain de houille'": "NOV",
}

def read_bio_file(filepath: str) -> list[dict]:
    sentences = []
    current_tokens, current_labels = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if current_tokens:
                    sentences.append({
                        "tokens": current_tokens,
                        "gold_labels": current_labels
                    })
                    current_tokens, current_labels = [], []
            else:
                parts = line.split("\t")
                current_tokens.append(parts[0])
                current_labels.append(parts[1] if len(parts) >= 2 else "O")

    if current_tokens:
        sentences.append({"tokens": current_tokens, "gold_labels": current_labels})

    return sentences

def load_model(model_name: str):
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=1024,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop=["</s>", "USER:"],  # empêche la boucle sur le prompt
    )
    return llm, sampling_params

def build_prompt(text: str, entity_type: str) -> str:
    return (
        "A virtual assistant answers questions from a user based on the provided text.\n"
        f"USER: Text: {text}\n"
        "ASSISTANT: I've read this text.\n"
        f"USER: What describes {entity_type} in the text?\n"
        "ASSISTANT:"
    )

def extract_entities(llm, sampling_params, text: str, entity_type: str) -> list[str]:
    prompt = build_prompt(text, entity_type)
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    raw_output = outputs[0].outputs[0].text.strip()

    try:
        match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
        if match:
            entities = json.loads(match.group())
            return [str(e) for e in entities if e]
        return []
    except (json.JSONDecodeError, ValueError):
        return []

def entities_to_bio(tokens: list[str], entities_by_type: dict) -> list[str]:
    bio_labels = ["O"] * len(tokens)

    for entity_type, entity_list in entities_by_type.items():
        bio_type = ENTITY_TYPE_TO_BIO_TAG.get(entity_type, entity_type[:10].upper())

        for entity_span in entity_list:
            entity_tokens = entity_span.split()
            n = len(entity_tokens)

            for i in range(len(tokens) - n + 1):
                window = tokens[i:i + n]
                if [t.lower() for t in window] == [t.lower() for t in entity_tokens]:
                    if bio_labels[i] == "O":
                        bio_labels[i] = f"B-{bio_type}"
                        for j in range(1, n):
                            bio_labels[i + j] = f"I-{bio_type}"
                        break

    return bio_labels

def run_pipeline(input_file: str, output_file: str):
    sentences = read_bio_file(input_file)

    llm, sampling_params = load_model(MODEL_NAME)
    all_entity_types = ENTITY_TYPES + CUSTOM_ENTITY_TYPES

    results = []

    for idx, sentence in enumerate(sentences):
        tokens      = sentence["tokens"]
        gold_labels = sentence["gold_labels"]
        text        = " ".join(tokens)

        if idx % 1000 == 0:
            print(f"[{idx+1}/{len(sentences)}]")

        entities_by_type = {}
        for entity_type in all_entity_types:
            extracted = extract_entities(llm, sampling_params, text, entity_type)
            entities_by_type[entity_type] = extracted

        pred_labels = entities_to_bio(tokens, entities_by_type)
        results.append({
            "tokens":      tokens,
            "gold_labels": gold_labels,
            "pred_labels": pred_labels,
        })

    print(f"\nSauvegarde dans {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in results:
            for token, gold, pred in zip(
                sentence["tokens"],
                sentence["gold_labels"],
                sentence["pred_labels"]
            ):
                f.write(f"{token}\t{gold}\t{pred}\n")
            f.write("\n")


if __name__ == "__main__":
    run_pipeline(INPUT_FILE, OUTPUT_FILE)