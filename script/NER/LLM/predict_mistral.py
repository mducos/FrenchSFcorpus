import langextract as lx
import textwrap
import json
import os
from pathlib import Path
from collections import defaultdict



def read_conll_tsv(filepath):
    """
    Lit un fichier TSV de type CoNLL :
    - 1 token par ligne
    - phrases séparées par une ligne vide
    Retourne une liste de phrases (str).
    """
    sentences = []
    current_tokens = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_tokens:
                    sentences.append(" ".join(current_tokens))
                    current_tokens = []
            else:
                # on prend le premier champ = token
                token = line.split("\t")[0]
                current_tokens.append(token)

        # dernière phrase si pas de ligne vide finale
        if current_tokens:
            sentences.append(" ".join(current_tokens))

    return sentences

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""
    A named entity is a specific and unique designation, typically:
    - a proper noun,
    - or a noun phrase referring to a single identifiable entity in 
    the story world.              

    Extract named entities (NER) according to the following categories: 
    PER for characters
    LOC for places
    ORG for organizations
    MISC for peoples
    NOV for novums
    ALL TERMS BEGINNING WITH A CAPITAL LETTER (other than the first words of
    the sentence) MUST BE ANNOTATED IN A CLASS
                            
    PER = a person identified by a proper name, title, or pseudonym ("Mr. 
    Héricourt", "Dr. Flax", "Georges").
    The following are not PERs:
    - descriptions ("a widow", "a son"),
    - personal pronouns ("I", "you", "he", "she", "we", "you", "they"),
    - professions, degrees, titles ("the doctor", "the mailman"),
    - noun phrases without proper nouns ("the neighbor", "the widow").
                            
    LOC = a named geographical location, place or toponym, including parks, 
    harbours, valleys and cities ("Paris", "England", "Cape Hatteras").
    The following are not LOCs:
    - partial addresses ("5th Avenue"),
    - expressions referring to a place ("his home"),
    - the spatiotemporal context in adverbial phrases ("over there", "here"),
    - materials or substances ("ice", "water", "sand"),
    - parts of an object or vehicle ("the edge", "the cabin").
                            
    ORG = a named institution, government, administration, military force 
    or organization, including fictional ones ("Ecole d'Arts et Métiers", 
    "the State", "Electric-Standard").
    The following are not ORGs:
    - described functions, ranks, or units ("the commander"),
    - organizational groups without an official name ("the group", "the 
    association").
            
    NOV = a new and speculative idea, concept, object, technology, or substance 
    that does not exist in the real world or in known encyclopedic culture 
    ("hydrostat", "napusify", "artificial blood"). 
    The following are not NOVs:
    - invented or non-invented events (funerals, wars, revolutions),
    - known concepts (sirens, automata)
    - invented or non-invented social practices or rituals,
    - non-invented abstract concepts (ideology, belief, morality),
    - metaphors or figures of speech,
    - real objects described in an unusual way (metaphor).

    MISC = a people, nationality, ethnic group or inhabitants of a place 
    referred to collectively or individually by their origin ("a Frenchman",
    "a Martian")
                            
    Extract only the exact text segments present in the sentence.
    Do not rephrase, do not complete, do not infer.
                            
    Entities must not overlap; they belong to 
    a single class. They consist of a maximum of 6 tokens.
    Entities cannot consist of pronouns. It must represent a
    concrete element of the world.
    If in doubt, do not annotate.
                            
    Examples of sentences without entities to annotate:
    - C ' était l ' automate qui faisait maintenant ce qu ' elle voulait !
    - Sa mère était veuve et sans fortune .
    - On éteignit les incendies avec une composition chimique .
    - La portion du globe terrestre occupée par les eaux est évaluée à trois millions huit cent trente - deux mille cinq cent cinquante - huit myriamètres carrés , soit plus de trente - huit millions d ' hectares .
    """)

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="C ' est alors que Gasguin désespéré avait eu la sinistre impression , effroyablement exacte , que sa fille s ' était un jour réveillée comme avec un autre cerveau .",
        extractions=[
            lx.data.Extraction(
                extraction_class="PER",
                extraction_text="Gasguin",
            ),
        ]
    ),
    lx.data.ExampleData(
        text="Mais le capitaine n ' eût pas le temps de se livrer à ses réflexions .",
        extractions=[
            lx.data.Extraction(
                extraction_class="PER",
                extraction_text="le capitaine",
            ),
        ]
    ),
    lx.data.ExampleData(
        text="J ' en rends grâces à Dieu , répondit le vieillard des Conils .",
        extractions=[
            lx.data.Extraction(
                extraction_class="PER",
                extraction_text="Dieu"
            ),
            lx.data.Extraction(
                extraction_class="PER",
                extraction_text="le vieillard des Conils"
            ),
        ]
    ),
    lx.data.ExampleData(
        text="D ' ailleurs , Boucaud compte au 3e tirailleurs algériens , et moi je suis brigadier réserviste au 6e chasseurs à cheval et je dois me rendre aussitôt que possible au dépôt de mon régiment , à Castelsarrazin .",
        extractions=[ 
            lx.data.Extraction(
                extraction_class="PER",
                extraction_text="Boucaud"
            ),
            lx.data.Extraction(
                extraction_class="ORG",
                extraction_text="3e tirailleurs"
            ),
            lx.data.Extraction(
                extraction_class="ORG",
                extraction_text="6e chasseurs"
            ),
            lx.data.Extraction(
                extraction_class="LOC",
                extraction_text="Castelsarrazin"
            ),
        ]
    ), 
    lx.data.ExampleData(
        text="Le véhicule passa sous la voûte réservée aux voitures , traversa la place de Paris , où se trouve l ' ambassade française et pénétra dans l ' allée des Tilleuls .",
        extractions=[
            lx.data.Extraction(
                extraction_class="LOC",
                extraction_text="la place de Paris",
            ),
            lx.data.Extraction(
                extraction_class="ORG",
                extraction_text="l ' ambassade française",
            ),
            lx.data.Extraction(
                extraction_class="LOC",
                extraction_text="l ' allée des Tilleuls",
            ),
        ]
    ),
    lx.data.ExampleData(
        text="Plus tard , Jem West verrait à remplacer les mâts de hune et de flèche , et , dans tous les cas , ils n ' étaient point indispensables pour regagner soit les Falklands , soit quelque autre lieu d ' hivernage .",
        extractions=[
            lx.data.Extraction(
                extraction_class="PER",
                extraction_text="Jem West"
            ),
            lx.data.Extraction(
                extraction_class="LOC",
                extraction_text="les Falklands"
            ),
        ]
    ),
    lx.data.ExampleData(
        text="Ses roues avaient marqué le sol d ' une profonde empreinte , et deux lignes blanchâtres tracées à la surface du champ de glace trahissaient sa fuite vers le Sud .",
        extractions=[
            lx.data.Extraction(
                extraction_class="LOC",
                extraction_text="le Sud"
            ),
        ]
    ),
    lx.data.ExampleData(
        text="Près de lui , l ' harmonica chimique avait commencé sa musique mystérieuse .",
        extractions=[
            lx.data.Extraction(
                extraction_class="NOV",
                extraction_text="harmonica chimique"
            ),
        ]
    ),
    lx.data.ExampleData(
        text="Le cent - cinquième , impatienté , inventa une machine à compter les poils et la passa au cent - sixième .",
        extractions=[
            lx.data.Extraction(
                extraction_class="NOV",
                extraction_text="machine à compter les poils"
            ),
        ]
    ),
    lx.data.ExampleData(
        text="Pour produire le génie dans le cerveau d ' un de mes petits bonshommes , j ' introduis une parcelle , un grain de ce radium - flaxium à l ' endroit même où gît la faculté , la fonction intellectuelle que je veux centupler .",
        extractions=[
            lx.data.Extraction(
                extraction_class="NOV",
                extraction_text="radium - flaxium"
            ),
        ]
    ),
    lx.data.ExampleData(
        text="Escortés par les Amazouns , les voyageurs quittèrent Beharsand juchés sur quatre yaks vigoureux .",
        extractions=[
            lx.data.Extraction(
                extraction_class="MISC",
                extraction_text="les Amazouns"
            ),
            lx.data.Extraction(
                extraction_class="LOC",
                extraction_text="Beharsand"
            ),
        ]
    ),
    lx.data.ExampleData(
        text="— Sur la pêche , répondit le Canadien .",
        extractions=[
            lx.data.Extraction(
                extraction_class="MISC",
                extraction_text="le Canadien"
            ),
        ]
    ),
    lx.data.ExampleData(
        text="La ferveur des Hindous était motivée par la présence , au centre du cercle , d ' un brahme étendu sur le sol et faisant de vains efforts pour se relever .",
        extractions=[
            lx.data.Extraction(
                extraction_class="MISC",
                extraction_text="des Hindous"
            ),
        ]
    ),
]

def annotated_document_to_dict(ad):
    return {
        "extractions": [
            {
                "extraction_class": ex.extraction_class,
                "extraction_text": ex.extraction_text,
                "char_interval": {
                    "start_pos": ex.char_interval.start_pos if ex.char_interval else None,
                    "end_pos": ex.char_interval.end_pos if ex.char_interval else None
                }
            } for ex in ad.extractions
        ],
        "text": ad.text,
        "document_id": ad.document_id
    }

def fix_char_intervals(record: dict) -> dict:
    text = record["text"]
    valid_extractions = []
    for extraction in record["extractions"]:
        span = extraction["extraction_text"]
        start = text.find(span)
        if start != -1:
            extraction["char_interval"]["start_pos"] = start
            extraction["char_interval"]["end_pos"] = start + len(span)
            valid_extractions.append(extraction)
        # sinon on ne l'ajoute pas → supprimé
    record["extractions"] = valid_extractions
    return record

def jsonl_to_bio(jsonl_path: str, bio_path: str):
    """Convertit le fichier JSONL en fichier TSV BIO."""
    with open(jsonl_path, "r", encoding="utf-8") as f_in, \
         open(bio_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record["text"]
            tokens = text.split()
            extractions = record["extractions"]

            # Calcul des offsets de chaque token dans le texte
            token_spans = []
            cursor = 0
            for tok in tokens:
                start = text.index(tok, cursor)
                token_spans.append((start, start + len(tok)))
                cursor = start + len(tok)

            # Initialisation des tags à O
            tags = ["O"] * len(tokens)

            # Pour chaque entité, on cherche les tokens couverts
            for ext in extractions:
                ent_start = ext["char_interval"]["start_pos"]
                ent_end   = ext["char_interval"]["end_pos"]
                label     = ext["extraction_class"]

                covered = []
                for i, (ts, te) in enumerate(token_spans):
                    if ts is not None and te is not None and ent_start is not None and ent_end is not None:
                        if ts >= ent_start and te <= ent_end:
                            covered.append(i)

                if not covered:
                    continue
                tags[covered[0]] = f"B-{label}"
                for i in covered[1:]:
                    tags[i] = f"I-{label}"

            for tok, tag in zip(tokens, tags):
                f_out.write(f"{tok}\t{tag}\n")
            f_out.write("\n")

sentences = read_conll_tsv("src/NER_training_files/test.tsv")

batch_size = 5
with open("src/NER_training_files/pred_by_mistral.jsonl", "a+", encoding="utf-8") as f:
    for i in range(0, len(sentences), batch_size): #len(sentences)
        batch = sentences[i:i+batch_size]
        batch_text = "\n\n".join(batch)
        try:
            # Run the extraction
            result = lx.extract(
                text_or_documents=batch_text,
                prompt_description=prompt,
                examples=examples,
                model_id="mistral-nemo:latest",
                temperature=0.0,
                language_model_params={"timeout": 300},
            )

            # result_batch peut être un seul AnnotatedDocument ou liste selon LangExtract
            if not isinstance(result, list):
                result = [result]

            # Si le résultat contient tout le batch concaténé, on le découpe
            for res in result:
                d = annotated_document_to_dict(res)
                
                # Si le texte contient plusieurs phrases (batch entier), on découpe
                if "\n\n" in d["text"]:
                    # Découpage phrase par phrase
                    for sent_text in batch:
                        sent_extractions = [
                            ext for ext in d["extractions"]
                            if sent_text.find(ext["extraction_text"]) != -1
                        ]
                        sent_doc = {
                            "extractions": sent_extractions,
                            "text": sent_text,
                            "document_id": None
                        }
                        sent_doc = fix_char_intervals(sent_doc)
                        f.write(json.dumps(sent_doc, ensure_ascii=False) + "\n")
                        f.flush()
                else:
                    # Un seul document, écriture normale
                    d = fix_char_intervals(d)
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                    f.flush()

        except Exception as e:
            # annotation vide pour CHAQUE phrase du batch en cas d'erreur
            for sent_text in batch:
                empty = {"extractions": [], "text": sent_text, "document_id": None}
                f.write(json.dumps(empty, ensure_ascii=False) + "\n")
                f.flush()
            print(f"Erreur -> {e}")
            
jsonl_to_bio("src/NER_training_files/pred_by_mistral.jsonl", "src/NER_training_files/pred_by_mistral.tsv")
