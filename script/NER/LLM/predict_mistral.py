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
    Une entité nommée est une désignation spécifique et unique, généralement :
    - un nom propre,
    - ou un groupe nominal référant à une entité identifiable unique dans 
    le monde du récit.              

    Extraire les entités nommées (NER) selon les entités suivantes : 
    PER pour les personnages
    LOC pour les lieux
    ORG pour les organisations
    MISC pour les peuples
    NOV pour les novum
    TOUS LES TERMES COMMENCANT AVEC UNE MAJUSCULE (en dehors des permiers mots de
    la phrase) DOIVENT ÊTRE ANNOTES DANS UNE CLASSE
                         
    PER = personnes identifiées par un nom propre ou pseudonyme ("M.
    Héricourt", "le docteur Flax", "Georges").
    Ne sont pas des PER :
    - descriptions ("une veuve", "un fils"),
    - les pronoms personnels ("je", "tu", "il", "elle", "nous", "vous", "ils"),
    - professions, diplômes, statuts ("le docteur", "le facteur"),
    - groupes nominaux sans nom propre ("le voisin", "la veuve").
                         
    LOC = toponymes avec nom propre ("Paris", "l'Angleterre", "le cap 
    Hatteras").
    Ne sont pas des LOC :
    - dates, âges, durées ("huit ans", "deux heures"),
    - adresses partielles ("5 rue"),
    - expressions qui font référence à un lieu ("son domicile"),
    - le cadre spacio-temporel dans les compléments circonstanciels ("là-bas", "ici"),
    - matières ou substances ("la glace", "l'eau", "le sable"),
    - parties d'un objet ou d'un véhicule ("le bord", "la cabine").
                         
    ORG = institutions nommées avec nom propre (administrations, écoles, armées) 
    ("Ecole d'Arts et Métiers", "l'Etat", "l'Electric-Standard").
    Ne sont pas des ORG :
    - fonctions, grades, unités décrites ("le commandant"),
    - groupes organisationnels sans nom officiel ("le groupe", "l'association").
         
    NOV = idée, concept, objet, technologie ou substance nouvelle et 
    spéculative qui n'existe pas dans le monde réel ou dans la culture 
    encyclopédique connue ("hydrostat", "napusifier", "sang artificiel"). 
    Les concepts connus ou historiques 
    (automates, sirènes, maladies réelles, pratiques médicales 
    existantes) ne sont pas des novums.                     
    Ne sont pas des NOV :
    - événements inventés ou non (funérailles, guerres, révolutions),
    - pratiques sociales ou rituels inventés ou non,
    - concepts abstraits (idéologie, croyance, morale) non-inventés,
    - métaphores ou figures de style,
    - objets réels décrits de manière inhabituelle.
                         
    Extraire uniquement les segments textuels exacts présents dans la phrase.
    Ne pas reformuler, ne pas compléter, ne pas inférer.
                         
    Les entités ne doivent pas se chevaucher, elles appartiennent à 
    une seule classe. Elles sont composées d'un maximum de 6 tokens.
    Les entités ne peuvent pas être composées de pronoms. Il faut au moins
    un nom commun (ou nom propre) dans l'entité et qu'elle représente un élément
    concret du monde.
    En cas de doute, ne pas annoter.
                         
    Exemples de phrases sans entité à annoter :
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

sentences = read_conll_tsv("src/test.tsv")

batch_size = 5
with open("src/pred_by_mistral.jsonl", "a+", encoding="utf-8") as f:
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
            
jsonl_to_bio("src/pred_by_mistral.jsonl", "src/pred_by_mistral.tsv")
