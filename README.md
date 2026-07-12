# FrenchSFcorpus

This repository contains a corpus of French science fiction stories (short stories, novellas, and novels) between 1860 and 1950 for linguistic analysis and annotation, as well as a set of scripts for producing different representations of the text from a source file.

## Repository structure

```
FrenchSFcorpus/
│
├── data/
│   ├── NerSFcorpus/
│   │   ├── About_LeCasDeMGuerin_1862/
│   │   │   ├── About_LeCasDeMGuerin_1862.ann
│   │   │   └── About_LeCasDeMGuerin_1862.tsv
│   │   │
│   │   ├── About_LeNezDUnNotaire_1862/
│   │   │   └── ...
│   │
│   ├── NovSFcorpus/
│   │   ├── About_LeCasDeMGuerin_1862/
│   │   │   ├── About_LeCasDeMGuerin_1862.ann
│   │   │   └── About_LeCasDeMGuerin_1862.tsv
│   │   │
│   │   ├── About_LeNezDUnNotaire_1862/
│   │   │   └── ...
│   │
│   ├── SFcorpus/
│   │   ├── About_LeCasDeMGuerin_1862/
│   │   │   ├── About_LeCasDeMGuerin_1862_sent.txt
│   │   │   └── About_LeCasDeMGuerin_1862.txt
│   │   │
│   │   ├── About_LeNezDUnNotaire_1862/
│   │   │   └── ...
│
├── script/
│   ├── NER/
│   │   ├── CamemBERT_NER_model/
│   │   │   ├── train.py
│   │   │   └── predict_tsv.py
│   │   ├── LLM/
│   │   │   ├── predict_mistral.py
│   │   │   ├── evaluate_mistral.py
│   │   │   ├── predict_universalNER.py
│   │   │   └── evaluate_universalNER.py
│   ├── annotate.py
│   └── build_dataset.py
│
├── src/
│   ├── SF_NER_final_model/
│   ├── NER_training_files/
│   ├── title2novum.json
│   └── metadata.csv

```

## Data folder 

Each subfolder in `data` corresponds to a story and is named according to the format:

```
author_title_date
```

Example:

```
Verne_VoyageAuCentreDeLaTerre_1864
```

In `SFcorpus`,

* `author_title_date.txt`: facsimile of the cleaned full text

* `author_title_date_sent.txt`: text segmented into sentences (one sentence per line)

In `NovSFcorpus`, 

* `author_title_date.tsv`: tabular version of the text with annotations of novums 

* `author_title_date.ann`: annotation file (BRAT format) of novums

In `NerSFcorpus`, 

* `author_title_date.tsv`: tabular version of the text with NER (and novum) annotations 

* `author_title_date.ann`: annotation file (BRAT format) of entities and novum

All metadata is grouped together in the file `src/metadata.csv` in the format `author,title,date_publication,nb_tokens`.

## Script folder

The `script/` folder contains:

* `NER/`: folder for the named entity and novum recognition

* `NOV_annotations/`: folder to automatically novum annotation

## Source folder

The `src/` folder contains:

* `NER_training_files`: set of files to train the NER model (train, dev, set, results from Mistral and UniversalNER zero- and few-shot)

* `SF_NER_final_model/`: trained model with the best hyperparameters

* `title2novum.json`: dictionary linking story titles to the novum they contain

* `metadata.csv`: textual corpus metadata 

* `metadata_complete.json`: more complete textual corpus metadata (narratives, registers and disciplines were added)

## Licence

This repository is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/deed.fr).