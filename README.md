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
│   ├── novum_detection/
│   │   ├── build_dataset.py
│   │   ├── train.py
│   │   ├── predict_tsv.py
│   ├── txt2ann.py
│   └── txt2tsv.py
│
├── src/
│   ├── model_ner_final/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   ├── training_args.json
│   ├── title2novum.json
│   ├── metadata.csv
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv

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

The scripts are located in the `script/` folder.

* `build_dataset.py`: builds the train, dev and test sets

* `novum_detection\train.py`: trains the NER+NOV model and save the model in the `src` folder

* `novum_detection\predict_tsv.py`: uses the trained model and predicts (then evaluates) the annotations on the test set or on a book (tsv format) 

To run the scripts, create a virtual environment. Once inside, run the following lines of code:

```
pip install -r requirements.txt
```

## Source folder

The `src/` folder contains:

* `train.tsv`: train set of the model

* `dev.tsv`: dev set of the model

* `test.tsv`: test set of the model

* `title2novum.json`: dictionary linking story titles to the novum they contain

* `metadata.csv`: textual corpus metadata 

## Licence

This repository is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/deed.fr).