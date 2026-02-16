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

## Organization of the corpus

Each subfolder corresponds to a story and is named according to the format:

```
author_title_date
```

Example:

```
Verne_VoyageAuCentreDeLaTerre_1864
```

Each folder contains the following files:

* `author_title_date.txt`
  Facsimile of the cleaned full text

* `author_title_date_sent.txt`
  Text segmented into sentences (one sentence per line)

* `author_title_date.tsv`
  Tabular version of the text with semi-automatic annotations of novums

* `author_title_date.ann`
  Annotation file (BRAT format) of novums

All metadata is grouped together in the file `metadata.csv` in the format `author,title,date_publication,nb_tokens`.

## Scripts

The scripts are located in the `script/` folder.

* `txt2sent_txt.py`: segments a `.txt` file into sentences and produces an associated `_sent.txt` file.

* `txt2tsv.py`: annotates novum in a `.txt` file in BIO format.

* `txt2ann.py`: annotates novum in an `.ann` file in BRAT format.

To run the scripts, create a virtual environment. Once inside, run the following lines of code:

```
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

## Resources

The `src/` folder contains:

* `title2novum.json`
  Dictionary linking story titles to the novum they contain.

## Licence

This repository is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/deed.fr).