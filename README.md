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

To run the scripts, create a virtual environment. Once inside, run the following line of code:

```
pip install -r requirements.txt
```

To build the dataset, run the followwing line. The train, dev and test files obtained are identical as those saved in `src`.

```
py .\script\build_dataset.py
```

To train the model using the best hyperparameters, run the following line. This new model will be saved in `src\SF_NER\`.

```
py .\script\novum_detection\train.py
```

To reproduce the results on the test set, make sure `test_sentences = read_tsv_file(Path("src/test.tsv"))` is uncommented and `test_sentences = read_tsv_file(Path("data/NerSFcorpus/BOOK_PATH.tsv"))` is commented in `script\novum_detection\predict_tsv.py`, then run the following line.

To evaluate the model on a complete book, make sure `test_sentences = read_tsv_file(Path("src/test.tsv"))` is commented and `test_sentences = read_tsv_file(Path("data/NerSFcorpus/BOOK_PATH.tsv"))` is uncommented in `script\novum_detection\predict_tsv.py`, then run the following line.

```
py .\script\novum_detection\predict_tsv.py
```

The results on the test set are:

| | Precision | Recall | F1-score |
|:----------|:----------:|
| PER | 92.00 | 94.03 | 93.00 |
| LOC | 80.73 | 85.55 | 83.07 |
| ORG | 68.66 | 71.91 | 70.25 |
| NOV | 67.02 | 60.29 | 63.48 |
| MISC | 83.00 | 87.23 | 85.06 |
| micro F1-score | 87.90 | 90.77 | 89.31 |
| macro F1-score | 78.28 | 79.80 | 78.97 |

## Source folder

The `src/` folder contains:

* `train.tsv`: train set of the model

* `dev.tsv`: dev set of the model

* `test.tsv`: test set of the model

* `SF_NER_final\`: trained model with the best hyperparameters

* `title2novum.json`: dictionary linking story titles to the novum they contain

* `metadata.csv`: textual corpus metadata 

## Licence

This repository is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/deed.fr).