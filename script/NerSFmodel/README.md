# Named entity and novum recognition

Python 3.11 is used.

1. To run the scripts, create a virtual environment. Once inside, run the following line of code:

```
pip install -r requirements.txt
```

2. To build the dataset, run the followwing line. The train, dev and test files obtained are identical as those saved in `src`.

```
py script/build_dataset.py
```

3. To fine-tune CamemBERT using the best hyperparameters, run the following line. This new model will be saved in `src/SF_NER/`.

```
py script/NER/train.py
```

4.  a) To reproduce the results on the test set with fine-tuned CamemBERT, make sure `test_sentences = read_tsv_file(Path("src/test.tsv"))` is uncommented and `test_sentences = read_tsv_file(Path("data/NerSFcorpus/BOOK_PATH.tsv"))` is commented in `script/NER/CamemBERT_NER_model/predict_tsv.py`, then run the following line.

    b) To evaluate fine-tuned CamemBERT on a complete book, make sure `test_sentences = read_tsv_file(Path("src/test.tsv"))` is commented and `test_sentences = read_tsv_file(Path("data/NerSFcorpus/BOOK_PATH.tsv"))` is uncommented in `script/NER/predict_tsv.py`, then run the following line.

```
py script/NER/predict_tsv.py
```

The results on the test set are:

|           | Precision | Recall | F1-score |
|:----------|:----------:|:----------:|:----------:|
| PER | 92.00 | 94.03 | 93.00 |
| LOC | 80.73 | 85.55 | 83.07 |
| ORG | 68.66 | 71.91 | 70.25 |
| NOV | 67.02 | 60.29 | 63.48 |
| MISC | 83.00 | 87.23 | 85.06 |
| micro F1-score | 87.90 | 90.77 | 89.31 |
| macro F1-score | 78.28 | 79.80 | 78.97 |

5. To reproduce the results on the test set with Mistral, run the following line.

```
py script/NER/LLM/evaluate_Mistral.py
```

The results on the test set are:

|           | Precision | Recall | F1-score |
|:----------|:----------:|:----------:|:----------:|
| micro F1-score | 57.83 | 38.81 | 46.45 |
| macro F1-score | 32.72 | 22.78 | 23.45 |

6. To reproduce the results on the test set with UniversalNER zero-shot, make sure to put `evaluate("src/pred_by_universalNER_zero_shot.tsv")` in `script/NER/LLM/evaluate_universalNER.py` then run the following line.

```
py script/NER/LLM/evaluate_universalNER.py
```

The results on the test set are:

|           | Precision | Recall | F1-score |
|:----------|:----------:|:----------:|:----------:|
| micro F1-score | 72.11 | 42.48 | 53.46 |
| macro F1-score | 34.41 | 16.40 | 20.35 |

7. To reproduce the results on the test set with UniversalNER few-shot, make sure to put `evaluate("src/pred_by_universalNER_few_shot.tsv")` in `script/NER/LLM/evaluate_universalNER.py` then run the following line.

```
py script/NER/LLM/evaluate_universalNER.py
```

The results on the test set are:

|           | Precision | Recall | F1-score |
|:----------|:----------:|:----------:|:----------:|
| micro F1-score | 72.60 | 33.57 | 45.91 |
| macro F1-score | 37.56 | 13.04 | 16.72 |

8. To annotate a new story using fine-tuned CamemBERT, make sure to put your input_file and your output_file in `script/NER/annotate.py`, then run the following line.

```
py script/NER/annotate.py
```