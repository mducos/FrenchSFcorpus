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

3. To fine-tune CamemBERT using the best hyperparameters, run the following line. This new model will be saved in `src/SF_NER_final_model/`.

```
py script/NER/train.py
```

4.  a) To reproduce the results on the test set with NerSFmodel, make sure `test_sentences = read_tsv_file(Path("src/test.tsv"))` is uncommented and `test_sentences = read_tsv_file(Path("data/NerSFcorpus/BOOK_PATH.tsv"))` is commented in `script/NerSFmodel/predict_tsv.py`, then run the following line.

    b) To evaluate fine-tuned NerSFmodel on a complete book, make sure `test_sentences = read_tsv_file(Path("src/test.tsv"))` is commented and `test_sentences = read_tsv_file(Path("data/NerSFcorpus/BOOK_PATH.tsv"))` is uncommented in `script/NerSFmodel/predict_tsv.py`, then run the following line.

```
py script/NER/predict_tsv.py
```

The results on the test set are:

|           | Precision | Recall | F1-score | support |
|:----------|:----------:|:----------:|:----------:|:----------:|
| COL | 85.44 | 86.85 | 86.14 | 1460 |
| DUR | 99.27 | 99.44 | 99.36 | 2324 |
| LOC | 86.14 | 84.43 | 85.28 | 7804 |
| ORG | 66.82 | 77.47 | 71.76 | 728 |
| PER | 93.72 | 93.40 | 93.56 | 18992 |
| PONC | 98.38 | 99.01 | 98.70 | 3746 |
| micro F1-score | 91.93 | 91.80 | 91.87 | 35054 |
| macro F1-score | 88.30 | 90.10 | 89.13 | 35054 |

8. To annotate a new story using fine-tuned CamemBERT, make sure to put your input_file and your output_file in `script/NER/annotate.py`, then run the following line.

```
py script/NER/annotate.py
```