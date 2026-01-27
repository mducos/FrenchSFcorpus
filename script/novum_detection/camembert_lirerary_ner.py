from renard.pipeline import Pipeline
from renard.pipeline.tokenization import NLTKTokenizer
from renard.pipeline.ner import BertNamedEntityRecognizer

pipeline = Pipeline(
    [
        NLTKTokenizer(),
        BertNamedEntityRecognizer(
            model="compnet-renard/camembert-base-literary-NER-v2"
        )
    ]
)

out = pipeline("""Quasimodo affronte un archidiacre nommé Claude Frollo. Il fuit ensuite.""")
print(out.entities)
