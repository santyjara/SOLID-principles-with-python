from typing import Dict, List, Optional, Union

from joblib import load
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from base import ModelStrategy


class TransformerEmbedding(TransformerMixin, BaseEstimator):
    """Compute sentence embeddings using a transformer based model

    Attributes
    ----------
    model_name : str
        Model embedding generator
    model : sentence_transformers.SentenceTransformer
        Transformer model to generate sentence embeddings
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        print("Downloading sentence transformer model ...", end=" ", flush=True)
        self.model = SentenceTransformer(model_name)
        print("[OK]")

    def fit(self, X, y=None):
        return self

    def transform(self, x: Union[str, List[str]]):
        embeddings = self.model.encode(x)
        return embeddings


class SentenceEncoderModel(ModelStrategy):
    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__(model_path)
        if self.model_path:
            self.model = self._load_model(self.model_path)
        else:
            self.model = None

    def train(
        self,
        x_train: List[str],
        y_train: List[str],
        x_test: List[str],
        y_test: List[str],
        params: Dict[str, Union[str, int]],
    ) -> None:
        if not self.model:
            self.model = Pipeline(
                [
                    (
                        "encoder",
                        TransformerEmbedding(params["sentence_encoder_name"]),
                    ),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=params["n_estimators"],
                            max_depth=params["max_depth"],
                        ),
                    ),
                ]
            )

        print("Fitting the Random Forest Classifier ...", end=" ", flush=True)
        self.model.fit(x_train, y_train)
        print("[OK]")
        print("Calculating metrics ...")
        print("train accuracy: {}".format(self.model.score(x_train, y_train)))
        print("test accuracy: {}".format(self.model.score(x_test, y_test)))

    def predict(self, samples: Union[str, List[str]]) -> List[int]:
        return self.model.predict(samples)

    def eval(self, x_test: List[str], y_test: List[str]) -> Dict[str, float]:
        return {"accuracy": self.model.score(x_test, y_test)}

    @staticmethod
    def _load_model(model_path: str) -> Pipeline:
        return load(model_path)
