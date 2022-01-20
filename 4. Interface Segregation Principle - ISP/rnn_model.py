from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from base import ModelStrategy


class RNNModel(ModelStrategy):
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_path)
        if self.model_path:
            self.model = load_model(self.model_path)
        else:
            self.model = None

    @staticmethod
    def get_rnn_model(x_train: List[str], params: Dict[str, int]) -> Model:
        """Get RNN model based on Tensor Flow framework

        Parameters
        ----------
        x_train : List[int]
            Sentences to train the model
        params : Dict[str: int]
            RNN model parameters

        Returns
        -------
        tf.keras.Model
            Compiled Keras model
        """
        encoder = tf.keras.layers.TextVectorization(max_tokens=params["vocab_size"])
        encoder.adapt(x_train)

        model = tf.keras.Sequential(
            [
                encoder,
                tf.keras.layers.Embedding(
                    input_dim=len(encoder.get_vocabulary()),
                    output_dim=params["embedding_output_dim"],
                    mask_zero=True,
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(params["embedding_output_dim"])
                ),
                tf.keras.layers.Dense(params["embedding_output_dim"], activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-3),
            metrics=["accuracy"],
        )
        return model

    def train(
        self,
        x_train: List[str],
        y_train: List[str],
        x_test: List[str],
        y_test: List[str],
        params: Dict[str, int],
    ) -> None:
        if not self.model_path:
            self.model = self.get_rnn_model(x_train, params)
        self.model.fit(
            np.array(x_train),
            np.array(y_train),
            validation_data=(np.array(x_test), np.array(y_test)),
            batch_size=params["batch_size"],
            epochs=params["epochs"],
        )

    def predict(self, samples: Union[str, List[str]]) -> List[int]:
        return self.model.predict(samples)

    def eval(self, x_test: List[str], y_test: List[str]) -> Dict[str, float]:
        return {"accuracy": self.model.evaluate(x_test, y_test)}
