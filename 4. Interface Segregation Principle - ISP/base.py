from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


class ModelStrategy(ABC):
    """The strategy interface declares operations common to all
    supported versions of some algorithm. The context uses this
    interface to call the algorithm defined by the concrete
    strategies.

    Attributes
    ----------
    model_path : Optional[str]
        Path where a pre-trained model is stored
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    @abstractmethod
    def train(
        self,
        x_train: List[str],
        y_train: List[str],
        x_test: List[str],
        y_test: List[str],
        params: Dict[str, Union[str, int]],
    ) -> None:
        pass

    @abstractmethod
    def predict(self, samples: Union[str, List[str]]) -> List[int]:
        pass

    @abstractmethod
    def eval(self, x_test: List[str], y_test: List[str]) -> Dict[str, float]:
        pass


class Model:
    """The context maintains a reference to one of the strategy
    objects. The context doesn't know the concrete class of a
    strategy. It should work with all strategies via the
    strategy interface.

    Attributes
    ----------
    strategy : ModelStrategy
        Model strategy to be used
    """

    def __init__(self, strategy: ModelStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> ModelStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: ModelStrategy) -> None:
        self._strategy = strategy

    def train(
        self,
        x_train: List[str],
        y_train: List[str],
        x_test: List[str],
        y_test: List[str],
        params: Dict[str, Union[str, int]],
    ) -> None:
        print("Training ....")
        self._strategy.train(x_train, y_train, x_test, y_test, params)
        print("Finished !!")

    def __call__(self, samples: Union[str, List[str]]) -> List[int]:
        results = self._strategy.predict(samples)
        print(f"Predictions: \n \t {results}")
        return results

    def eval(self, x_test: List[str], y_test: List[str]) -> Dict[str, float]:
        print("Evaluating model ....")
        metrics = self._strategy.eval(x_test, y_test)
        print("ok")
        print("\n Evaluation performance metrics:", end="\n * ")
        for metric, value in metrics.items():
            print(f"{metric}: {value}", end="\n * ")
        return metrics
