# 5. Dependency Inversion Principle - DIP

The Dependency Inversion Principle tells us how to achieve loose coupling accross several modules. We know we want to achieve loose coupling between some modules because it will improve our ability to extend the system quickly, by giving us high confidence that a change in some module will not break the entire application unexpectedly. 

As an example, let's imagine that we need to build a keras-like framework for training deep learning models; for such an application, we usually need to log training progress (e.g. how loss evolves as we do more training steps) and save model checkpoints following some rules (e.g. save the best model so far). The following code shows one possible way to do it[^1]:

```python
from .log_loss import log_loss
from .save_model import save_model

class Model:
    def fit(self, dataset: Iterable[Tuple[np.narray, np.ndarray]]):
        for epoch in range(self.num_epochs):
            for X, y in dataset:
                
```

[^1]: Be aware that there are a ton of simplifications and code that had to be omited in order to focus on the important aspects, the same is true for all other examples.
