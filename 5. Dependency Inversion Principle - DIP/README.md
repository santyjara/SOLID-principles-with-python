# 5. Dependency Inversion Principle - DIP

The Dependency Inversion Principle [states](https://en.wikipedia.org/wiki/Dependency_inversion_principle#cite_note-Martin2003-1):

1. High-level modules should not import anything from low-level modules. Both should depend on abstractions (e.g., interfaces).
2. Abstractions should not depend on details. Details (concrete implementations) should depend on abstractions.

It is actually impossible to follow those two statements everywhere when you are writing an application, at some point the main application needs to depend on some concrete implementations or nothing useful would execute. Another way to interpret the principle is to say it is telling us _how to achieve loose coupling accross several modules_. We know we want to achieve loose coupling between some modules because it will improve our ability to extend the system quickly, by giving us high confidence that a change in some module will not break the entire application unexpectedly. 

As an example, let's imagine that we need to build a library for training deep learning models; for such an application, we usually need to log training progress (e.g. how loss evolves as we do more training steps) and save model checkpoints following some rules (e.g. save the best model so far). The following `training.py` shows one possible way to do it[^1]:

```python
from .log_loss import log_loss
from .save_model import save_model_locally

def train_model(model: ModelInterface, dataset: Iterable[Tuple[np.narray, np.ndarray]]):
    best_loss = ... # pick the worst possible loss value
    for epoch in range(self.num_epochs):
        for X, y in dataset:
            loss, metrics = ... # do forward pass, computing loss and metrics
            ... # do the backwward pass for each batch
        log_loss(loss)
        if loss < best_loss:
            best_loss = loss
            save_model_locally(self)
```
Using this approach, there are many reasons for which we would need to work on this module:
* If we need to modify the rules for saving the model.
* If we need to log other things besides the loss (e.g. metrics).
* If we need to save the model to a remote storage.
* If we need to log training progress to a different system (e.g. a remote server instead of the standard output).

This means that the way in which we log trainnig progress and save a model checkpoint is **tightly coupled** with the implementation of the learning algorithm. This has the following consequences for the maintainability of the project:
* When any dependency of the `log_loss.py` (or the `save_model.py`) module fails, then the `trainining.py` fails as well.
* When we change the logic for loging (or for saving) we need to run the tests cases for training.
* When we change the training algorithm, we need to make sure the loging and saving logic is still in place.

Alternatively, we can make an implementation that leverages **dependency inversion** so that we can remove coupling as follows:

```python
from .callback_interface import CallbackInterface

def train_model(
    model: ModelInterface,
    dataset: Iterable[Tuple[np.narray, np.ndarray]],
    callbacks: List[CallbackInterface]
):
    best_loss = ... # pick the worst possible loss value
    for epoch in range(self.num_epochs):
        for X, y in dataset:
            loss, metrics = ... # do forward pass, computing loss and metrics
            ... # do the backwward pass for each batch
        for callback in callbacks:
            callback(loss, metrics)
```
Now, logging training progress and saving the model is not mentioned in this module, all we know is that the training algorithm executes some callbacks at the end of every epoch, but the _details_ of those callbacks are not important: we can change where and how the training progress is beind saved (e.g. we can log to a file, to the standard output or to a remote server) or we can modify the rules for saving the model (e.g. a model is saved only every few epochs to reduce latency or using a runing average of the loss to reduce noise).

Given that the DIP tells us _how_ to achieve loose coupling instead of _when_ to achieve it, the tricky part becomes when to use it. Here are a couple of cases when I find it useful:
* When it is likely that implementations will change in the future (e.g. save model every few epochs instead of every epoch). The designer of the system being built can forsee a few things based on their knowledge of the domain of the application being built, but it is impossible to know for sure; therefore, my suggestion is to apply it whenever you actually realize there was a change in the requirements. Forseeing too many things in advance can increase the cost of development and prevent you from delivering features in time. 
* When there are some IO operations involved (e.g. when connecting to a DB or calling an external server). This will allow you test your entire application without the need to connecting to a particular server (unittesting), which improves the speed and stability of the tests.


[^1]: Be aware that there are a ton of simplifications and code that had to be omited in order to focus on the important aspects, the same is true for all other examples.
