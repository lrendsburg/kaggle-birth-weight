from abc import ABC, abstractmethod


# TODO: implement
class BaseExperiment(ABC):
    """
    Base class for model benchmarking experiments.

    This abstract class serves as a foundation for running and benchmarking various models.
    It implements the common workflow of data loading, training, predicting, and logging.

    To use this class, define a new class for each model and inherit from `BaseExperiment`.
    Implement the `fit` method to specify how the model is trained and the `predict` method
    to define how the model makes predictions.

    Example:
        class MyModel(BaseExperiment):
            def fit(self):
                # Implementation for training MyModel
                pass

            def predict(self):
                # Implementation for making predictions with MyModel
                pass

        # Create an instance of MyModel and run the experiment
        model = MyModel()
        model.run_experiment()
    """

    def __init__(self):
        pass

    def _load_data(self):
        pass

    def run_experiment(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass
