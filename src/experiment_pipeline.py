from abc import ABC, abstractmethod
import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import joblib

from src.utils.metrics import Metrics
from src.utils.logging import log_results
from src.utils.analysis import Analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


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

    def __init__(
        self, dataset: str, make_analysis: bool = False, make_prediction: bool = False
    ):
        self.dataset = dataset
        self._validate_dataset()

        self.make_analysis = make_analysis
        self.make_prediction = make_prediction

        self.model_name = self.__class__.__name__

    def _validate_dataset(self):
        valid_datasets = [f.name for f in Path("datasets").iterdir() if f.is_dir()]
        if self.dataset not in valid_datasets:
            raise ValueError(
                f"Invalid dataset {self.dataset}. Valid datasets are {valid_datasets}"
            )

    def _load_dataset(self):
        dataset_path = Path("datasets", self.dataset)

        X_train = np.load(Path(dataset_path, "X_train.npy"))
        y_train = np.load(Path(dataset_path, "y_train.npy"))
        X_val = np.load(Path(dataset_path, "X_val.npy"))
        y_val = np.load(Path(dataset_path, "y_val.npy"))
        X_test = np.load(Path(dataset_path, "X_test.npy"))
        try:
            target_transform = joblib.load(Path(dataset_path, "target_transform.save"))
        except FileNotFoundError:
            target_transform = None

        return X_train, y_train, X_val, y_val, X_test, target_transform

    def _predict_test(self, X_test: np.ndarray, target_transform) -> None:
        """Save predictions on test set."""
        y_pred = self.predict(X_test)
        # Return to original scale
        if target_transform is not None:
            y_pred = target_transform.inverse_transform(y_pred)

        df = pd.read_csv(Path("predictions", "sample_submission.csv"))
        df[["pi_lower", "pi_upper"]] = y_pred
        df.to_csv(Path("predictions", f"{self.model_name}.csv"), index=False)

    def run_experiment(self):
        X_train, y_train, X_val, y_val, X_test, target_transform = self._load_dataset()

        logging.info(f"Fitting model '{self.model_name}' on dataset '{self.dataset}'.")
        self.fit(X_train, y_train, X_val, y_val)

        for X, y, stage in [(X_train, y_train, "train"), (X_val, y_val, "val")]:
            y_pred = self.predict(X)

            logging.info(f"({stage}): computing and logging metrics.")
            metrics = Metrics(y, y_pred).compute_metrics()
            log_results(metrics, stage)

            if self.make_analysis:
                logging.info(f"({stage}): running analysis.")
                analysis = Analysis(
                    y,
                    y_pred,
                    stage,
                    model_name=self.model_name,
                    save_path=Path("results"),
                )
                analysis.full_analysis()

        if self.make_prediction:
            logging.info("({stage}): making predictions.")
            self._predict_test(X_test, target_transform)

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
