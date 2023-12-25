from abc import ABC, abstractmethod
from pathlib import Path
import logging

import mlflow
import numpy as np
import pandas as pd
import joblib

from src.utils.metrics import Metrics
from src.utils.analysis import Analysis

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


WORTHINESS_THRESHOLD = 5


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

    def __init__(self, dataset: str):
        self.dataset = dataset
        self._validate_dataset()

        self.model_name = self.__class__.__name__
        self.prediction_file_name = None

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

    def _print_results(self, metrics: dict, stage: str) -> None:
        score, coverage = metrics["winkler"], metrics["coverage"]
        logging.info(f"({stage}): logging metrics. {score=:.4f}, {coverage=:.4f}")
        if stage == "val" and coverage < 0.9:
            logging.warning(f"({stage}): coverage is below 90%.")

    def _evaluate_model(self, metrics: dict) -> bool:
        score, coverage = metrics["winkler"], metrics["coverage"]
        model_is_worthy = (score < WORTHINESS_THRESHOLD) and (coverage > 0.9)

        if model_is_worthy:
            self.prediction_file_name = (
                f"{self.model_name}_s{score:.2f}_c{coverage:.2f}"
            )
        return model_is_worthy

    def _predict_test(self, X_test: np.ndarray, target_transform) -> None:
        """Save predictions on test set."""
        y_pred = self.predict(X_test)
        # Return to original scale
        if target_transform is not None:
            y_pred = target_transform.inverse_transform(y_pred)

        df = pd.read_csv(Path("predictions", "sample_submission.csv"))
        df[["pi_lower", "pi_upper"]] = y_pred
        df.to_csv(Path("predictions", f"{self.prediction_file_name}.csv"), index=False)

    def run_experiment(self):
        X_train, y_train, X_val, y_val, X_test, target_transform = self._load_dataset()

        logging.info(f"Fitting model '{self.model_name}' on dataset '{self.dataset}'.")
        self.fit(X_train, y_train, X_val, y_val)

        mlflow.set_experiment(self.model_name)
        with mlflow.start_run():
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_params(self.get_params())

            y_train_pred = self.predict(X_train)
            y_val_pred = self.predict(X_val)

            model_is_worthy = False
            for y, y_pred, stage in [
                (y_train, y_train_pred, "train"),
                (y_val, y_val_pred, "val"),
            ]:
                metrics = Metrics(y, y_pred).compute_metrics()
                self._print_results(metrics, stage)
                if stage == "val":
                    model_is_worthy = self._evaluate_model(metrics)

                stage_metrics = {f"{k}_{stage}": v for k, v in metrics.items()}
                mlflow.log_metrics(stage_metrics)

            if model_is_worthy:
                logging.info(f"(train/val): running analysis.")
                for X, y, stage in [(X_train, y_train, "train"), (X_val, y_val, "val")]:
                    y_pred = self.predict(X)

                    analysis = Analysis(
                        y,
                        y_pred,
                        stage,
                        model_name=self.prediction_file_name,
                    )
                    analysis.full_analysis()

                logging.info("(test): making predictions.")
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

    @abstractmethod
    def get_params(self) -> dict:
        pass
