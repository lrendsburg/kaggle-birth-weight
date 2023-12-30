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


ANALYZE_AND_PREDICT_THRESHOLD = 3


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

        self.analyze_and_predict = False

    def _validate_dataset(self):
        valid_datasets = [f.name for f in Path("datasets").iterdir() if f.is_dir()]
        if self.dataset not in valid_datasets:
            raise ValueError(
                f"Invalid dataset {self.dataset}. Valid datasets are {valid_datasets}"
            )

    def _validate_params(self, params):
        if "prediction_type" not in params.keys():
            raise ValueError("prediction_type must be specified in params.")

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
        self.analyze_and_predict = (score < ANALYZE_AND_PREDICT_THRESHOLD) and (
            coverage > 0.9
        )
        if self.analyze_and_predict:
            self.prediction_file_name = (
                f"{self.model_name}_s{score:.2f}_c{coverage:.2f}"
            )

    def _predict_test(self, X_test: np.ndarray, target_transform) -> None:
        """Save predictions on test set."""
        y_pred = self.predict(X_test)
        # Return to original scale
        if target_transform is not None:
            y_pred = target_transform.inverse_transform(y_pred)

        df = pd.read_csv(Path("predictions", "sample_submission.csv"))
        df[["pi_lower", "pi_upper"]] = y_pred
        df.to_csv(Path("predictions", f"{self.prediction_file_name}.csv"), index=False)

    def _log_params(self):
        params = {
            "dataset": self.dataset,
            "model": self.model_name,
            **self.get_params(),
        }
        self._validate_params(params)
        mlflow.log_params(params)

    def _compute_and_log_metrics(self, X: np.ndarray, y: np.ndarray, stage: str):
        """Compute and log metrics for a given stage. Also computes the boolean flag
        'analyze_and_predict' based on the validation metrics to decide whether the model
        is further analyzed and used to make predictions on the test set."""
        # Compute metrics
        y_pred = self.predict(X)
        metrics = Metrics(y, y_pred).compute_metrics()

        # Print and log metrics
        self._print_results(metrics, stage)
        stage_metrics = {f"{k}_{stage}": v for k, v in metrics.items()}
        mlflow.log_metrics(stage_metrics)

        # Evaluate model
        if stage == "val":
            self._evaluate_model(metrics)
            model_score = -metrics["winkler"]
            return model_score

    def _analyze_model(self, X: np.ndarray, y: np.ndarray, stage: str):
        y_pred = self.predict(X)
        analysis = Analysis(
            y,
            y_pred,
            stage,
            model_name=self.prediction_file_name,
        )
        analysis.full_analysis()

    def run_experiment(self):
        X_train, y_train, X_val, y_val, X_test, target_transform = self._load_dataset()

        logging.info(f"Fitting model '{self.model_name}' on dataset '{self.dataset}'.")
        self.fit(X_train, y_train, X_val, y_val)

        mlflow.set_experiment(self.model_name)
        with mlflow.start_run():
            self._log_params()

            self._compute_and_log_metrics(X_train, y_train, "train")
            model_score = self._compute_and_log_metrics(X_val, y_val, "val")

            if self.analyze_and_predict:
                logging.info(f"(train/val): running analysis.")
                self._analyze_model(X_train, y_train, "train")
                self._analyze_model(X_val, y_val, "val")

                logging.info("(test): making predictions.")
                self._predict_test(X_test, target_transform)

        if hasattr(self, "get_model_score"):
            model_score = self.get_model_score()
        return model_score

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
