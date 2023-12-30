import numpy as np

from src.experiment_pipeline import BaseExperiment


class DummyModel(BaseExperiment):
    """Dummy model that predicts a constant confidence interval based on the target distribution.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        coverage (float): The coverage of the confidence interval on the train set.
    """

    def __init__(
        self,
        dataset: str,
        coverage: float,
    ):
        super().__init__(dataset)
        if not (0 < coverage < 1):
            raise ValueError("Coverage must be between 0 and 1")
        self.coverage = coverage
        self.confidence_interval = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        lower_percentile = (1 - self.coverage) / 2
        upper_percentile = 1 - lower_percentile
        self.confidence_interval = np.quantile(
            y_train, [lower_percentile, upper_percentile]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.confidence_interval is None:
            raise RuntimeError("Model not fitted")
        return np.tile(self.confidence_interval, (X.shape[0], 1))

    def get_params(self) -> dict:
        return {"prediction_type": "direct", "coverage": self.coverage}


if __name__ == "__main__":
    dataset = "simple"
    model_kwargs = {
        "coverage": 0.90,
    }

    model = DummyModel(dataset, **model_kwargs)
    model.run_experiment()
