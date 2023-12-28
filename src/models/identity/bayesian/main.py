import numpy as np

from src.experiment_pipeline import BaseExperiment
from src.utils.prediction import BayesianPrediction


class BayesianLinearRegression(BayesianPrediction, BaseExperiment):
    """Dummy model that uses Bayesian linear regression after the identity transformation.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        coverage (float): The coverage of the confidence interval on the train set.
    """

    def __init__(
        self,
        dataset: str,
        b_0: float,
        alpha_0: float,
        delta_0: float,
        coverage: float,
    ):
        BayesianPrediction.__init__(self, b_0, alpha_0, delta_0, coverage)
        BaseExperiment.__init__(self, dataset)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.fit_bayesian_head(X_train, y_train)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X

    def get_params(self) -> dict:
        prediction_params = BayesianPrediction.get_params(self)
        return {**prediction_params}


if __name__ == "__main__":
    dataset = "simple"

    prediction_kwargs = {
        "b_0": 1.00,
        "alpha_0": 1.0,
        "delta_0": 1.0,
        "coverage": 0.90,
    }

    model = BayesianLinearRegression(dataset, **prediction_kwargs)
    model.run_experiment()
