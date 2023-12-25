from typing import List

import numpy as np
from lightgbm import LGBMRegressor

from src.experiment_pipeline import BaseExperiment
from src.utils.prediction import QuantileRegression


class QuantileLightGBM(QuantileRegression, BaseExperiment):
    """LightGBM trained with the square loss to predict a given set of quantiles.
    Confidence intervals are created from quantiles by minimizing their width for a given coverage.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        model_kwargs (dict): Parameters of the model.
        lower_percentiles (List[float]): List of lower percentiles that are predicted, must be between 0 and 1 - coverage.
            Corresponding upper percentiles are created implicitly by adding the coverage.
        coverage (float): coverage of the confidence intervals, must be between 0 and 1.
    """

    def __init__(
        self,
        dataset: str,
        model_kwargs: dict,
        lower_percentiles: List[float],
        coverage: float,
    ) -> None:
        QuantileRegression.__init__(self, lower_percentiles, coverage)
        BaseExperiment.__init__(self, dataset)

        self.models = [
            LGBMRegressor(
                objective="quantile", alpha=percentile, verbose=-1, **model_kwargs
            )
            for percentile in self.percentiles
        ]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        for model in self.models:
            model.fit(X_train, y_train.flatten())

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.stack([model.predict(X) for model in self.models], axis=1)

    def get_params(self):
        model_params = self.models[0].get_params(deep=True)
        params = {
            **model_params,
            "percentiles": self.percentiles,
            "coverage": self.coverage,
        }
        return params


if __name__ == "__main__":
    dataset = "simple"
    model_kwargs = {
        "n_estimators": 10,
        "max_depth": -1,
        "num_leaves": None,
        "learning_rate": 0.1,
        "reg_alpha": 0.0,  # L1 regularization on weights
        "reg_lambda": 1.0,  # L2 regularization on weights
        "subsample": 1.0,  # Subsample ratio of the training instance
        "colsample_bytree": 1.0,  # Subsample ratio of columns when constructing each tree
        "min_child_weight": 1.0,  # Minimum sum of instance weight (hessian) needed in a child
        "subsample_for_bin": 200000,  # Number of samples for constructing bins
    }

    lower_percentiles = [0.03, 0.05, 0.07]
    coverage = 0.91

    model = QuantileLightGBM(dataset, model_kwargs, lower_percentiles, coverage)
    model.run_experiment()
