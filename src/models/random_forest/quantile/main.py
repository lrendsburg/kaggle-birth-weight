from typing import List

import numpy as np
from quantile_forest import RandomForestQuantileRegressor

from src.experiment_pipeline import BaseExperiment
from src.utils.prediction import QuantileRegression


class QuantileForest(QuantileRegression, BaseExperiment):
    """Random forest trained with the square loss to predict a given set of quantiles.
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

        self.model = RandomForestQuantileRegressor(**model_kwargs)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.model.fit(X_train, y_train.flatten())

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, self.percentiles)

    def get_params(self):
        model_params = self.model.get_params(deep=True)
        prediction_params = QuantileRegression.get_params(self)
        return {**model_params, **prediction_params}


if __name__ == "__main__":
    dataset = "simple"
    model_kwargs = {
        "n_estimators": 10,
        "max_depth": 10,
        "max_features": 1.0,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }

    lower_percentiles = [0.01, 0.03, 0.05, 0.07, 0.09]
    coverage = 0.9

    model = QuantileForest(dataset, model_kwargs, lower_percentiles, coverage)
    model.run_experiment()
