from typing import List

import numpy as np
from quantile_forest import RandomForestQuantileRegressor

from src.experiment_pipeline import BaseExperiment
from src.utils.prediction import QuantileRegression


class QuantileForest(QuantileRegression, BaseExperiment):
    """Random forest trained with the square loss to make point predictions. Predictions
    are then upgraded to confidence intervals using conformal prediction.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        model_kwargs (dict): Parameters of the random forest.
        conformal_coverage (float): The coverage of the confidence interval on the validation set.
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
        # return self.model.predict(X).reshape(-1, 1)
        #   y_pred = qrf.predict(X, quantiles=[0.025, 0.5, 0.975])
        return self.model.predict(X, self.percentiles)

    def get_params(self):
        forest_params = self.model.get_params(deep=True)
        params = {
            **forest_params,
            "percentiles": self.percentiles,
            "coverage": self.coverage,
        }
        return params


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
