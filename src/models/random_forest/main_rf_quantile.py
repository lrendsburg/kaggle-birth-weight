from typing import List

import numpy as np
from quantile_forest import RandomForestQuantileRegressor
import optuna

from src.experiment_pipeline import BaseExperiment
from src.utils.prediction import QuantileRegression
from src.utils.config import get_params


class QuantileForest(QuantileRegression, BaseExperiment):
    """Random forest trained with the square loss to predict a given set of quantiles.
    Confidence intervals are created from quantiles by minimizing their width for a given coverage.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        model_kwargs (dict): Parameters of the model.
        prediction_kwargs (dict): Parameters of the prediction head.
    """

    def __init__(
        self, dataset: str, model_kwargs: dict, prediction_kwargs: dict
    ) -> None:
        QuantileRegression.__init__(self, **prediction_kwargs)
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


def objective(trial):
    dataset, model_kwargs, prediction_kwargs = get_params(
        trial, "random_forest", "quantile"
    )
    model = QuantileForest(dataset, model_kwargs, prediction_kwargs)
    score = model.run_experiment()
    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=1,
        # timeout=3600,
    )

    best_params, best_value = study.best_params, study.best_value
    print(f"\n{best_value=} at {best_params=}")
