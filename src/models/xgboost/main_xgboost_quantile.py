from typing import List
import argparse

import numpy as np
from xgboost import XGBRegressor
import optuna

from src.experiment_pipeline import BaseExperiment
from src.prediction.prediction import QuantileRegression
from src.utils.config import get_params


class QuantileXGBoost(QuantileRegression, BaseExperiment):
    """XGBoost trained with the square loss to predict a given set of quantiles.
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

        self.models = [
            XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=percentile, **model_kwargs
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
        prediction_params = QuantileRegression.get_params(self)
        return {**model_params, **prediction_params}


def objective(trial):
    dataset, model_kwargs, prediction_kwargs = get_params(trial, "xgboost", "quantile")
    model = QuantileXGBoost(dataset, model_kwargs, prediction_kwargs)
    score = model.run_experiment()
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize with a timeout.")
    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout for optimization in seconds"
    )
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=args.timeout)

    best_params, best_value = study.best_params, study.best_value
    print(f"\n{best_value=} at {best_params=}")
