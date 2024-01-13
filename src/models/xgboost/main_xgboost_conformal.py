import argparse

import numpy as np
from xgboost import XGBRegressor
import optuna

from src.experiment_pipeline import BaseExperiment
from src.prediction.prediction import ConformalPrediction
from src.utils.config import get_params


class ConformalXGBoost(ConformalPrediction, BaseExperiment):
    """XGBoost trained with the square loss to make point predictions. Predictions
    are then upgraded to confidence intervals using conformal prediction.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        model_kwargs (dict): Parameters of the model.
        prediction_kwargs (dict): Parameters of the prediction head.
    """

    def __init__(
        self, dataset: str, model_kwargs: dict, prediction_kwargs: dict
    ) -> None:
        ConformalPrediction.__init__(self, **prediction_kwargs)
        BaseExperiment.__init__(self, dataset)

        self.model = XGBRegressor(**model_kwargs)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.model.fit(X_train, y_train.flatten())
        self.fit_conformal_width(X_val, y_val)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).reshape(-1, 1)

    def get_params(self):
        model_params = self.model.get_params(deep=True)
        prediction_params = ConformalPrediction.get_params(self)
        return {**model_params, **prediction_params}


def objective(trial):
    dataset, model_kwargs, prediction_kwargs = get_params(trial, "xgboost", "conformal")
    model = ConformalXGBoost(dataset, model_kwargs, prediction_kwargs)
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
