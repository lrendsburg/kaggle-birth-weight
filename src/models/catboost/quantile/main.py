from typing import List

import numpy as np
from catboost import CatBoostRegressor

from src.experiment_pipeline import BaseExperiment
from src.utils.prediction import QuantileRegression


class QuantileCatBoost(QuantileRegression, BaseExperiment):
    """CatBoost trained with the square loss to predict a given set of quantiles.
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
            CatBoostRegressor(
                loss_function=f"Quantile:alpha={percentile}",
                **model_kwargs,
                verbose=False,
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


if __name__ == "__main__":
    dataset = "simple"
    model_kwargs = {
        "iterations": 100,  # The maximum number of trees that can be built.
        "learning_rate": 0.03,  # The learning rate used for reducing the gradient step.
        "depth": 6,  # Depth of the tree.
        "l2_leaf_reg": 3.0,  # Coefficient at the L2 regularization term of the cost function.
        "border_count": 32,  # The number of splits for numerical features.
    }
    prediction_kwargs = {
        "lower_percentiles": [0.03, 0.05, 0.07],
        "coverage": 0.91,
    }

    model = QuantileCatBoost(dataset, model_kwargs, prediction_kwargs)
    model.run_experiment()
