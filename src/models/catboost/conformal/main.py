import numpy as np
from catboost import CatBoostRegressor

from src.experiment_pipeline import BaseExperiment
from src.utils.prediction import ConformalPrediction


class ConformalCatBoost(ConformalPrediction, BaseExperiment):
    """CatBoost trained with the square loss to make point predictions. Predictions
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

        self.model = CatBoostRegressor(**model_kwargs, verbose=False)

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
        "coverage": 0.9,
    }

    model = ConformalCatBoost(dataset, model_kwargs, prediction_kwargs)
    model.run_experiment()
