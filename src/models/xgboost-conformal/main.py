import numpy as np
from xgboost import XGBRegressor

from src.experiment_pipeline import BaseExperiment
from src.utils.prediction import ConformalPrediction


class ConformalXGBoost(ConformalPrediction, BaseExperiment):
    """XGBoost trained with the square loss to make point predictions. Predictions
    are then upgraded to confidence intervals using conformal prediction.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        model_kwargs (dict): Parameters of the model.
        conformal_coverage (float): The coverage of the confidence interval on the validation set.
    """

    def __init__(
        self, dataset: str, model_kwargs: dict, conformal_coverage: float
    ) -> None:
        ConformalPrediction.__init__(self, conformal_coverage)
        BaseExperiment.__init__(self, dataset)

        self.model = XGBRegressor(**model_kwargs)
        self.conformal_coverage = conformal_coverage

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
        return {**model_params, "conformal_coverage": self.conformal_coverage}


if __name__ == "__main__":
    dataset = "simple"
    model_kwargs = {
        "n_estimators": 10,
        "max_depth": 10,
        "max_leaves": None,
        "learning_rate": 0.1,
        "reg_alpha": 0.0,  # L1 regularization on weights
        "reg_lambda": 1.0,  # L2 regularization on weights
        "subsample": 1.0,  # Subsample ratio of the training instance
        "colsample_bytree": 1.0,  # Subsample ratio of columns when constructing each tree
        "colsample_bylevel": 1.0,  # Subsample ratio of columns for each split, in each level
        "colsample_bynode": 1.0,  # Subsample ratio of columns for each split, in each node
        "min_child_weight": 1.0,  # Minimum sum of instance weight (hessian) needed in a child
    }

    conformal_coverage = 0.90

    model = ConformalXGBoost(dataset, model_kwargs, conformal_coverage)
    model.run_experiment()
