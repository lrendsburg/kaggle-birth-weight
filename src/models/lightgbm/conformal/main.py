import numpy as np
from lightgbm import LGBMRegressor

from src.experiment_pipeline import BaseExperiment
from src.utils.prediction import ConformalPrediction


class ConformalLightGBM(ConformalPrediction, BaseExperiment):
    """LightGBM trained with the square loss to make point predictions. Predictions
    are then upgraded to confidence intervals using conformal prediction.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        model_kwargs (dict): Parameters of the model.
        coverage (float): The coverage of the confidence interval on the validation set.
    """

    def __init__(self, dataset: str, model_kwargs: dict, coverage: float) -> None:
        ConformalPrediction.__init__(self, coverage)
        BaseExperiment.__init__(self, dataset)

        self.model = LGBMRegressor(verbose=-1, **model_kwargs)
        self.coverage = coverage

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

    coverage = 0.90

    model = ConformalLightGBM(dataset, model_kwargs, coverage)
    model.run_experiment()
