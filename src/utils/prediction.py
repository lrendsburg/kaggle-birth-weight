from typing import List
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from src.utils.bayesian_linear_regression import BayesianLinearRegression


class Prediction(ABC):
    """Generic base class that converts any type of model output to a specific prediction (confidence intervals)."""

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass of the model. The output can be intermediate and does not need to be a confidence interval.

        Args:
            X (np.ndarray): input feature data, shape (n, d).

        Returns:
            np.ndarray: intermediate output.
        """
        pass

    @abstractmethod
    def output_to_prediction(self, output: np.ndarray) -> np.ndarray:
        """Convert the intermediate output to a prediction of confidence intervals.

        Args:
            output (np.ndarray): intermediate output.

        Returns:
            np.ndarray: confidence intervals [lower, upper], shape (n, 2).
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict confidence intervals based on the input feature data.

        Args:
            X (np.ndarray): input feature data, shape (n, d).

        Returns:
            np.ndarray: confidence intervals [lower, upper], shape (n, 2).
        """
        output = self.forward(X)
        prediction = self.output_to_prediction(output)
        return prediction

    @abstractmethod
    def get_params(self) -> dict:
        """Get the parameters of the prediction model."""
        pass


class QuantileRegression(Prediction):
    """Predict confidence intervals based on quantiles.

    Args:
        lower_percentiles (List[float]): List of lower percentiles that are predicted, must be between 0 and 1 - coverage.
            Corresponding upper percentiles are created implicitly by adding the coverage.
        coverage (float): coverage of the confidence intervals, must be between 0 and 1.
    """

    def __init__(self, lower_percentiles: List[float], coverage: float) -> None:
        self._validate_parameters(lower_percentiles, coverage)
        upper_percentiles = [p + coverage for p in lower_percentiles]
        self.percentiles = sorted(lower_percentiles + upper_percentiles)
        self.coverage = coverage
        self.num_lower = len(lower_percentiles)

    def _validate_parameters(self, percentiles, coverage) -> None:
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be between 0 and 1, got {coverage}")
        if not all(0 < p < 1 - coverage for p in percentiles):
            raise ValueError(
                f"lower percentiles must be between 0 and {1-coverage:.2f} (= 1 - coverage), got {percentiles}"
            )

    def output_to_prediction(self, output: ndarray) -> ndarray:
        """Converts quantiles to confidence intervals by minimizing the width for the given coverage.

        Args:
            output (ndarray): sorted array of predicted quantiles corresponding to self.percentiles,
                shape (n, len(self.percentiles)).

        Returns:
            ndarray: confidence intervals [lower, upper], shape (n, 2).
        """
        width = output[:, self.num_lower :] - output[:, : self.num_lower]
        best_lower_idx = np.argmin(width, axis=1)

        lower = output[np.arange(output.shape[0]), best_lower_idx]
        upper = output[np.arange(output.shape[0]), best_lower_idx + self.num_lower]
        prediction = np.stack([lower, upper], axis=1)
        return prediction

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this!")

    def get_params(self) -> dict:
        return {
            "prediction_type": "quantile",
            "percentiles": self.percentiles,
            "quantile_coverage": self.coverage,
        }


class ConformalPrediction(Prediction):
    """Predict confidence intervals based on point predictions.

    Args:
        coverage (float): coverage of the confidence intervals, must be between 0 and 1.
    """

    def __init__(self, coverage: float) -> None:
        self.coverage = coverage
        self.width = None

    def fit_conformal_width(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Fit the width of the confidence intervals based on the validation error."""
        y_pred = self.forward(X_val)
        error = y_pred - y_val
        lower_percentile = (1 - self.coverage) / 2
        upper_percentile = 1 - lower_percentile
        self.width = np.quantile(error, upper_percentile) - np.quantile(
            error, lower_percentile
        )

    def output_to_prediction(self, output: np.ndarray) -> np.ndarray:
        """Converts point predictions to confidence intervals based on validation errors.

        Args:
            output (ndarray): array of point predictions, shape (n, 1).

        Returns:
            ndarray: confidence intervals [lower, upper], shape (n, 2).
        """
        if self.width is None:
            raise RuntimeError("Width not fitted on validation set")
        lower = (output - self.width / 2).reshape(-1, 1)
        upper = (output + self.width / 2).reshape(-1, 1)
        prediction = np.hstack([lower, upper])
        return prediction

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this!")

    def get_params(self) -> dict:
        return {
            "prediction_type": "conformal",
            "conformal_coverage": self.coverage,
        }


class BayesianPrediction(Prediction):
    """Predicts confidence intervals based on Bayesian linear regression.

    The forward method is assumed to simply pre-process the data to some other
    feature space R^d -> R^d'. A Bayesian linear regression model is then fitted on
    the pre-processed data. Confidence intervals are predicted based on the posterior
    predictive distribution under this model.

    Args:
        b_0 (float): variance of the prior on the weights. Needs to be positive.
        alpha_0 (float): prior location parameter for the observation noise. Needs to be positive.
        delta_0 (float): prior scale parameter for the observation noise. Needs to be positive.
        coverage (float): coverage of the confidence intervals, must be between 0 and 1.
    """

    def __init__(self, b_0: float, alpha_0: float, delta_0: float, coverage: float):
        self.b_0 = b_0
        self.alpha_0 = alpha_0
        self.delta_0 = delta_0
        self.coverage = coverage

        self.bayesian_head = None

    def fit_bayesian_head(self, X_train: np.ndarray, y_train: np.ndarray):
        """Computes parameters for the posterior on the latent parameters given the training data.

        The latent parameters for the Bayesian linear regression model are the weights
        of the linear transformation and the observation noise. Features are pre-processed
        by the forward method before being passed to the Bayesian model.

        Args:
            X_train (np.ndarray): training feature data, shape (n, d).
            y_train (np.ndarray): training regression targets, shape (n, 1).
        """
        output_train = self.forward(X_train)
        self.bayesian_head = BayesianLinearRegression(
            b_0=self.b_0,
            alpha_0=self.alpha_0,
            delta_0=self.delta_0,
            X_train=output_train,
            y_train=y_train,
        )

    def output_to_prediction(self, output: np.ndarray) -> np.ndarray:
        """Predicts confidence intervals based on the posterior predictive distribution.

        Args:
            output (np.ndarray): Processed feature data, shape (n, d').

        Returns:
            ndarray: confidence intervals [lower, upper], shape (n, 2).
        """
        if self.bayesian_head is None:
            raise RuntimeError("Bayesian head not fitted on training set")
        prediction = self.bayesian_head.predict_confidence_interval(
            output, self.coverage
        )
        return prediction

    def get_evidence(self) -> float:
        """Returns a monotonic transformation of the evidence of the training data under
        the Bayesian model. Used for model selection.

        Returns:
            float: evidence.
        """
        if self.bayesian_head is None:
            raise RuntimeError("Bayesian head not fitted on training set")
        return self.bayesian_head.evidence()

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this!")

    def get_params(self) -> dict:
        return {
            "prediction_type": "bayesian",
            "bayesian_coverage": self.coverage,
            "weight_var": self.b_0,
            "obs_noise_loc": self.alpha_0,
            "obs_noise_scale": self.delta_0,
        }
