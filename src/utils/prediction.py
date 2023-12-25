from typing import List
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray


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


class ConformalPrediction(Prediction):
    """Predict confidence intervals based on point predictions.

    Args:
        conformal_coverage (float): coverage of the confidence intervals, must be between 0 and 1.
    """

    def __init__(self, conformal_coverage: float) -> None:
        self.conformal_coverage = conformal_coverage
        self.width = None

    def fit_conformal_width(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Fit the width of the confidence intervals based on the validation error."""
        y_pred = self.forward(X_val)
        error = y_pred - y_val
        lower_percentile = (1 - self.conformal_coverage) / 2
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
