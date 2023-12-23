import numpy as np


class Metrics:
    def __init__(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        self.y = y.flatten()
        self.lower, self.upper = y_pred.T

    def compute_metrics(self) -> dict[str, float]:
        """Compute a set of pre-defined metrics between targets and prediction intervals.

        Returns:
            dict[str, float]: dictionary of metrics.
        """
        metrics = {
            "winkler": self._winkler_interval_score(),
            "coverage": self._coverage(),
        }
        return metrics

    def _winkler_interval_score(self) -> float:
        alpha = 0.1  # pre-set by competition host
        width = self.upper - self.lower

        excess_upper = np.maximum(0, self.y - self.upper)
        excess_lower = np.maximum(0, self.lower - self.y)
        total_excess = excess_upper + excess_lower

        score = (width + (2 / alpha) * total_excess).mean()
        return score

    def _coverage(self) -> float:
        is_covered = (self.lower <= self.y) & (self.y <= self.upper)
        coverage = is_covered.mean()
        return coverage
