import numpy as np
from scipy.special import loggamma
from scipy.stats import t


class BayesianLinearRegression:
    """Bayesian linear regression with conjugate prior on weights and observation noise.

    The model assumes the special case of a Gaussian prior on the weights with zero mean
    and a diagonal covariance matrix. The observation noise has a conjugate inverse Gamma
    prior, and the likelihood is a linear function with Gaussian noise.

    The computations are based on a singular value decomposition of the feature matrix X.

    Args:
        b_0 (float): variance of the prior on the weights. Needs to be positive.
        alpha_0 (float): prior location parameter for the observation noise. Needs to be positive.
        delta_0 (float): prior scale parameter for the observation noise. Needs to be positive.
        X_train (np.ndarray): training features
        y_train (np.ndarray): training targets
        bias (bool): whether to include a bias term in the model. Defaults to True.
    """

    def __init__(
        self,
        b_0: float,
        alpha_0: float,
        delta_0: float,
        X_train: np.ndarray,
        y_train: np.ndarray,
        bias: bool = True,
    ):
        if b_0 <= 0:
            raise ValueError(f"Argument b_0 must be positive. Got {b_0}")
        if alpha_0 <= 0:
            raise ValueError(f"Argument alpha_0 must be positive. Got {alpha_0}")
        if delta_0 <= 0:
            raise ValueError(f"Argument delta_0 must be positive. Got {delta_0}")
        self.b_0 = b_0
        self.alpha_0 = alpha_0
        self.delta_0 = delta_0

        self.bias = bias
        if self.bias:
            X_train = self._add_constant_feature(X_train)

        self.fit(X_train, y_train.reshape(-1, 1))

    def _add_constant_feature(self, X: np.ndarray) -> np.ndarray:
        """Add a constant feature to the feature matrix."""
        return np.hstack((X, np.ones((len(X), 1))))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute parameters needed for posterior inference and evidence given the training data."""
        # SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt.T
        self.lam = s**2

        # Auxiliary quantities
        UTy = U.T @ y
        self.n, self.d = X.shape

        # Posterior parameters
        self.B_n = V @ np.diag(1 / (1 / self.b_0 + self.lam)) @ V.T
        self.beta_n = V @ np.diag(s / (1 / self.b_0 + self.lam)) @ UTy
        self.alpha_n = self.alpha_0 + self.n
        self.delta_n = (
            self.delta_0
            + y.T @ y
            - UTy.T @ np.diag(self.lam / (1 / self.b_0 + self.lam)) @ UTy
        ).squeeze()

    def evidence(self) -> float:
        """Compute (a monotonic transformation of the) evidence for the training data."""
        score = (
            self.alpha_0 * np.log(self.delta_0)
            - self.alpha_n * np.log(self.delta_n)
            + np.sum(np.log(1 / self.b_0 + self.lam))
            - self.d * np.log(self.b_0)
            + 2 * loggamma(self.alpha_n / 2)
            - 2 * loggamma(self.alpha_0 / 2)
        )
        return score

    def predict_confidence_interval(self, X: np.ndarray, coverage: float) -> np.ndarray:
        """Predicts confidence intervals to a given coverage for each test point."""
        if not 0 < coverage < 1:
            raise ValueError(
                f"Argument coverage must be between 0 and 1. Got {coverage}"
            )

        if self.bias:
            X = self._add_constant_feature(X)

        post_pred_mean = X @ self.beta_n
        post_pred_var = (
            self.delta_n / self.alpha_n * (1 + np.einsum("ij,jk,ik->i", X, self.B_n, X))
        )

        lower, upper = t.interval(
            coverage,
            df=self.alpha_n,
            loc=post_pred_mean.flatten(),
            scale=np.sqrt(post_pred_var).flatten(),
        )
        confidence_interval = np.stack([lower, upper], axis=1)
        return confidence_interval

    def predict_MAP(self, X: np.ndarray) -> np.ndarray:
        """Predict maximum a posteriori estimate for each test point."""
        if self.bias:
            X = self._add_constant_feature(X)

        post_pred_mean = X @ self.beta_n
        return post_pred_mean
