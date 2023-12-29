from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class MLPBase(nn.Module, ABC):
    def __init__(self, params: dict):
        """Base class for MLPs. Implements the forward pass until the last hidden layer.
        The head of the network is implemented in the child class.

        Args:
            params (dict): parameter dictionary containing the following keys:
                - in_size (int): input size
                - layer_widths (list[int]): hidden layer widths for the base network
        """
        super().__init__()
        layers = [nn.Linear(params["in_size"], params["layer_widths"][0])]
        for in_features, out_features in zip(
            params["layer_widths"], params["layer_widths"][1:]
        ):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def base(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.layers(x)
        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.base(x)
        out = self.head(hidden)
        return out

    @abstractmethod
    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


class LinearWithInit(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_init: float = 0.0,
        weight_factor: float = 0.1,
    ):
        super().__init__(in_features, out_features)
        self.weight.data *= weight_factor
        nn.init.constant_(self.bias, bias_init)


class RegressionNetwork(MLPBase):
    """MLP that predicts the continuous target directly.

    Args:
        params (dict): parameter dictionary containing the following keys:
            - in_size (int): input size
            - layer_widths (list[int]): hidden layer widths for the base network
        X_train (np.ndarray): train features. Used to compute in_size.
        y_train (np.ndarray): train target. Used to initialize the head.
    """

    def __init__(self, params: dict, X_train: np.ndarray, y_train: np.ndarray):
        X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
        params["in_size"] = X_train.shape[1]
        super().__init__(params)

        self._initialize_head(params["layer_widths"][-1], y_train)

    def _initialize_head(self, hidden_width: int, y_train: torch.Tensor):
        head_bias_init = y_train.mean()
        self.head_layer = LinearWithInit(hidden_width, 1, head_bias_init)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head_layer(hidden)

    def predict(self, x: torch.Tensor, return_last_layer: bool = False) -> torch.Tensor:
        if return_last_layer:
            return self.base(x)
        else:
            return self.forward(x)


class IntervalNetwork(MLPBase):
    """MLP that predicts confidence intervals directly, parameterized via (mean, width).

    Args:
        params (dict): parameter dictionary containing the following keys:
            - in_size (int): input size
            - layer_widths (list[int]): hidden layer widths for the base network
        X_train (np.ndarray): train features. Used to compute in_size.
        y_train (np.ndarray): train target. Used to initialize the heads.
    """

    def __init__(self, params: dict, X_train: np.ndarray, y_train: np.ndarray):
        X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
        params["in_size"] = X_train.shape[1]
        super().__init__(params)

        self._initialize_heads(params["layer_widths"][-1], y_train)

    def _initialize_heads(self, hidden_width: int, y_train: torch.Tensor):
        mean_bias_init = y_train.mean()
        width_bias_init = self._inverse_softplus(
            torch.quantile(y_train, 0.95) - torch.quantile(y_train, 0.05)
        )

        self.mean_head = LinearWithInit(hidden_width, 1, mean_bias_init)
        self.width_head = nn.Sequential(
            LinearWithInit(hidden_width, 1, width_bias_init),
            nn.Softplus(),
        )

    def _inverse_softplus(self, y: torch.Tensor, beta=1) -> torch.Tensor:
        return y + 1 / beta * torch.log(-torch.expm1(-beta * y))

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        mean_output = self.mean_head(hidden)
        width_output = self.width_head(hidden)
        return torch.cat((mean_output, width_output), dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        mean, width = out[:, 0], out[:, 1]
        lower = mean - width / 2
        upper = mean + width / 2
        return torch.stack((lower, upper), dim=1)
