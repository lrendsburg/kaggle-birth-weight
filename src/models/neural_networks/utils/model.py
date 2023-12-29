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
