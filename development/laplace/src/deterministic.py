import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

##################################################
################ DATA MODULE #####################
##################################################


class FlatDataModule(pl.LightningDataModule):
    def __init__(self, X, Y, batch_size=32):
        super().__init__()
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = TensorDataset(self.X, self.Y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class HierarchicalDataModule(pl.LightningDataModule):
    def __init__(self, X, group, Y, batch_size=32):
        super().__init__()
        self.X = torch.tensor(X)
        self.group = torch.tensor(group)
        self.Y = torch.tensor(Y)
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = TensorDataset(self.X, self.group, self.Y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


##################################################
################ ARCHITECTURE ####################
##################################################


class Network(nn.Module):
    def __init__(self, hidden_dim=20, n_layers=2):
        super().__init__()

        hidden_layers = [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_layers - 1)
        self.layers = nn.ModuleList(
            [
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                *hidden_layers,
                nn.Linear(hidden_dim, 1),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class HierarchicalWrapper(nn.Module):
    def __init__(self, model_class, model_kwargs, num_groups):
        super().__init__()

        self.global_model = model_class(**model_kwargs)
        self.local_models = nn.ModuleList()
        for _ in range(num_groups):
            local_model = model_class(**model_kwargs)
            local_model.load_state_dict(self.global_model.state_dict())
            self.local_models.append(local_model)

    def forward(self, x, group):
        y_pred = torch.zeros(len(x), 1, device=x.device)
        for i, local_model in enumerate(self.local_models):
            mask = group == i
            if mask.any():
                y_pred[mask] = local_model(x[mask])
        return y_pred


##################################################
################### TRAINER ######################
##################################################


class FlatLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model

        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.mse_loss(y, y_pred)
        self.log("train_loss", loss)
        return loss


class HierarchicalLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate, weight_decay_global, weight_decay_local):
        super().__init__()
        self.model = model

        self.learning_rate = learning_rate
        self.weight_decay_global = weight_decay_global
        self.weight_decay_local = weight_decay_local

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, group, y = batch
        y_pred = self.model(x, group)
        loss = self.hierarchical_loss(y, y_pred)
        self.log("train_loss", loss)
        return loss

    def hierarchical_loss(self, y, y_pred):
        data_loss = F.mse_loss(y, y_pred)

        global_params = self._flatten_parameters(self.model.global_model)
        global_reg = F.mse_loss(
            global_params, torch.zeros_like(global_params), reduction="sum"
        )

        local_reg = 0
        for local_model in self.model.local_models:
            local_params = self._flatten_parameters(local_model)
            local_reg += F.mse_loss(local_params, global_params, reduction="sum")

        loss = (
            data_loss
            + self.weight_decay_global * global_reg
            + self.weight_decay_local * local_reg
        )
        return loss

    def _flatten_parameters(self, model):
        return torch.cat([p.flatten() for p in model.parameters()])


##################################################
################## CALLBACKS #####################
##################################################


class LogTrainErrorCallback(Callback):
    def __init__(self):
        self.train_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = pl_module.trainer.logged_metrics["train_loss"]
        self.train_losses.append(train_loss)

    def on_fit_end(self, trainer, pl_module):
        plt.plot(self.train_losses)
        plt.title("Train Loss vs. Epochs")
        plt.show()


class PlotPredictionCallback(Callback):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def on_fit_end(self, trainer, pl_module):
        grid = torch.linspace(
            np.min(self.X) - 0.2, np.max(self.X) + 0.2, 100
        ).unsqueeze(1)

        pl_module.model.eval()
        with torch.no_grad():
            predictions = pl_module.model(grid).numpy()

        plt.scatter(self.X, self.Y, label="Original Data")
        plt.plot(grid.numpy(), predictions, label="Predicted Function", color="red")
        plt.legend()
        plt.title("Predicted Function and Original Training Data")
        plt.show()


class PlotGroupPredictionCallback(Callback):
    def __init__(self, X, group, Y):
        self.X = X
        self.group = group
        self.Y = Y
        self.num_groups = len(set(group))

    def on_fit_end(self, trainer, pl_module):
        fig, axs = plt.subplots(1, self.num_groups, figsize=(5 * self.num_groups, 5))
        for group_idx in range(self.num_groups):
            group_mask = self.group == group_idx
            X_group = self.X[group_mask]
            Y_group = self.Y[group_mask]

            self._plot_predictions(
                X_group, Y_group, group_idx, pl_module.model, axs[group_idx]
            )
        plt.tight_layout()
        plt.show()

    def _plot_predictions(self, X, Y, group_idx, model, ax):
        x_min, x_max = np.min(self.X) - 0.2, np.max(self.X) + 0.2
        y_min, y_max = np.min(self.Y) - 0.2, np.max(self.Y) + 0.2
        grid = torch.linspace(x_min, x_max, 100).unsqueeze(1)

        model.eval()
        with torch.no_grad():
            group = torch.full_like(grid, group_idx).squeeze()
            predictions = model(grid, group).numpy()

        ax.scatter(X, Y, label="Original Data")
        ax.plot(grid.numpy(), predictions, label="Predicted Function", color="red")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()
        ax.set_title(f"Predictions for group {group_idx}")
