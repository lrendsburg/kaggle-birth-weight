import warnings

import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD

import pytorch_lightning as pl


warnings.filterwarnings("ignore", ".*TracerWarning.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Set a lower value for log_every_n_steps.*")


class LightningModule(pl.LightningModule):
    def __init__(self, model, params, loss: str):
        super().__init__()
        self.model = model
        self.params = params

        self.optimizer = {"Adam": Adam, "SGD": SGD}[params["optimizer"]]
        self.loss = {
            "mse": F.mse_loss,
            "mae": F.l1_loss,
            "winkler": self.winkler_loss,
        }[loss]

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.params["learning_rate"])

    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "y": y, "pred": pred}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def winkler_loss(self, pred: torch.tensor, y: torch.tensor, alpha: float = 0.1):
        mean, width = pred[:, 0], pred[:, 1]
        lower = mean - width / 2
        upper = mean + width / 2

        excess_upper = F.relu(y - upper)
        excess_lower = F.relu(lower - y)
        excess = excess_upper + excess_lower

        winkler_score = width + (2 / alpha) * excess
        return torch.mean(winkler_score)
