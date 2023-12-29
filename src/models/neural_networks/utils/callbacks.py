from typing import Union, List

import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import Callback


class Coverage(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_covered", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        mean, width = pred[:, 0], pred[:, 1]
        lower = mean - width / 2
        upper = mean + width / 2

        covered = (lower <= target.flatten()) & (target.flatten() <= upper)

        self.total_covered += torch.sum(covered)
        self.total_samples += target.numel()

    def compute(self):
        coverage = self.total_covered.float() / self.total_samples
        return coverage


class MetricsCallback(Callback):
    METRICS = {
        "mse": torchmetrics.MeanSquaredError,
        "mae": torchmetrics.MeanAbsoluteError,
        "coverage": Coverage,
    }

    def __init__(self, metrics: Union[str, List[str]]):
        metric_names = [metrics] if isinstance(metrics, str) else metrics
        metrics = {
            k: v for k, v in MetricsCallback.METRICS.items() if k in metric_names
        }
        self.train_metrics = {name: Metric() for name, Metric in metrics.items()}
        self.val_metrics = {name: Metric() for name, Metric in metrics.items()}

    def update_metrics(self, metrics, outputs, batch):
        y, pred = outputs["y"], outputs["pred"]
        for metric in metrics.values():
            metric.update(pred, y)

    def log_metrics(self, metrics, trainer, stage):
        if not trainer.logger:
            return
        for name, metric in metrics.items():
            epoch_metric = metric.compute()
            trainer.logger.log_metrics(
                {f"{name}/{stage}": epoch_metric}, step=trainer.current_epoch
            )
            metric.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.update_metrics(self.train_metrics, outputs, batch)

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_metrics(self.train_metrics, trainer, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.update_metrics(self.val_metrics, outputs, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_metrics(self.val_metrics, trainer, "val")
