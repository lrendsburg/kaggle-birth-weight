import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int,
    ):
        super().__init__()
        self.train_dataset = self._make_dataset(X_train, y_train)
        self.val_dataset = self._make_dataset(X_val, y_val)
        self.batch_size = batch_size

    def _make_dataset(self, X: np.ndarray, y: np.ndarray):
        return TensorDataset(torch.Tensor(X), torch.tensor(y).view(-1, 1))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
