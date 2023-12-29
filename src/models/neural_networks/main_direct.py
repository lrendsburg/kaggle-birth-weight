import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.experiment_pipeline import BaseExperiment
from src.models.neural_networks.utils.data_module import DataModule
from src.models.neural_networks.utils.model import IntervalNetwork
from src.models.neural_networks.utils.train import LightningModule
from src.models.neural_networks.utils.callbacks import MetricsCallback


class DirectNN(BaseExperiment):
    """Fully connected neural network that directly predicts confidence intervals.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        coverage (float): The coverage of the confidence interval on the train set.
    """

    def __init__(self, dataset: str, model_kwargs: dict):
        super().__init__(dataset)
        self.dataset = dataset
        self.model_kwargs = model_kwargs

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        # Setup
        data_module = DataModule(
            X_train, y_train, X_val, y_val, self.model_kwargs["batch_size"]
        )
        self.model = IntervalNetwork(self.model_kwargs, X_train, y_train)
        lightning_model = LightningModule(self.model, self.model_kwargs, loss="winkler")

        # Train
        trainer = pl.Trainer(
            max_epochs=self.model_kwargs["max_epochs"],
            callbacks=[MetricsCallback("coverage")],
            logger=WandbLogger(project="kaggle-birth-weight", config=self.get_params()),
            enable_progress_bar=True,
        )
        trainer.fit(lightning_model, data_module)

    def predict(self, X: np.ndarray) -> np.ndarray:
        confidence_intervals = (
            self.model.predict(torch.from_numpy(X).type(torch.float32)).detach().numpy()
        )
        return confidence_intervals

    def get_params(self) -> dict:
        return {"prediction_type": "direct", **self.model_kwargs}


if __name__ == "__main__":
    dataset = "simple"
    model_kwargs = {
        # Architecture
        "layer_widths": [30, 20, 10],
        # Training
        "optimizer": "Adam",
        "batch_size": 256,
        "max_epochs": 2,
        "learning_rate": 1e-3,
    }

    model = DirectNN(dataset, model_kwargs)
    model.run_experiment()
