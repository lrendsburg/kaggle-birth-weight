import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.experiment_pipeline import BaseExperiment
from src.utils.prediction import BayesianPrediction
from src.models.neural_networks.utils.data_module import DataModule
from src.models.neural_networks.utils.model import RegressionNetwork
from src.models.neural_networks.utils.train import LightningModule


class BayesianNN(BayesianPrediction, BaseExperiment):
    """Fully connected neural network trained with the square loss to make point predictions.
    After training, a Bayesian linear regression model is fitted on the last layer, whose
    posterior predictive is then used to predict confidence intervals.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        model_kwargs (dict): Parameters of the model.
        prediction_kwargs (dict): Parameters of the prediction head.
    """

    def __init__(
        self, dataset: str, model_kwargs: dict, prediction_kwargs: dict
    ) -> None:
        BayesianPrediction.__init__(self, **prediction_kwargs)
        BaseExperiment.__init__(self, dataset)
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
        self.model = RegressionNetwork(self.model_kwargs, X_train, y_train)
        lightning_model = LightningModule(self.model, self.model_kwargs, loss="mae")

        # Train
        trainer = pl.Trainer(
            max_epochs=self.model_kwargs["max_epochs"],
            logger=WandbLogger(project="kaggle-birth-weight", config=self.get_params()),
            enable_progress_bar=True,
        )
        trainer.fit(lightning_model, data_module)

        self.fit_bayesian_head(X_train, y_train)

    def forward(self, X: np.ndarray) -> np.ndarray:
        hidden = self.model.predict(
            torch.from_numpy(X).type(torch.float32), return_last_layer=True
        )
        return hidden.detach().numpy()

    def get_params(self) -> dict:
        prediction_params = BayesianPrediction.get_params(self)
        return {**self.model_kwargs, **prediction_params}


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
    prediction_kwargs = {
        "b_0": 1.00,
        "alpha_0": 1.0,
        "delta_0": 1.0,
        "coverage": 0.90,
    }

    model = BayesianNN(dataset, model_kwargs, prediction_kwargs)
    model.run_experiment()
