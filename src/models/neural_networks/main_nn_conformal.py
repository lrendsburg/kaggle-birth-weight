import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.experiment_pipeline import BaseExperiment
from src.prediction.prediction import ConformalPrediction
from src.models.neural_networks.utils.data_module import DataModule
from src.models.neural_networks.utils.model import RegressionNetwork
from src.models.neural_networks.utils.train import LightningModule
from src.utils.config import get_params


class ConformalNN(ConformalPrediction, BaseExperiment):
    """Fully connected neural network trained with the square loss to make point predictions.
    Predictions are then upgraded to confidence intervals using conformal prediction.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        model_kwargs (dict): Parameters of the model.
        prediction_kwargs (dict): Parameters of the prediction head.
    """

    def __init__(
        self,
        dataset: str,
        model_kwargs: dict,
        prediction_kwargs: dict,
        trial: optuna.trial.Trial,
    ) -> None:
        ConformalPrediction.__init__(self, **prediction_kwargs)
        BaseExperiment.__init__(self, dataset)
        self.dataset = dataset
        self.model_kwargs = model_kwargs
        self.trial = trial

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
        callbacks = [
            PyTorchLightningPruningCallback(self.trial, monitor="loss/val"),
            EarlyStopping(
                monitor="loss/val",
                min_delta=0.00,
                patience=10,
                mode="min",
            ),
        ]

        # Train
        trainer = pl.Trainer(
            max_epochs=self.model_kwargs["max_epochs"],
            callbacks=callbacks,
            logger=WandbLogger(project="kaggle-birth-weight", config=self.get_params()),
            enable_progress_bar=True,
        )
        trainer.fit(lightning_model, data_module)

        self.fit_conformal_width(X_val, y_val)

    def forward(self, X: np.ndarray) -> np.ndarray:
        pred = self.model.predict(torch.from_numpy(X).type(torch.float32))
        return pred.view(-1, 1).detach().numpy()

    def get_params(self) -> dict:
        prediction_params = ConformalPrediction.get_params(self)
        return {**self.model_kwargs, **prediction_params}


def objective(trial):
    dataset, model_kwargs, prediction_kwargs = get_params(
        trial, "neural_network", "conformal"
    )
    model = ConformalNN(dataset, model_kwargs, prediction_kwargs, trial)
    score = model.run_experiment()
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize with a timeout.")
    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout for optimization in seconds"
    )
    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, timeout=args.timeout)

    best_params, best_value = study.best_params, study.best_value
    print(f"\n{best_value=} at {best_params=}")
