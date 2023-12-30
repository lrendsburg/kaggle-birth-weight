import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.experiment_pipeline import BaseExperiment
from src.models.neural_networks.utils.data_module import DataModule
from src.models.neural_networks.utils.model import IntervalNetwork
from src.models.neural_networks.utils.train import LightningModule
from src.models.neural_networks.utils.callbacks import MetricsCallback
from src.utils.config import get_params


class DirectNN(BaseExperiment):
    """Fully connected neural network that directly predicts confidence intervals.

    Args:
        dataset (str): The dataset (preprocessing method) to be used.
        coverage (float): The coverage of the confidence interval on the train set.
    """

    def __init__(self, dataset: str, model_kwargs: dict, trial: optuna.trial.Trial):
        super().__init__(dataset)
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
        self.model = IntervalNetwork(self.model_kwargs, X_train, y_train)
        lightning_model = LightningModule(self.model, self.model_kwargs, loss="winkler")
        callbacks = [
            MetricsCallback("coverage"),
            # PyTorchLightningPruningCallback(self.trial, monitor="loss/val"),
            EarlyStopping(
                monitor="loss/val",
                min_delta=0.00,
                patience=15,
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        confidence_intervals = (
            self.model.predict(torch.from_numpy(X).type(torch.float32)).detach().numpy()
        )
        return confidence_intervals

    def get_params(self) -> dict:
        return {"prediction_type": "direct", **self.model_kwargs}


def objective(trial):
    dataset, model_kwargs, _ = get_params(trial, "neural_network")
    model = DirectNN(dataset, model_kwargs, trial)
    score = model.run_experiment()
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        objective,
        n_trials=1,
        # timeout=3600,
    )

    best_params, best_value = study.best_params, study.best_value
    print(f"\n{best_value=} at {best_params=}")
