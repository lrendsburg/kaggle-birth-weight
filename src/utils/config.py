from typing import Tuple, Optional

from pathlib import Path
import yaml


def read_optuna_yaml(file_path: str, trial) -> dict:
    with open(file_path, "r") as file:
        optuna_search_space = yaml.safe_load(file)

    params = {}
    for name, param in optuna_search_space.items():
        if ("value" in param.keys()) == ("type" in param.keys()):
            raise ValueError(
                f"Invalid search space for {name}. Choose either 'value' or 'type'"
            )

        if "value" in param.keys():
            params[name] = param["value"]
        elif param["type"] == "int":
            params[name] = trial.suggest_int(
                name,
                float(param["low"]),
                float(param["high"]),
                log=param.get("log", False),
            )
        elif param["type"] == "float":
            params[name] = trial.suggest_float(
                name,
                float(param["low"]),
                float(param["high"]),
                log=param.get("log", False),
            )
        elif param["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, param["choices"])
    return params


def get_dataset_params(trial) -> dict:
    return read_optuna_yaml(Path("conf", f"dataset.yaml"), trial)["dataset"]


def get_model_params(model: str, trial) -> dict:
    valid_models = [
        f.name.removesuffix(".yaml") for f in Path("conf", "models").iterdir()
    ]
    if model not in valid_models:
        raise ValueError(f"Invalid model {model}. Valid models are {valid_models}")
    return read_optuna_yaml(Path("conf", "models", f"{model}.yaml"), trial)


def get_prediction_params(prediction: str, trial) -> dict:
    valid_prediction_types = [
        f.name.removesuffix(".yaml") for f in Path("conf", "prediction").iterdir()
    ]
    if prediction not in valid_prediction_types:
        raise ValueError(
            f"Invalid prediction type {prediction}. Choose from {valid_prediction_types}"
        )
    return read_optuna_yaml(Path("conf", "prediction", f"{prediction}.yaml"), trial)


def get_params(
    trial, model: str, prediction: Optional[str] = None
) -> Tuple[dict, dict, dict]:
    dataset_params = get_dataset_params(trial)
    model_params = get_model_params(model, trial)
    prediction_params = get_prediction_params(prediction, trial) if prediction else None
    return dataset_params, model_params, prediction_params
