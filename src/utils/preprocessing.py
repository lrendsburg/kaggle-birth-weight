from typing import List, Optional, Tuple
import os
from pathlib import Path
import logging

from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
import joblib


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def process_data(
    method_name: str,
    pipeline: TransformerMixin,
    target_transform: Optional[TransformerMixin] = None,
) -> None:
    """Main preprocessing function. Loads data, applies transforms, and saves results.

    Args:
        method_name (str): Name of the preprocessing method.
        pipeline (TransformerMixin): Preprocessing pipeline to be applied to the features.
        target_transform (Optional[TransformerMixin], optional): Transforms for the
        target variables. Defaults to None.
    """
    df_train = pd.read_csv("datasets/train.csv")
    df_val = pd.read_csv("datasets/val.csv")
    df_test = pd.read_csv("datasets/test.csv")

    target_name = "DBWT"
    y_train = df_train.pop(target_name).to_numpy().astype(np.float32).reshape(-1, 1)
    y_val = df_val.pop(target_name).to_numpy().astype(np.float32).reshape(-1, 1)

    folder_path = f"datasets/{method_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if target_transform is not None:
        y_train = target_transform.fit_transform(y_train)
        y_val = target_transform.transform(y_val)
        joblib.dump(target_transform, Path(folder_path, "target_transform.save"))

    X_train = pipeline.fit_transform(df_train)
    X_val = pipeline.transform(df_val)
    X_test = pipeline.transform(df_test)

    logging.info(f"Complete method '{method_name}' with {X_train.shape[1]} features.")

    for data, name in zip(
        [X_train, X_val, X_test, y_train, y_val],
        ["X_train", "X_val", "X_test", "y_train", "y_val"],
    ):
        np.save(Path(folder_path, f"{name}.npy"), data)

    logging.info(f"Saving results to '{folder_path}'.")


def get_columns_by_dtype(
    transforms: Optional[TransformerMixin] = None,
) -> Tuple[List[str], List[str]]:
    """Returns numerical and categorical columns of the dataset.

    Args:
        transforms (Optional[TransformerMixin], optional): Transforms to apply before
        extracting the columns of different dtypes. Defaults to None.

    Returns:
        Tuple[List[str], List[str]]: numerical_columns, categorical_columns
    """
    df = pd.read_csv("datasets/test.csv")
    if transforms is not None:
        df = transforms.fit_transform(df)
    numerical_columns = df.select_dtypes(exclude=object).columns.tolist()
    categorical_columns = df.select_dtypes(include=object).columns.tolist()
    return numerical_columns, categorical_columns
