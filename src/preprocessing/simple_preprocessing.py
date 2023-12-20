from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from src.utils.preprocessing_utils import process_data
from src.preprocessing.transforms import (
    NanEncoder,
    NumericalToCategorical,
    IntervalEncoder,
)

from src.utils.preprocessing_utils import get_columns_by_dtype


def make_preprocessing_pipeline():
    common_pipeline = make_pipeline(
        NanEncoder(), IntervalEncoder(), NumericalToCategorical()
    )
    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    numerical_columns, categorical_columns = get_columns_by_dtype(common_pipeline)

    pipeline = make_pipeline(
        common_pipeline,
        ColumnTransformer(
            [
                ("num", numerical_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ]
        ),
    )

    target_transform = StandardScaler()

    return pipeline, target_transform


if __name__ == "__main__":
    pipeline, target_transform = make_preprocessing_pipeline()
    process_data(
        method_name="simple", pipeline=pipeline, target_transform=target_transform
    )
