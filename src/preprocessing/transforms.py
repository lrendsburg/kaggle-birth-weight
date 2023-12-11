from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class NanEncoder(BaseEstimator, TransformerMixin):
    """Converts preselected set of values to NaN.

    The dataset has missing values, but encodes them as specific values. This transformer
    converts those values to np.nan to facilitate further processing.
    """

    COLUMN_TO_NVAL = {
        "BMI": 99.9,
        "WTGAIN": 99,
        "FAGECOMB": 99,
        "PREVIS": 99,
        "CIG_0": 99,
        "M_Ht_In": 99,
        "PRIORDEAD": 99,
        "PRIORLIVE": 99,
        "PRIORTERM": 99,
        "PRECARE": 99,
        "RF_CESARN": 99,
        "PWgt_R": 999,
        "MEDUC": 9,
        "FEDUC": 9,
        "PAY": 9,
        "BFACIL": 9,
        "ATTEND": 9,
        "RDMETH_REC": 9,
        "PAY_REC": 9,
        "NO_INFEC": 9,
        "MBSTATE_REC": 3,
        "NO_RISKS": 9,
        "NO_MMORB": 9,
        "DLMP_MM": 99,
        "DOB_TT": 9999,
        "DMAR": " ",
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for column, value in NanEncoder.COLUMN_TO_NVAL.items():
            X[column] = X[column].replace(value, np.nan)
        return X


class NumericalToCategorical(BaseEstimator, TransformerMixin):
    """Converts a preselected set of numerical columns to categorical.

    Some columns are encoded as numerical but are actually categorical. This transformer
    converts those columns to categorical to facilitate further processing like one-hot encoding.
    """

    COLUMNS = [
        "MEDUC",
        "FEDUC",
        "PAY",
        "BFACIL",
        "ATTEND",
        "RDMETH_REC",
        "PAY_REC",
        "RESTATUS",
        "NO_INFEC",
        "MBSTATE_REC",
        "NO_RISKS",
        "NO_MMORB",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for column in NumericalToCategorical.COLUMNS:
            X[column] = X[column].astype("object")
        return X


class IntervalEncoder(BaseEstimator, TransformerMixin):
    """Seperates columns that encode multiple categories while retaining numerical values.

    The columns 'ILLB_R', 'ILOP_R', 'ILP_R' encode different categories with different values,
    specifically the categories 3, 4-300, 888, 999. This transformer splits those columns
    into separate one-hot encoded categories while retaining the value for the numerical
    category 3-400. Also handles imputation of the missing value category 999."""

    COLUMNS = ["ILLB_R", "ILOP_R", "ILP_R"]

    def __init__(self):
        self.impute_vals = {}

    def fit(self, X, y=None):
        for column in IntervalEncoder.COLUMNS:
            # Determine most frequent category between (3, 4-300, 888)
            temp_col = X[column].copy()

            idxs_4_300 = (temp_col >= 4) & (temp_col <= 300)
            temp_col[idxs_4_300] = "4-300"

            impute_val = temp_col[temp_col != 999].mode()[0]
            # If numeric category is most frequent, use its median for imputation
            if impute_val == "4-300":
                impute_val = X[column][idxs_4_300].median()

            self.impute_vals[column] = impute_val

        return self

    def transform(self, X, y=None):
        X = X.copy()
        for column in IntervalEncoder.COLUMNS:
            # Impute
            X[column] = X[column].replace(999, self.impute_vals[column])

            # One-hot encode with numerical column
            X[f"{column}_003"] = (X[column] == 3).astype(int)
            X[f"{column}_num"] = X[column].where(
                (X[column] >= 4) & (X[column] <= 300), 0
            )
            X[f"{column}_888"] = (X[column] == 888).astype(int)

            X.drop(column, axis=1, inplace=True)
        return X


class CapEncoder(BaseEstimator, TransformerMixin):
    """Introduce binary flags for capped values."""

    COLUMN_TO_CAP = {
        "WTGAIN": 98,
        "FAGECOMB": 98,
        "PREVIS": 98,
        "MAGER": 50,
        "CIG_0": 98,
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for column, cap in CapEncoder.COLUMN_TO_CAP.items():
            X[f"{column}_capped"] = (X[column] == cap).astype(int)
        return X
