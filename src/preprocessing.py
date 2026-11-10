"""Feature preprocessing for the heart disease models.

Everything that touches raw feature values lives inside one scikit-learn
pipeline, so training, evaluation and the serving API all apply exactly the
same transformations:

    Pipeline
      1. engineer   optional ClinicalFeatures (two derived features)
      2. transform  ColumnTransformer
           numeric      median impute -> scale (standard or robust)
           categorical  most-frequent impute -> one-hot encode

Because imputation and scaling are fitted inside the pipeline, cross-validation
fits them on each training fold only, so nothing leaks from the validation fold.

Note for packaging: a saved model that uses ClinicalFeatures references this
module by name, so `src/` must be present wherever the model is loaded
(including the Docker image).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src import config

SCALERS = {"standard": StandardScaler, "robust": RobustScaler}


# ---------------------------------------------------------------------------
# Optional feature engineering
# ---------------------------------------------------------------------------
class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Add two derived features to a DataFrame.

    hr_reserve_pct
        Max heart rate achieved as a fraction of the age-predicted maximum
        (220 - age). A low value means the heart could not reach the rate
        expected for the patient's age.
    oldpeak_log
        log(1 + oldpeak). ST depression is strongly right-skewed with many
        zeros; the log transform compresses the long tail.

    The transformer is stateless: fit() only records the input column names.
    """

    NEW_FEATURES = ["hr_reserve_pct", "oldpeak_log"]
    REQUIRED = ["age", "thalach", "oldpeak"]

    def _check(self, X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatures expects a pandas DataFrame")
        missing = [c for c in self.REQUIRED if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def fit(self, X, y=None):
        self._check(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        self._check(X)
        out = X.copy()
        out["hr_reserve_pct"] = out["thalach"] / (220.0 - out["age"])
        out["oldpeak_log"] = np.log1p(out["oldpeak"])
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(
            list(self.feature_names_in_) + self.NEW_FEATURES, dtype=object
        )


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_preprocessor(
    scaler: str = "standard", add_engineered: bool = False
) -> Pipeline:
    """Build the (unfitted) preprocessing pipeline.

    Parameters
    ----------
    scaler : "standard" or "robust"
        "robust" scales with the median and IQR, which is less affected by
        the outliers seen in cholesterol and blood pressure.
    add_engineered : bool
        Add the ClinicalFeatures-derived columns before scaling.
    """
    if scaler not in SCALERS:
        raise ValueError(
            f"Unknown scaler '{scaler}'. Choose from {sorted(SCALERS)}"
        )

    numeric = list(config.NUMERIC_FEATURES)
    if add_engineered:
        numeric += ClinicalFeatures.NEW_FEATURES

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", SCALERS[scaler]()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    columns = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, list(config.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    engineer = ClinicalFeatures() if add_engineered else "passthrough"
    return Pipeline([("engineer", engineer), ("transform", columns)])


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def split_xy(df: pd.DataFrame):
    """Separate the feature matrix from the binary target."""
    return df[config.FEATURES].copy(), df[config.TARGET].astype(int)


def train_test_split_stratified(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
):
    """Stratified hold-out split: returns X_train, X_test, y_train, y_test."""
    X, y = split_xy(df)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def get_feature_names(preprocessor: Pipeline) -> list:
    """Names of the columns the fitted preprocessor outputs."""
    return list(preprocessor.get_feature_names_out())
