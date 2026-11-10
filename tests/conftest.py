"""Shared pytest fixtures: small synthetic data in the raw UCI format.

The tests never touch the network or the real dataset, so they run anywhere
(including CI) in a couple of seconds.
"""
import numpy as np
import pandas as pd
import pytest

from src import config, data_loader


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """120 synthetic patients in the raw 14-column layout, with NaNs."""
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.integers(30, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(126, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(80, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": np.round(rng.uniform(0, 5, n), 1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "num": rng.choice([0, 1, 2, 3, 4], n, p=[0.55, 0.2, 0.1, 0.1, 0.05]),
    })
    df.loc[[3, 17, 40], "ca"] = np.nan
    df.loc[[8, 60], "thal"] = np.nan
    return df[config.RAW_COLUMNS]


@pytest.fixture
def model_df(raw_df, tmp_path) -> pd.DataFrame:
    """Cleaned data as the modelling code sees it (via the CSV round trip)."""
    path = tmp_path / "clean.csv"
    data_loader.save_clean(data_loader.clean(raw_df), path)
    return data_loader.load_clean(path)
