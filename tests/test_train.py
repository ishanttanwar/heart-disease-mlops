"""Tests for the training pipeline (fast: quick grid, temp MLflow store)."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src import config, train


@pytest.fixture
def signal_df(model_df) -> pd.DataFrame:
    """Synthetic data where the label depends on oldpeak and thalach.

    The shared fixture has random labels; here we plant a learnable
    pattern so we can check that training actually learns something.
    """
    df = model_df.copy()
    risk = df["oldpeak"] / 5 + (200 - df["thalach"]) / 120
    df[config.TARGET] = (risk > risk.median()).astype(int)
    return df


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Send every file the training run writes into a temp folder."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "m.joblib")
    monkeypatch.setattr(config, "MODEL_METADATA_PATH",
                        tmp_path / "models" / "metadata.json")
    monkeypatch.setattr(config, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACTS_DIR", tmp_path / "mlruns")
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest",
                                  "gradient_boosting"])
def test_each_model_fits_and_returns_probabilities(signal_df, name):
    factory, _ = train.model_specs(quick=True)[name]
    X, y = signal_df[config.FEATURES], signal_df[config.TARGET]
    model = train.build_pipeline(factory(), "standard", True).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_quick_mode_shrinks_every_grid_to_one_candidate():
    for _, grid in train.model_specs(quick=True).values():
        assert all(len(values) == 1 for values in grid.values())
    assert train.preprocessing_variants(quick=True) == [("standard", False)]


def test_full_mode_compares_all_four_preprocessing_variants():
    assert len(train.preprocessing_variants()) == 4


def test_evaluate_metrics_in_valid_range():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = train.evaluate(y, proba)
    assert set(metrics) == set(train.METRICS)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_select_best_uses_cv_score_not_test_score():
    table = pd.DataFrame({
        "run_name": ["a", "b"],
        "cv_roc_auc_mean": [0.90, 0.85],
        "cv_recall_mean": [0.8, 0.9],
        "cv_roc_auc_std": [0.02, 0.01],
        "test_roc_auc": [0.80, 0.99],
    })
    assert train.select_best(table)["run_name"] == "a"


def test_cv_summary_reports_mean_and_std_for_every_metric(signal_df):
    from sklearn.model_selection import GridSearchCV
    factory, grid = train.model_specs(quick=True)["logistic_regression"]
    search = GridSearchCV(
        train.build_pipeline(factory(), "standard", False), grid,
        scoring=train.SCORERS, refit="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=0),
    ).fit(signal_df[config.FEATURES], signal_df[config.TARGET])
    summary = train.cv_summary(search)
    for m in train.METRICS:
        assert f"cv_{m}_mean" in summary and f"cv_{m}_std" in summary


def test_end_to_end_quick_training(signal_df, isolated_paths):
    result = train.run_training(df=signal_df, quick=True,
                                tracking_uri=isolated_paths, register=False)

    # one comparison row per model family
    assert len(result["table"]) == 3
    assert (config.TRAINING_DIR / "model_comparison.csv").exists()
    assert (config.TRAINING_DIR / "model_comparison.png").exists()

    # saved pipeline reloads and clearly beats chance on the planted signal
    model = joblib.load(config.MODEL_PATH)
    X, y = signal_df[config.FEATURES], signal_df[config.TARGET]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, model.predict_proba(X)[:, 1]) > 0.8

    meta = json.loads(config.MODEL_METADATA_PATH.read_text())
    assert meta["features"] == config.FEATURES
    assert meta["model"] in train.model_specs()
    assert 0.5 < meta["cv_metrics"]["roc_auc"] <= 1.0
