"""Train, tune and compare classifiers, with every run tracked in MLflow.

For each model family and each preprocessing variant (scaler x engineered
features) this script:

  1. tunes hyperparameters with GridSearchCV over stratified k-fold CV,
  2. logs params, CV metrics (mean and std), hold-out test metrics,
     plots and the fitted pipeline to MLflow as one run,
  3. collects the results into a comparison table.

The winner is chosen on mean CV ROC-AUC (never on the test set), saved to
models/ as a single joblib pipeline (preprocessing + model) with a metadata
file, and registered in the MLflow Model Registry.

Usage (from the repo root):
    python -m src.train            # full experiment grid
    python -m src.train --quick    # small grid, used in CI and tests
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import json
import logging
import platform
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed; also required in CI and Docker

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config, preprocessing  # noqa: E402
from src.data_loader import load_clean  # noqa: E402

log = logging.getLogger("train")

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
SCORERS = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


# ---------------------------------------------------------------------------
# Model families and their search spaces
# ---------------------------------------------------------------------------
def model_specs(quick: bool = False) -> dict:
    """Estimator factory + hyperparameter grid for each model family.

    quick=True keeps one value per hyperparameter so CI runs in seconds.
    """
    rs = config.RANDOM_STATE
    specs = {
        "logistic_regression": (
            lambda: LogisticRegression(
                solver="saga", max_iter=10000, random_state=rs
            ),
            {
                # l1_ratio 0 = ridge (L2), 1 = lasso (L1), 0.5 = elastic net
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__l1_ratio": [0.0, 0.5, 1.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            lambda: RandomForestClassifier(
                n_estimators=300, random_state=rs, n_jobs=1
            ),
            {
                "clf__max_depth": [None, 4, 8],
                "clf__min_samples_leaf": [1, 3, 5],
                "clf__max_features": ["sqrt", 0.5],
            },
        ),
        "gradient_boosting": (
            lambda: GradientBoostingClassifier(random_state=rs),
            {
                "clf__n_estimators": [100, 300],
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3],
                "clf__subsample": [0.8, 1.0],
            },
        ),
    }
    if quick:
        specs = {
            name: (factory, {k: v[:1] for k, v in grid.items()})
            for name, (factory, grid) in specs.items()
        }
    return specs


def preprocessing_variants(quick: bool = False) -> list:
    """(scaler, add_engineered) combinations to compare as experiments."""
    if quick:
        return [("standard", False)]
    return [
        (scaler, engineered)
        for scaler in ("standard", "robust")
        for engineered in (False, True)
    ]


def build_pipeline(estimator, scaler: str, add_engineered: bool) -> Pipeline:
    """Preprocessing + classifier as one object (what gets saved/served)."""
    return Pipeline([
        ("pre", preprocessing.build_preprocessor(
            scaler=scaler, add_engineered=add_engineered
        )),
        ("clf", estimator),
    ])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(y_true, proba, threshold: float = config.DECISION_THRESHOLD):
    """Test-set metrics from predicted probabilities."""
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def cv_summary(search: GridSearchCV) -> dict:
    """Mean and std of every CV metric for the best hyperparameters."""
    i = search.best_index_
    res = search.cv_results_
    out = {}
    for m in METRICS:
        out[f"cv_{m}_mean"] = float(res[f"mean_test_{m}"][i])
        out[f"cv_{m}_std"] = float(res[f"std_test_{m}"][i])
    return out


def feature_importance(model: Pipeline) -> pd.Series:
    """Importance per transformed feature (|coef| for linear models)."""
    names = preprocessing.get_feature_names(model.named_steps["pre"])
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    else:
        values = np.abs(clf.coef_[0])
    return pd.Series(values, index=names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Plots (each returns the saved path)
# ---------------------------------------------------------------------------
def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc(y_true, proba, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax, name="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_confusion(y_true, pred, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, ax=ax, colorbar=False, cmap="Blues",
        display_labels=[config.CLASS_LABELS[0], config.CLASS_LABELS[1]],
    )
    ax.set_title(title)
    return _save(fig, path)


def plot_importance(importance: pd.Series, title: str, path: Path,
                    top: int = 15) -> Path:
    top_items = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(top_items) + 1.2))
    ax.barh(top_items.index, top_items.values, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("importance")
    return _save(fig, path)


def plot_comparison(table: pd.DataFrame, path: Path) -> Path:
    """Mean CV ROC-AUC (with std error bars) for every run."""
    t = table.sort_values("cv_roc_auc_mean")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(t) + 1.5))
    ax.barh(t["run_name"], t["cv_roc_auc_mean"], xerr=t["cv_roc_auc_std"],
            color="#4C72B0", capsize=3)
    lo = max(0.0, (t["cv_roc_auc_mean"] - t["cv_roc_auc_std"]).min() - 0.05)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("mean CV ROC-AUC (error bar = 1 std)")
    ax.set_title("Model comparison")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
def default_tracking_uri() -> str:
    return f"sqlite:///{config.MLFLOW_DB_PATH.as_posix()}"


def setup_mlflow(tracking_uri: str, artifacts_dir: Path,
                 experiment: str) -> None:
    """Point MLflow at the local SQLite store and our artefact folder."""
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(
            experiment, artifact_location=artifacts_dir.resolve().as_uri()
        )
    mlflow.set_experiment(experiment)


# ---------------------------------------------------------------------------
# One tracked run
# ---------------------------------------------------------------------------
def train_one(model_name, factory, grid, scaler, engineered,
              X_train, X_test, y_train, y_test, cv):
    """Tune one model/variant and log everything to a single MLflow run."""
    variant = f"{scaler}{'_eng' if engineered else ''}"
    run_name = f"{model_name}__{variant}"
    pipe = build_pipeline(factory(), scaler, engineered)
    search = GridSearchCV(
        pipe, grid, scoring=SCORERS, refit=config.PRIMARY_METRIC,
        cv=cv, n_jobs=-1, error_score="raise",
    )

    with mlflow.start_run(run_name=run_name) as run:
        start = time.perf_counter()
        search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        best = search.best_estimator_
        proba = best.predict_proba(X_test)[:, 1]
        pred = (proba >= config.DECISION_THRESHOLD).astype(int)
        cv_metrics = cv_summary(search)
        test_metrics = evaluate(y_test, proba)

        mlflow.set_tags({
            "model_family": model_name,
            "preprocessing_variant": variant,
            "selection_metric": config.PRIMARY_METRIC,
        })
        mlflow.log_params({
            "model": model_name,
            "scaler": scaler,
            "engineered_features": engineered,
            "cv_folds": cv.get_n_splits(),
            "grid_size": len(search.cv_results_["params"]),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "decision_threshold": config.DECISION_THRESHOLD,
            **{k.replace("clf__", "best_"): v
               for k, v in search.best_params_.items()},
        })
        mlflow.log_metrics({
            **cv_metrics,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "fit_seconds": fit_seconds,
        })

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_roc(y_test, proba, f"ROC - {run_name}", tmp / "roc_curve.png")
            plot_confusion(y_test, pred, f"Confusion - {run_name}",
                           tmp / "confusion_matrix.png")
            plot_importance(feature_importance(best),
                            f"Top features - {run_name}",
                            tmp / "feature_importance.png")
            pd.DataFrame(search.cv_results_).drop(
                columns="params"
            ).to_csv(tmp / "cv_results.csv", index=False)
            (tmp / "classification_report.txt").write_text(
                classification_report(
                    y_test, pred, zero_division=0,
                    target_names=list(config.CLASS_LABELS.values()),
                )
            )
            mlflow.log_artifacts(str(tmp), artifact_path="evaluation")

        model_info = mlflow.sklearn.log_model(
            best,
            name="model",
            signature=infer_signature(X_train, best.predict_proba(X_train)),
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        log.info(
            "%-34s cv_roc_auc=%.3f±%.3f  test_roc_auc=%.3f  (%.1fs)",
            run_name, cv_metrics["cv_roc_auc_mean"],
            cv_metrics["cv_roc_auc_std"], test_metrics["roc_auc"],
            fit_seconds,
        )
        return {
            "run_name": run_name,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "model": model_name,
            "scaler": scaler,
            "engineered_features": engineered,
            "best_params": search.best_params_,
            **cv_metrics,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "fit_seconds": round(fit_seconds, 2),
        }, best


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def select_best(table: pd.DataFrame) -> pd.Series:
    """Highest mean CV ROC-AUC; ties broken by CV recall, then lower std."""
    ranked = table.sort_values(
        ["cv_roc_auc_mean", "cv_recall_mean", "cv_roc_auc_std"],
        ascending=[False, False, True],
    )
    return ranked.iloc[0]


def save_final_model(model: Pipeline, row: pd.Series, X_train) -> dict:
    """Persist the winning pipeline and a metadata file for the API."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    metadata = {
        "model": row["model"],
        "run_name": row["run_name"],
        "mlflow_run_id": row["run_id"],
        "scaler": row["scaler"],
        "engineered_features": bool(row["engineered_features"]),
        "best_params": {k.replace("clf__", ""): v
                        for k, v in row["best_params"].items()},
        "decision_threshold": config.DECISION_THRESHOLD,
        "features": list(X_train.columns),
        "cv_metrics": {m: round(float(row[f"cv_{m}_mean"]), 4)
                       for m in METRICS},
        "test_metrics": {m: round(float(row[f"test_{m}"]), 4)
                         for m in METRICS},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    config.MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    return metadata


def run_training(df: pd.DataFrame = None, quick: bool = False,
                 tracking_uri: str = None, register: bool = True) -> dict:
    """Full experiment: every model x variant, then select and save."""
    df = load_clean() if df is None else df
    X_train, X_test, y_train, y_test = (
        preprocessing.train_test_split_stratified(df)
    )
    cv = StratifiedKFold(
        n_splits=3 if quick else config.CV_FOLDS,
        shuffle=True, random_state=config.RANDOM_STATE,
    )
    setup_mlflow(tracking_uri or default_tracking_uri(),
                 config.MLFLOW_ARTIFACTS_DIR, config.MLFLOW_EXPERIMENT)
    log.info("Train %d / test %d rows | %d-fold CV | quick=%s",
             len(X_train), len(X_test), cv.get_n_splits(), quick)

    rows, models = [], {}
    for model_name, (factory, grid) in model_specs(quick).items():
        for scaler, engineered in preprocessing_variants(quick):
            row, fitted = train_one(model_name, factory, grid, scaler,
                                    engineered, X_train, X_test, y_train,
                                    y_test, cv)
            rows.append(row)
            models[row["run_name"]] = fitted

    table = pd.DataFrame(rows)
    best = select_best(table)
    metadata = save_final_model(models[best["run_name"]], best, X_train)

    # Comparison table + chart, kept in the repo for the report
    config.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["run_name"] + [f"cv_{m}_mean" for m in METRICS] + [
        "cv_roc_auc_std"] + [f"test_{m}" for m in METRICS] + ["fit_seconds"]
    summary = table[cols].sort_values("cv_roc_auc_mean", ascending=False)
    summary.round(4).to_csv(config.TRAINING_DIR / "model_comparison.csv",
                            index=False)
    plot_comparison(table, config.TRAINING_DIR / "model_comparison.png")

    if register:
        try:
            mlflow.register_model(best["model_uri"],
                                  config.REGISTERED_MODEL_NAME)
        except Exception as exc:  # registry is optional; never fail training
            log.warning("Model registration skipped: %s", exc)

    log.info("Best: %s | cv_roc_auc=%.3f | test_roc_auc=%.3f | saved to %s",
             best["run_name"], best["cv_roc_auc_mean"], best["test_roc_auc"],
             config.MODEL_PATH)
    return {"table": summary, "best": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="small grid and 3-fold CV (CI / smoke test)")
    parser.add_argument("--no-register", action="store_true",
                        help="skip MLflow Model Registry registration")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("mlflow", "alembic", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # We deliberately use cloudpickle (the custom ClinicalFeatures transformer
    # is not on skops' trusted list), so silence the repeated format notice.
    logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

    result = run_training(quick=args.quick, register=not args.no_register)
    print("\n" + result["table"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
