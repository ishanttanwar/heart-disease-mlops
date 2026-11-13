"""Load the saved pipeline and turn raw patient records into predictions.

Used by the API and usable on its own from the command line:

    python -m src.predict                       # predicts a built-in sample
    python -m src.predict --json patient.json   # one record or a list
"""
import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config

# Probability bands for a human-readable risk level (tunable, documented
# in the report). They are independent of the decision threshold.
RISK_BANDS = [(0.30, "low"), (0.60, "moderate"), (1.01, "high")]

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def model_path() -> Path:
    """MODEL_PATH env var wins (used in Docker), else the repo default."""
    return Path(os.getenv("MODEL_PATH", config.MODEL_PATH))


def metadata_path() -> Path:
    return Path(os.getenv("MODEL_METADATA_PATH", config.MODEL_METADATA_PATH))


@lru_cache(maxsize=1)
def load_model():
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    path = metadata_path()
    return json.loads(path.read_text()) if path.exists() else {}


def to_frame(records) -> pd.DataFrame:
    """List of dicts -> DataFrame in training column order (None -> NaN)."""
    df = pd.DataFrame(records)
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return df[config.FEATURES].astype(float)


def risk_level(probability: float) -> str:
    return next(label for limit, label in RISK_BANDS if probability < limit)


def predict(records, model=None, threshold: float = None) -> list:
    """Predict for one or more patients.

    Returns one dict per patient with the class, its label, the probability
    of disease, the confidence in the returned class and a risk level.
    """
    model = model if model is not None else load_model()
    if threshold is None:
        threshold = load_metadata().get(
            "decision_threshold", config.DECISION_THRESHOLD
        )
    proba = model.predict_proba(to_frame(records))[:, 1]
    pred = (proba >= threshold).astype(int)
    return [
        {
            "prediction": int(c),
            "label": config.CLASS_LABELS[int(c)],
            "probability_disease": round(float(p), 4),
            "confidence": round(float(np.where(c == 1, p, 1 - p)), 4),
            "risk_level": risk_level(float(p)),
        }
        for c, p in zip(pred, proba)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict heart disease risk")
    parser.add_argument("--json", type=Path,
                        help="file with one patient record or a list")
    args = parser.parse_args()
    records = (json.loads(args.json.read_text()) if args.json
               else SAMPLE_PATIENT)
    records = records if isinstance(records, list) else [records]
    print(json.dumps(predict(records), indent=2))


if __name__ == "__main__":
    main()
