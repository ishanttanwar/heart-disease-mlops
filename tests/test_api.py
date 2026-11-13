"""API tests: a small model is trained on synthetic data and served."""
import joblib
import json

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config, predict as predictor, preprocessing
from src.api.app import app
from src.predict import SAMPLE_PATIENT


@pytest.fixture
def client(model_df, tmp_path, monkeypatch):
    """Train a tiny pipeline, point the API at it, start the app."""
    X, y = preprocessing.split_xy(model_df)
    model = Pipeline([
        ("pre", preprocessing.build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(X, y)
    model_file = tmp_path / "model.joblib"
    meta_file = tmp_path / "metadata.json"
    joblib.dump(model, model_file)
    meta_file.write_text(json.dumps({
        "run_name": "test_model", "trained_at": "2026-01-01T00:00:00",
        "decision_threshold": 0.5, "features": config.FEATURES,
    }))
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(meta_file))
    with TestClient(app) as test_client:  # runs the startup (lifespan) hook
        yield test_client
    predictor.load_model.cache_clear()
    predictor.load_metadata.cache_clear()


def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"].startswith("test_model@")


def test_predict_returns_prediction_and_confidence(client):
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (5, 6)
    assert body["label"] in config.CLASS_LABELS.values()
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["risk_level"] in ("low", "moderate", "high")
    assert response.headers["x-request-id"] == body["request_id"]


def test_confidence_matches_probability(client):
    body = client.post("/predict", json=SAMPLE_PATIENT).json()
    p = body["probability_disease"]
    expected = p if body["prediction"] == 1 else 1 - p
    assert body["confidence"] == pytest.approx(expected, abs=1e-3)


def test_missing_ca_and_thal_are_accepted(client):
    patient = {**SAMPLE_PATIENT, "ca": None, "thal": None}
    assert client.post("/predict", json=patient).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("age", 5),          # below the allowed range
    ("cp", 9),           # not a valid chest pain code
    ("thal", 4),         # not 3, 6 or 7
    ("chol", "high"),    # wrong type
])
def test_invalid_input_rejected(client, field, value):
    response = client.post("/predict", json={**SAMPLE_PATIENT, field: value})
    assert response.status_code == 422


def test_missing_required_field_rejected(client):
    patient = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
    assert client.post("/predict", json=patient).status_code == 422


def test_unknown_field_rejected(client):
    response = client.post("/predict", json={**SAMPLE_PATIENT, "bmi": 25})
    assert response.status_code == 422


def test_batch_prediction(client):
    payload = {"patients": [SAMPLE_PATIENT, {**SAMPLE_PATIENT, "age": 40}]}
    body = client.post("/predict/batch", json=payload).json()
    assert len(body["predictions"]) == 2


def test_metrics_endpoint_counts_predictions(client):
    client.post("/predict", json=SAMPLE_PATIENT)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
    assert "api_request_latency_seconds" in text


def test_health_reports_unavailable_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(tmp_path / "missing.json"))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=SAMPLE_PATIENT).status_code == 503
    predictor.load_model.cache_clear()
    predictor.load_metadata.cache_clear()
