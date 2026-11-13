"""FastAPI service for heart disease risk prediction.

Endpoints
    GET  /                health-style summary and links
    GET  /health          liveness/readiness (is the model loaded?)
    GET  /model           metadata of the served model
    POST /predict         one patient -> prediction + confidence
    POST /predict/batch   up to 100 patients
    GET  /metrics         Prometheus metrics

Run locally (from the repo root):
    uvicorn src.api.app:app --reload
"""
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from src import predict as predictor
from src.predict import SAMPLE_PATIENT

API_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Structured (JSON) logging: one line per request, easy to grep or ship
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


log = logging.getLogger("heart_api")
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def log_event(message: str, **fields) -> None:
    log.info(message, extra={"extra_fields": fields})


# ---------------------------------------------------------------------------
# Prometheus metrics (module level so they are registered exactly once)
# ---------------------------------------------------------------------------
REQUESTS = Counter(
    "api_requests_total", "HTTP requests", ["method", "endpoint", "status"]
)
LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter(
    "model_predictions_total", "Predictions served", ["label"]
)
PROBABILITY = Histogram(
    "model_prediction_probability", "Predicted probability of disease",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded, else 0")


# ---------------------------------------------------------------------------
# Request / response schemas (validation happens before the model sees data)
# ---------------------------------------------------------------------------
class Patient(BaseModel):
    """One patient's 13 clinical features (UCI Cleveland coding)."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"example": SAMPLE_PATIENT}
    )

    age: float = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type")
    trestbps: float = Field(..., ge=70, le=250,
                            description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=700,
                        description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=60, le=220,
                           description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10,
                           description="ST depression (exercise vs rest)")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak ST")
    # ca and thal are the two features that can be missing in the source
    # data; the model imputes them, so the API accepts null for both.
    ca: Optional[Literal[0, 1, 2, 3]] = Field(
        None, description="Major vessels coloured (0-3), null if unknown")
    thal: Optional[Literal[3, 6, 7]] = Field(
        None, description="3 normal, 6 fixed, 7 reversible; null if unknown")


class Prediction(BaseModel):
    prediction: int
    label: str
    probability_disease: float
    confidence: float
    risk_level: str


class PredictResponse(Prediction):
    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    patients: List[Patient] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: List[Prediction]
    model_version: str
    request_id: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def model_version() -> str:
    meta = predictor.load_metadata()
    if not meta:
        return "unknown"
    return f"{meta.get('run_name', 'model')}@{meta.get('trained_at', '?')}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; keep serving /health if it fails."""
    predictor.load_model.cache_clear()
    predictor.load_metadata.cache_clear()
    app.state.model_loaded = False
    try:
        predictor.load_model()
        app.state.model_loaded = True
        MODEL_LOADED.set(1)
        log_event("model_loaded", path=str(predictor.model_path()),
                  version=model_version())
    except Exception as exc:  # report through /health instead of crashing
        MODEL_LOADED.set(0)
        log_event("model_load_failed", error=str(exc))
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    version=API_VERSION,
    description="Predicts presence of heart disease from 13 clinical "
                "features (UCI Cleveland). Educational project, not a "
                "medical device.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe(request: Request, call_next):
    """Request id, latency, Prometheus counters and one JSON log line."""
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        if endpoint != "/metrics":  # don't let scrapes dominate the numbers
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            log_event("request", request_id=request_id,
                      method=request.method, path=request.url.path,
                      status=status, latency_ms=round(elapsed * 1000, 2))


def _ensure_model():
    try:
        return predictor.load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _record(results) -> None:
    for r in results:
        PREDICTIONS.labels(r["label"]).inc()
        PROBABILITY.observe(r["probability_disease"])


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "version": API_VERSION,
            "docs": "/docs", "health": "/health", "metrics": "/metrics"}


@app.get("/health")
def health(request: Request, response: Response):
    loaded = getattr(request.app.state, "model_loaded", False)
    if not loaded:
        response.status_code = 503
    return {"status": "ok" if loaded else "model_unavailable",
            "model_loaded": loaded, "model_version": model_version()}


@app.get("/model")
def model_info():
    meta = predictor.load_metadata()
    if not meta:
        raise HTTPException(status_code=404, detail="No model metadata")
    return meta


@app.post("/predict", response_model=PredictResponse)
def predict_one(patient: Patient, request: Request):
    model = _ensure_model()
    result = predictor.predict([patient.model_dump()], model=model)[0]
    _record([result])
    log_event("prediction", request_id=request.state.request_id,
              prediction=result["prediction"],
              probability=result["probability_disease"])
    return {**result, "model_version": model_version(),
            "request_id": request.state.request_id}


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    model = _ensure_model()
    results = predictor.predict(
        [p.model_dump() for p in batch.patients], model=model
    )
    _record(results)
    log_event("batch_prediction", request_id=request.state.request_id,
              size=len(results))
    return {"predictions": results, "model_version": model_version(),
            "request_id": request.state.request_id}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
