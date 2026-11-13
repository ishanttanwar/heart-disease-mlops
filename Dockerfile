# Heart Disease Risk API - production image
# Build:  docker build -t heart-disease-api:1.0 .
# Run:    docker run -d -p 8000:8000 --name heart-api heart-disease-api:1.0

FROM python:3.12-slim

# No .pyc files, unbuffered logs (so JSON request logs appear immediately)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_METADATA_PATH=/app/models/metadata.json

WORKDIR /app

# Dependencies first: this layer is cached until requirements change,
# so code-only edits rebuild in seconds.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Application code and the trained pipeline. src/ is required because the
# saved pipeline references src.preprocessing (ClinicalFeatures).
COPY src/ ./src/
COPY models/model.joblib models/metadata.json ./models/

# Run as an unprivileged user rather than root
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Container reports unhealthy if /health stops answering (curl is not in
# the slim image, so use Python's standard library)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# One worker per container: Prometheus counters live in process memory, so
# multiple workers would split the metrics. Scale out with more replicas
# (Kubernetes) instead of more workers.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
