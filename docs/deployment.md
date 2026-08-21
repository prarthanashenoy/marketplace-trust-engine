# Deployment

## Local Development

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Make

### Setup

```bash
git clone <repo-url>
cd marketplace-exp
make install
```

This runs `poetry install`, creating a virtual environment and installing all dependencies.

### Full Pipeline

```bash
# 1. Generate synthetic data  →  data/raw/*.parquet
make generate-data

# 2. Train all models         →  models/*.pkl
make train

# 3. Evaluate                 →  data/processed/eval_*.json
make evaluate

# 4. Run API (port 8000)
make serve

# 5. Run dashboard (port 8501)
make dashboard
```

### Run Tests

```bash
make test          # 126 unit tests with coverage
make lint          # ruff check
make typecheck     # mypy
```

---

## Docker

### Single Container (API only)

Build and run the API container:

```bash
docker build -t marketplace-trust-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models marketplace-trust-api
```

The container expects trained model artefacts to be mounted at `/app/models`. Run `make train` first to produce them.

### Docker Compose (Full Stack)

```bash
docker-compose up
```

This starts four services:

| Service | URL | Purpose |
|---------|-----|---------|
| `api` | http://localhost:8000 | FastAPI inference service |
| `dashboard` | http://localhost:8501 | Streamlit dashboard |
| `mlflow` | http://localhost:5000 | Experiment tracking UI |
| `postgres` | localhost:5432 | MLflow backend database |

#### First-time setup with Docker Compose

```bash
# Start infrastructure (postgres + mlflow)
docker-compose up postgres mlflow -d

# Generate data and train (runs inside the api container)
docker-compose run api poetry run python scripts/generate_data.py
docker-compose run api poetry run python scripts/train.py

# Start all services
docker-compose up
```

#### Volumes

| Volume | Container path | Purpose |
|--------|---------------|---------|
| `./data` | `/app/data` | Raw and processed data |
| `./models` | `/app/models` | Trained model artefacts |

#### Environment Variables (docker-compose.yml)

```yaml
environment:
  - MODEL_DIR=/app/models
  - MLFLOW_TRACKING_URI=http://mlflow:5000
```

---

## Dockerfile Details

The Dockerfile uses a multi-stage build:

```dockerfile
# Stage 1: dependency installation
FROM python:3.11-slim AS deps
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev

# Stage 2: production image
FROM python:3.11-slim
COPY --from=deps /root/.cache /root/.cache
COPY src/ ./src/
CMD ["uvicorn", "marketplace_trust.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

This keeps the production image smaller by separating dependency installation from the application code copy.

### Health Check

The Dockerfile declares a health check that polls `GET /health`:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD curl -f http://localhost:8000/health || exit 1
```

Docker Compose and orchestrators (ECS, Kubernetes) use this to determine when the container is ready.

---

## MLflow Experiment Tracking

MLflow logs the following to each training run:

| Category | Items logged |
|----------|-------------|
| Parameters | All config values (n_estimators, learning_rate, etc.) |
| Metrics | PR-AUC, ROC-AUC on val and test sets |
| Artefacts | `ensemble.pkl`, `classifier.pkl`, `feature_meta.json` |

Access the MLflow UI after starting:

```bash
# Local
mlflow ui --port 5000

# Via Docker Compose
docker-compose up mlflow
# → http://localhost:5000
```

To disable MLflow logging (e.g., in CI):

```bash
poetry run python scripts/train.py --no-mlflow
```

---

## CI/CD (GitHub Actions)

The workflow at `.github/workflows/ci.yml` runs on every push and pull request:

```
1. Lint         →  ruff check src/ tests/
2. Format       →  ruff format --check src/ tests/
3. Type check   →  mypy src/
4. Tests        →  pytest tests/ --cov=src --cov-report=xml
```

All four steps must pass for a PR to be mergeable.

---

## Makefile Targets

| Target | Command | Description |
|--------|---------|-------------|
| `install` | `poetry install` | Install all dependencies |
| `generate-data` | `python scripts/generate_data.py` | Generate synthetic data |
| `train` | `python scripts/train.py` | Train all models |
| `evaluate` | `python scripts/evaluate.py` | Run evaluation pipeline |
| `serve` | `uvicorn ... --reload` | Start FastAPI dev server |
| `dashboard` | `streamlit run dashboard/app.py` | Start Streamlit |
| `test` | `pytest tests/ --cov` | Run test suite |
| `lint` | `ruff check src/ tests/` | Check code style |
| `typecheck` | `mypy src/` | Run type checker |
| `clean` | `rm -rf data/ models/ mlruns/` | Remove generated artefacts |

---

## Production Considerations

### Feature Store

The current implementation computes features from raw tables on demand. In production with millions of sellers, you would:

1. Pre-compute features periodically (e.g., hourly batch job via Airflow or dbt).
2. Store features in a feature store (e.g., Feast, Tecton, or a materialized view in your data warehouse).
3. The `/predict` endpoint reads pre-computed features from the store rather than accepting raw feature values.

### Model Retraining

A production retraining cadence would:

1. Accumulate new labelled data from analyst reviews and confirmed fraud cases.
2. Retrain on a rolling window (e.g., last 6 months of data).
3. Compare new model metrics against champion model via MLflow.
4. Deploy new model as a canary (small traffic fraction) before full rollout.

### Latency Targets

Current median inference latency is < 5 ms per seller (excluding feature computation). For high-throughput scenarios:

- Run multiple API replicas behind a load balancer.
- Use async uvicorn workers (`--workers N`).
- Cache repeated predictions for the same seller ID (Redis, short TTL).

### Horizontal Scaling

The API is stateless — the model is loaded into process memory at startup. Horizontal scaling is straightforward: run multiple container replicas behind a load balancer. Each replica loads its own model copy.
