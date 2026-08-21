# System Architecture

## Overview

The Marketplace Trust Engine is structured as a full ML lifecycle system — from synthetic data generation through model training, evaluation, a REST inference API, and an interactive dashboard.

```
Raw Marketplace Data (synthetic)
           │
           ▼
┌──────────────────────┐
│   Feature Pipeline   │  50+ features across 5 groups
│   (4 modules)        │
└──────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────┐
│            Ensemble Risk Scorer                 │
│  60% XGBoost + 25% IsolationForest + 15% Rules │
└────────────────────────────────────────────────┘
           │
           ▼
    Risk Score [0–100]
           │
     ┌─────┴─────┐
     ▼           ▼
 FastAPI      Streamlit
 /predict     Dashboard
```

## Repository Layout

```
marketplace-exp/
├── src/marketplace_trust/
│   ├── api/              # FastAPI application and Pydantic schemas
│   ├── data/             # Synthetic data generator and validation
│   ├── features/         # Feature pipeline and 5 feature modules
│   ├── models/           # Baseline, XGBoost, IsolationForest, Ensemble
│   ├── evaluation/       # Metrics, SHAP, reporting
│   ├── inference/        # Predictor for real-time scoring
│   └── config.py         # Pydantic configuration hierarchy
├── dashboard/            # Streamlit 6-page dashboard
├── scripts/              # CLI entry points (generate, train, evaluate, predict)
├── configs/config.yaml   # All hyperparameters
├── tests/                # 126 pytest tests (5 modules)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/    # GitHub Actions CI (lint → type-check → test)
```

## Components

### Data Layer (`src/marketplace_trust/data/`)

Generates fully synthetic marketplace data across seven relational tables (users, sellers, listings, transactions, reviews, devices, events). Fraud is injected at seller-creation time across nine scenario types so no target leakage is possible when features are computed.

See [data-pipeline.md](data-pipeline.md) for full details.

### Feature Pipeline (`src/marketplace_trust/features/`)

Transforms raw tables into a flat seller-level feature matrix of 50+ features grouped by:

- Seller profile
- Price signals
- Transaction signals
- Behavioural signals
- Graph / network signals

See [feature-engineering.md](feature-engineering.md) for full details.

### Model Layer (`src/marketplace_trust/models/`)

| Module | Class | Purpose |
|--------|-------|---------|
| `baseline.py` | `BaselineClassifier` | Logistic regression reference point |
| `classifier.py` | `FraudClassifier` | XGBoost supervised model |
| `anomaly.py` | `AnomalyDetector` | Isolation Forest (unsupervised) |
| `ensemble.py` | `EnsembleRiskScorer` | Weighted combination → risk score |
| `splitter.py` | `TemporalSplitter` | Temporal 70/15/15 train/val/test split |

See [ml-algorithms.md](ml-algorithms.md) and [ensemble-strategy.md](ensemble-strategy.md).

### Evaluation Layer (`src/marketplace_trust/evaluation/`)

| Module | Purpose |
|--------|---------|
| `metrics.py` | PR-AUC, ROC-AUC, threshold analysis, Precision@K |
| `shap_explainer.py` | SHAP values via TreeExplainer |
| `reporter.py` | Save JSON reports and parquet artefacts |

See [evaluation.md](evaluation.md) and [explainability.md](explainability.md).

### Inference Layer (`src/marketplace_trust/inference/`)

`Predictor` wraps the serialised ensemble for real-time scoring. It handles feature-schema alignment (missing → 0, extras dropped) so the API is tolerant of schema evolution.

### API (`src/marketplace_trust/api/`)

FastAPI application with three endpoints (`GET /health`, `GET /model-info`, `POST /predict`). The model is loaded once at startup via the lifespan context manager.

See [api-reference.md](api-reference.md).

### Dashboard (`dashboard/`)

Six-page Streamlit application:

1. **Overview** — headline KPIs
2. **Risk Distribution** — score histogram, risk-level breakdown
3. **Seller Investigation** — search by ID, SHAP explanation
4. **Fraud Scenarios** — per-scenario feature distributions
5. **Model Performance** — PR/ROC curves, SHAP importance
6. **Network Investigation** — shared-device/IP graph

## Training vs. Inference Flow

### Training

```
scripts/generate_data.py  →  data/raw/*.parquet
scripts/train.py          →  models/ensemble.pkl
                             models/classifier.pkl
                             data/processed/feature_meta.json
                             MLflow run
scripts/evaluate.py       →  data/processed/eval_*.json
                             data/processed/shap_importance.csv
                             data/processed/shap_values_test.parquet
```

### Inference

```
POST /predict  →  Predictor.predict()
               →  feature alignment to training schema
               →  EnsembleRiskScorer.predict_single()
               →  RiskPrediction (score, level, components, top factors)
```

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | numpy, pandas |
| ML | scikit-learn, xgboost, shap |
| Experiment tracking | mlflow |
| API | fastapi, uvicorn, pydantic |
| Dashboard | streamlit, plotly, networkx |
| Testing | pytest, pytest-cov |
| Linting / typing | ruff, mypy |
| CI/CD | GitHub Actions |
| Containerisation | Docker, docker-compose |
