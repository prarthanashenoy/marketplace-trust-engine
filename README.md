# Marketplace Trust Engine

An end-to-end machine learning system for detecting suspicious sellers and scoring fraud risk on online marketplaces. Built as a production-style portfolio project covering the full ML lifecycle: synthetic data generation, feature engineering, model training, evaluation, explainability, a REST API, and an interactive dashboard.

---
---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Synthetic Dataset](#synthetic-dataset)
- [Feature Engineering](#feature-engineering)
- [ML Approach](#ml-approach)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [API](#api)
- [Dashboard](#dashboard)
- [Running Locally](#running-locally)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Problem Statement

Online marketplaces face significant losses from fraudulent sellers: scam listings, payment abuse, fake review networks, and organised fraud rings. Traditional rule-based systems flag obvious violations but miss novel patterns and produce high false-positive rates that hurt legitimate sellers.

The goal is a **seller risk scoring system** that:

- Assigns every seller a risk score from 0 to 100
- Combines supervised fraud detection, unsupervised anomaly detection, and interpretable rule-based signals
- Explains **why** a seller is flagged, not just that they are flagged
- Serves predictions in real time via a REST API

---

## Solution Overview

```
Raw marketplace data
        │
        ▼
┌──────────────────┐
│  Feature Pipeline │  50+ features across 5 groups
│  (4 modules)      │  (seller profile, price, transaction,
└──────────────────┘   behavioral, graph)
        │
        ▼
┌──────────────────────────────────────────────┐
│              Ensemble Risk Scorer             │
│                                              │
│  60% × XGBoost classifier probability        │
│  25% × Isolation Forest anomaly score        │
│  15% × Rule-based behavioral signal          │
└──────────────────────────────────────────────┘
        │
        ▼
  Risk Score [0–100]  →  LOW / MEDIUM / HIGH / CRITICAL
        │
        ▼
┌──────────────┐    ┌───────────────────┐
│  FastAPI     │    │  Streamlit        │
│  /predict    │    │  Dashboard        │
└──────────────┘    └───────────────────┘
```

---

## Architecture

```
marketplace-exp/
├── src/marketplace_trust/
│   ├── data/          # Synthetic data generator + validation
│   ├── features/      # Feature engineering (4 modules + pipeline)
│   ├── models/        # Baseline, XGBoost, Isolation Forest, Ensemble
│   ├── evaluation/    # Metrics, SHAP explainability, reports
│   ├── inference/     # Predictor: loads artefacts + serves predictions
│   └── api/           # FastAPI app + Pydantic schemas
├── dashboard/         # Streamlit multi-page dashboard
├── scripts/           # CLI: generate_data, train, evaluate, predict
├── configs/           # config.yaml (all hyperparameters)
├── tests/             # 126 pytest tests
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ # GitHub Actions CI
```

**Data flow at training time:**

1. `scripts/generate_data.py` → synthetic raw tables saved to `data/raw/`
2. `scripts/train.py` → feature pipeline → temporal split → fit models → save to `models/`
3. `scripts/evaluate.py` → load models → compute metrics → save JSON + SHAP artefacts

**Data flow at inference time:**

1. `POST /predict` with seller features
2. `Predictor.predict()` aligns features to training schema (missing → 0, extras dropped)
3. `EnsembleRiskScorer.predict_single()` combines three signals → `RiskPrediction`

---

## Synthetic Dataset

Because real marketplace fraud data is confidential, the project generates a fully synthetic dataset with configurable properties.

**Default scale:**

| Table        | Rows    |
|-------------|---------|
| Users        | 10,000  |
| Sellers      | 1,000   |
| Listings     | 10,000  |
| Transactions | 50,000  |
| Reviews      | 30,000  |
| Devices      | 5,000   |
| Events       | 100,000 |

**Fraud scenarios (15% of sellers):**

| Scenario              | Description |
|----------------------|-------------|
| `new_account_abuse`  | Very new accounts, many listings, high cancellation |
| `mass_listing_abuse` | Burst of identical listings far below market price |
| `scam_listings`      | Unrealistically low prices, poor images, short descriptions |
| `payment_refund_abuse` | High refund/dispute rates, crypto-heavy payments |
| `account_takeover`   | Sudden device/IP change, unusual transaction burst |
| `fraud_ring`         | Sellers sharing devices and IP addresses |
| `fake_review_network` | Reviews posted within minutes of transactions, reviewer concentration |
| `device_sharing`     | Multiple seller accounts per device |
| `mixed_suspicious`   | Combination of multiple fraud signals |

Fraud labels are assigned from the **scenario at seller creation time**, not inferred from features post-hoc. This prevents target leakage. A `noise_level` parameter (default 0.3) controls how much the fraud and normal distributions overlap, making the classification problem realistically hard.

---

## Feature Engineering

50+ features across five groups, all computed at the seller level:

**Seller Profile** — account age, listing counts, rating statistics, reviewer concentration, listings-per-day velocity, review coverage rate

**Price** — price-to-market ratios (avg/min/std), fraction of listings below 50% market price, scam price rate, image count and description length

**Transaction** — cancel/refund/dispute ratios, crypto payment rate, buyer concentration, rolling windows (1d/7d/30d counts and amounts), transaction velocity and burst ratio

**Behavioral** — device switching count, IP diversity, event-based login patterns, review timing (lag hours, std of lag, reviews-within-1h count), profile update frequency

**Graph** — sellers sharing the same device, max accounts per IP address, buyer-seller overlap across shared buyers, shared reviewer count, seller degree centrality (NetworkX)

All features are computed **before** labels are merged, making target leakage structurally impossible.

---

## ML Approach

### Temporal Train/Val/Test Split

Sellers are ordered by their **first transaction timestamp** and partitioned 70/15/15. This mirrors production deployment: models are always trained on past data and evaluated on future data. Sellers with no transactions go to the training set.

### Model 1: Logistic Regression Baseline

StandardScaler → clip(±10σ) → logistic regression with `class_weight="balanced"`, `solver="lbfgs"`, and `C=0.1` (L2 regularization). The clipping and regularization prevent weight explosion on near-perfectly-separable synthetic data. Provides a simple, interpretable baseline to compare against tree models.

### Model 2: XGBoost Classifier

- `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`
- `scale_pos_weight=5` to handle ~15% fraud prevalence
- `eval_metric="logloss"` for early stopping — more sensitive to probability calibration than PR-AUC on near-separable data
- Early stopping with 30 rounds of patience on the validation set

### Model 3: Isolation Forest Anomaly Detector

- `n_estimators=200`, `contamination=0.15`
- Trained on the **full training set** (no labels used)
- Scores are normalised to [0, 1] using training-set min/max
- Catches novel fraud patterns that the supervised model may have never seen

### Ensemble Scoring

```
risk_score = clip(
    (0.60 × classifier_probability
   + 0.25 × anomaly_score
   + 0.15 × behavioral_risk) × 100,
    0, 100
)
```

The behavioral component is a set of calibrated rules (device sharing, IP abuse, cancellation rate, refund rate, fake reviews, scam prices, new-seller burst) that provide transparent, auditable escalation independent of both black-box models.

**Risk levels:**

| Score  | Level    |
|--------|----------|
| 0–24   | LOW      |
| 25–49  | MEDIUM   |
| 50–74  | HIGH     |
| 75–100 | CRITICAL |

---

## Evaluation

Models are evaluated on both the validation and test sets. Key metrics:

- **PR-AUC** (primary) — area under the precision-recall curve; robust to class imbalance
- **ROC-AUC**
- **Precision, Recall, F1** at thresholds 0.30 / 0.50 / 0.70 / 0.90
- **Precision@K / Recall@K** at K = 100, 500, 1000 (simulates analyst review queues)

Run evaluation after training:

```bash
make evaluate
# or with flags:
poetry run python scripts/evaluate.py --no-shap   # faster, skips SHAP
```

Evaluation artefacts are saved to `data/processed/`:
- `eval_baseline_{val,test}.json`
- `eval_classifier_{val,test}.json`
- `eval_ensemble_{val,test}.json`
- `shap_importance.csv`
- `shap_values_test.parquet`

---

## Explainability

SHAP (SHapley Additive exPlanations) is used to explain individual predictions from the XGBoost classifier.

**Global importance** — mean absolute SHAP value across the test set, ranked by contribution. Saved to `data/processed/shap_importance.csv`.

**Per-prediction explanation** — for any seller, `SHAPExplainer.explain()` returns a `SellerExplanation` with:

```
cancel_to_transaction_ratio   +0.312  ↑ INCREASES risk
seller_age_days               -0.187  ↓ DECREASES risk
sellers_sharing_device        +0.145  ↑ INCREASES risk
```

**Human-readable risk factors** — the ensemble scorer also generates plain-English descriptions of the top contributing signals (e.g., *"47% cancellation rate"*, *"5 accounts sharing the same device"*, *"Listings priced 68% below market value"*). These are returned in the `/predict` API response and shown in the dashboard.

---

## API

FastAPI service with three endpoints:

| Method | Path          | Description |
|--------|--------------|-------------|
| GET    | `/health`    | Liveness check — always 200, reports `model_loaded` |
| GET    | `/model-info` | Model metadata, feature names, ensemble weights |
| POST   | `/predict`   | Score a seller, return risk assessment |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "seller_id": "sel_001234",
    "features": {
      "seller_age_days": 14,
      "total_listings": 87,
      "cancel_to_transaction_ratio": 0.42,
      "sellers_sharing_device": 5,
      "min_price_to_market_ratio": 0.31
    }
  }'
```

**Example response:**

```json
{
  "seller_id": "sel_001234",
  "risk_score": 78,
  "risk_level": "CRITICAL",
  "model_probability": 0.8821,
  "anomaly_score": 0.6134,
  "behavioral_risk": 0.4500,
  "top_risk_factors": [
    "5 seller accounts share the same device",
    "High cancellation rate (42% of transactions)",
    "Listings priced 69% below market value",
    "Seller account is only 14 days old with 87 listings"
  ]
}
```

The API starts even when no model is trained; `/predict` returns 503 with a clear message rather than crashing. Pydantic validates all feature bounds (e.g., `avg_rating` must be in [1.0, 5.0]) before the request reaches the model layer.

Interactive docs are available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

---

## Dashboard

A six-page Streamlit dashboard for exploring data and model outputs:

| Page | Description |
|------|-------------|
| **Overview** | Dataset statistics, fraud prevalence, scenario breakdown |
| **Risk Distribution** | Score histograms, risk level breakdown, score-by-scenario comparison |
| **Seller Investigation** | Search by seller ID, full risk profile, SHAP explanation, feature comparison |
| **Fraud Scenarios** | Side-by-side feature distributions for each fraud type vs. normal |
| **Model Performance** | PR/ROC curves, precision-recall at multiple thresholds, top SHAP features |
| **Network Investigation** | Shared device/IP graph visualisation with NetworkX + Plotly |

Start the dashboard:

```bash
make dashboard
# → http://localhost:8501
```

---

## Running Locally

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- macOS: `brew install libomp` (required by XGBoost)

### Setup

```bash
git clone <repo-url>
cd marketplace-exp

# Install all dependencies
make install

# Generate synthetic data (writes to data/raw/)
make generate-data

# Train models (writes to models/)
make train

# Evaluate models (writes to data/processed/)
make evaluate
```

### Run services

```bash
# REST API (http://localhost:8000)
make serve

# Dashboard (http://localhost:8501)
make dashboard
```

### Score a single seller from the CLI

```bash
# Score with zero-feature baseline
poetry run python scripts/predict.py --seller-id sel_000042

# Supply specific features
poetry run python scripts/predict.py \
    --seller-id sel_000042 \
    --features '{"seller_age_days": 7, "total_listings": 120, "cancel_to_transaction_ratio": 0.45}'

# Score from a saved parquet feature file
poetry run python scripts/predict.py \
    --from-file data/processed/features.parquet \
    --seller-id sel_000042
```

### Development

```bash
# Run 126 tests with coverage
make test

# Lint + format check
make lint

# Type checking
make typecheck

# Remove all generated artefacts
make clean
```

---

## Docker

```bash
# Start all services
docker-compose up

# Build and start in background
docker-compose up --build -d
```

Services:

| Service   | URL                      | Description |
|-----------|--------------------------|-------------|
| API        | http://localhost:8000   | FastAPI inference service |
| Dashboard  | http://localhost:8501   | Streamlit dashboard |
| MLflow     | http://localhost:5000   | Experiment tracking UI (run `poetry run mlflow ui` locally against `mlruns/`) |
| PostgreSQL | localhost:5432          | MLflow backend store |

The API container uses `MODEL_DIR=/app/models` and loads artefacts from a mounted volume. If no model exists at startup, the container still starts healthy and returns 503 on `/predict`.

---

## Project Structure

```
marketplace-exp/
│
├── configs/
│   └── config.yaml              # All hyperparameters and paths
│
├── dashboard/
│   └── app.py                   # Streamlit app (6 pages, ~900 lines)
│
├── data/
│   ├── raw/                     # Generated parquet tables
│   └── processed/               # Evaluation JSON, SHAP artefacts
│
├── models/                      # Trained model pickles + feature metadata
│
├── scripts/
│   ├── generate_data.py         # Generate and save synthetic dataset
│   ├── train.py                 # Full training pipeline + MLflow logging
│   ├── evaluate.py              # Evaluation + SHAP artefacts
│   └── predict.py               # CLI scorer for a single seller
│
├── src/marketplace_trust/
│   ├── api/
│   │   ├── main.py              # FastAPI app with lifespan loader
│   │   └── schemas.py           # Pydantic request/response models
│   ├── data/
│   │   ├── generator.py         # Synthetic data generator (seeded)
│   │   ├── schemas.py           # Pydantic table row schemas + enums
│   │   └── validation.py        # Post-generation validation checks
│   ├── evaluation/
│   │   ├── explainability.py    # SHAP wrapper (TreeExplainer)
│   │   ├── metrics.py           # PR-AUC, ROC-AUC, threshold metrics, P@K
│   │   └── reports.py           # Print and save evaluation summaries
│   ├── features/
│   │   ├── seller_features.py   # Account age, ratings, listing ratios
│   │   ├── transaction_features.py  # Rolling windows, velocity, payment mix
│   │   ├── behavioral_features.py   # Device/IP switching, review timing
│   │   ├── graph_features.py    # Shared devices/IPs, buyer overlap, centrality
│   │   └── pipeline.py          # Unified entry point → FeatureMatrix
│   ├── inference/
│   │   └── predictor.py         # Loads artefacts, aligns features, calls ensemble
│   ├── models/
│   │   ├── baseline.py          # LogisticRegression + StandardScaler
│   │   ├── classifier.py        # XGBoost FraudClassifier
│   │   ├── anomaly.py           # Isolation Forest AnomalyDetector
│   │   ├── ensemble.py          # EnsembleRiskScorer + RiskPrediction
│   │   └── splitter.py          # Temporal 70/15/15 split
│   └── config.py                # Pydantic config hierarchy, load_config()
│
├── tests/
│   ├── test_data_generator.py   # 25 tests: determinism, schema, labels, integrity
│   ├── test_features.py         # 27 tests: shape, leakage, NaN/inf, determinism
│   ├── test_models.py           # 23 tests: split, probabilities, save/load
│   ├── test_evaluation.py       # 27 tests: metric ranges, SHAP, JSON round-trip
│   └── test_api.py              # 24 tests: endpoints, 503, validation rejection
│
├── .github/workflows/ci.yml     # Lint → format check → 126 tests → coverage
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── pyproject.toml
```

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Data generation | `numpy`, `pandas` |
| Feature engineering | `pandas`, `networkx` |
| ML models | `scikit-learn`, `xgboost` |
| Explainability | `shap` |
| Experiment tracking | `mlflow` |
| API | `fastapi`, `uvicorn`, `pydantic` |
| Dashboard | `streamlit`, `plotly`, `networkx` |
| Configuration | `pydantic-settings`, `pyyaml` |
| Testing | `pytest`, `pytest-cov` |
| Linting / formatting | `ruff` |
| Type checking | `mypy` |
| Containerisation | `docker`, `docker-compose` |
| CI | GitHub Actions |

---

## Limitations and Future Work

**Current limitations:**

- The dataset is fully synthetic. Real-world fraud patterns are more complex, noisier, and evolve over time as fraudsters adapt.
- The ensemble weights (60/25/15) are hand-tuned. A calibration step (Platt scaling, isotonic regression) on a held-out set would produce better-calibrated probabilities.
- Features are computed in batch. A production system would need a low-latency feature store to serve fresh features to the API without recomputing the full pipeline.
- The model has no feedback loop. In production, analyst labels on flagged sellers would flow back into retraining.

**Potential improvements:**

- Replace the static behavioral rules with a learned threshold model (e.g., a small decision tree) for the behavioral risk component
- Add LightGBM and a stacking meta-learner to the ensemble
- Implement online feature computation using a streaming framework (Kafka + Flink)
- Add a label studio integration for analyst review and relabelling workflow
- Time-decay weighting in the temporal split to reflect recency in training
- Fairness audit: check whether risk scores are systematically skewed by seller country or category
- A/B testing framework for controlled model rollouts
