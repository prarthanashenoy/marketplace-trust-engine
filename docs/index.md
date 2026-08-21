# Marketplace Trust Engine — Documentation

This folder contains technical documentation for the Marketplace Trust Engine project.

## Contents

| File | Description |
|------|-------------|
| [architecture.md](architecture.md) | System architecture, component breakdown, and data flow |
| [data-pipeline.md](data-pipeline.md) | Synthetic data generation, schema, and fraud injection strategy |
| [feature-engineering.md](feature-engineering.md) | All 50+ features, groupings, and engineering decisions |
| [ml-algorithms.md](ml-algorithms.md) | ML algorithms used, why they were chosen, and configuration |
| [ensemble-strategy.md](ensemble-strategy.md) | Ensemble design, component weights, and risk scoring logic |
| [evaluation.md](evaluation.md) | Evaluation strategy, metrics, and train/val/test split rationale |
| [explainability.md](explainability.md) | SHAP explainability, per-prediction explanations, and limitations |
| [api-reference.md](api-reference.md) | FastAPI endpoint reference with request/response examples |
| [configuration.md](configuration.md) | All configurable parameters and the Pydantic config system |
| [deployment.md](deployment.md) | Docker, docker-compose, and local development setup |

## Quick Start

```bash
make install          # Install dependencies
make generate-data    # Synthesise marketplace data
make train            # Train all models
make evaluate         # Run evaluation and generate SHAP
make serve            # Start FastAPI on :8000
make dashboard        # Start Streamlit on :8501
make test             # Run 126 unit tests
```

## Project at a Glance

The Marketplace Trust Engine is an end-to-end ML system that assigns every seller on an online marketplace a **risk score from 0 to 100**. The score combines:

- **60 %** — XGBoost supervised fraud classifier
- **25 %** — Isolation Forest anomaly detector
- **15 %** — Rule-based behavioural signals

Risk levels:

| Score | Level |
|-------|-------|
| 0–24 | LOW |
| 25–49 | MEDIUM |
| 50–74 | HIGH |
| 75–100 | CRITICAL |
