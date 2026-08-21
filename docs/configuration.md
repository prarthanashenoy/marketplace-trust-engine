# Configuration

## Overview

All configurable parameters live in `configs/config.yaml`. The configuration is loaded via a Pydantic `BaseModel` hierarchy defined in `src/marketplace_trust/config.py`, providing type validation and IDE autocompletion.

---

## Full Config Reference

```yaml
# configs/config.yaml

data:
  random_seed: 42           # Seed for all random operations (reproducibility)
  n_users: 10000            # Number of users to generate
  n_sellers: 1000           # Number of sellers to generate
  n_listings: 10000         # Number of product listings
  n_transactions: 50000     # Number of transactions
  n_reviews: 30000          # Number of reviews
  n_devices: 5000           # Number of device records
  n_events: 100000          # Number of platform events
  fraud_prevalence: 0.15    # Fraction of sellers who are fraudulent
  noise_level: 0.3          # Fraud/normal feature distribution overlap
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"

model:
  temporal_split:
    train_fraction: 0.70
    val_fraction: 0.15
    test_fraction: 0.15

  xgboost:
    n_estimators: 300         # Number of boosting rounds
    max_depth: 6              # Maximum tree depth
    learning_rate: 0.05       # Shrinkage rate per tree
    scale_pos_weight: 5       # Weight ratio for positive (fraud) class
    subsample: 0.8            # Row subsampling per tree
    colsample_bytree: 0.8     # Feature subsampling per tree
    min_child_weight: 5       # Minimum sum of instance weight in a leaf
    eval_metric: "aucpr"      # Metric for early stopping
    early_stopping_rounds: 30

  isolation_forest:
    n_estimators: 200         # Number of isolation trees
    contamination: 0.15       # Expected fraction of anomalies
    max_samples: "auto"       # Subsampling per tree
    random_state: 42

  logistic_regression:
    max_iter: 1000
    class_weight: "balanced"

  ensemble:
    classifier_weight: 0.60
    anomaly_weight: 0.25
    behavioral_weight: 0.15

  risk_levels:
    LOW: [0, 24]
    MEDIUM: [25, 49]
    HIGH: [50, 74]
    CRITICAL: [75, 100]

evaluation:
  thresholds: [0.30, 0.50, 0.70, 0.90]
  top_k_values: [100, 500, 1000]
  shap_background_samples: 100   # Background samples for SHAP (speed vs. accuracy)

api:
  host: "0.0.0.0"
  port: 8000

dashboard:
  port: 8501

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "marketplace-trust-engine"
  run_name: null              # Auto-generated timestamp if null
```

---

## Pydantic Config Classes

The YAML is loaded into a hierarchy of Pydantic models (`src/marketplace_trust/config.py`):

```
MarketplaceConfig
├── DataConfig
├── ModelConfig
│   ├── TemporalSplitConfig
│   ├── XGBoostConfig
│   ├── IsolationForestConfig
│   ├── LogisticRegressionConfig
│   └── EnsembleConfig
├── EvaluationConfig
├── APIConfig
├── DashboardConfig
└── MLflowConfig
```

Loading the config:

```python
from marketplace_trust.config import load_config

cfg = load_config("configs/config.yaml")
print(cfg.model.xgboost.n_estimators)  # 300
print(cfg.data.fraud_prevalence)        # 0.15
```

Pydantic validates types at load time — a string in a numeric field raises a `ValidationError` immediately rather than failing silently at training time.

---

## Key Parameters Explained

### `noise_level`

Controls how much the synthetic fraud signals overlap with normal seller behaviour.

- `0.0` — fraud and normal are perfectly separable (trivial, unrealistic)
- `0.3` — moderate overlap (default; achievable but non-trivial PR-AUC)
- `0.5` — significant overlap (hard problem; tests model robustness)
- `0.8` — heavily overlapping (near-random; used for stress testing)

### `scale_pos_weight`

XGBoost parameter that weights gradient updates for the positive (fraud) class.

A good starting value is `(n_normal / n_fraud)`. With `fraud_prevalence: 0.15`:

```
scale_pos_weight ≈ 0.85 / 0.15 ≈ 5.67  →  use 5
```

If you increase `fraud_prevalence`, decrease `scale_pos_weight` proportionally.

### `contamination`

Isolation Forest parameter for the expected fraction of anomalies in training data. Should match `fraud_prevalence`. If your real fraud rate is lower, reduce this value to avoid over-flagging.

### `eval_metric: "aucpr"`

XGBoost uses this metric on the validation set to decide whether to continue training (early stopping). `aucpr` = area under the precision-recall curve, which is the primary evaluation metric.

### `shap_background_samples`

SHAP TreeExplainer uses a background dataset to compute expected values. More samples improve accuracy but increase computation. `100` is a reasonable default for a dataset of 1,000 sellers.

---

## Overriding Config at Runtime

All CLI scripts accept a `--config` flag:

```bash
poetry run python scripts/train.py --config configs/my_experiment.yaml
```

Environment variable override for the model directory (used by the API and Docker):

```bash
export MODEL_DIR=/app/models
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models/` | Directory containing `ensemble.pkl` and `classifier.pkl` |
| `CONFIG_PATH` | `configs/config.yaml` | Path to the YAML config file |
| `MLFLOW_TRACKING_URI` | From config | Overrides `mlflow.tracking_uri` |
