# Evaluation Strategy

## Philosophy

The evaluation strategy is designed to mirror how the model will actually be used in production: trained on historical data and deployed on future sellers whose behaviour it has never seen.

---

## Train / Validation / Test Split

### Temporal Split (70 / 15 / 15)

Rather than a random split, sellers are ordered by their **first transaction timestamp** and divided chronologically:

```
Oldest sellers ─────────────────────────────────── Most recent sellers
│────────── 70% train ──────────│── 15% val ──│── 15% test ──│
```

**Why temporal, not random?**

A random split would allow training on sellers whose activity occurred *after* the activity of sellers in the test set. In production, this never happens — models are trained on past data and must generalise to future data. A random split produces overly optimistic evaluation metrics.

**Edge case: sellers with no transactions**
Sellers who never completed a transaction cannot be ordered by first transaction timestamp. They are assigned to the training set (the safest assumption, since they have no future-pointing signal).

---

## Metrics

### Primary Metric: PR-AUC

**Precision-Recall Area Under the Curve** is the primary evaluation metric.

**Why PR-AUC, not ROC-AUC?**

ROC-AUC measures the trade-off between True Positive Rate and False Positive Rate across all thresholds. With 85 % normal sellers, a model that predicts "normal" for everything achieves near-perfect False Positive Rate and still gets a high ROC-AUC. PR-AUC focuses on the minority class (fraud) by measuring Precision vs. Recall directly.

- **Precision** = of all sellers flagged as fraud, what fraction are actually fraudulent?
- **Recall** = of all actually fraudulent sellers, what fraction did we catch?

A model that catches more fraudulent sellers (high recall) must typically accept more false alarms (lower precision). PR-AUC summarises the full trade-off.

**Rule of thumb:** A random classifier on 15 % fraud data would achieve PR-AUC ≈ 0.15. A perfect model achieves 1.0.

### Supporting Metrics

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Area under the ROC curve. Universally comparable across domains. |
| **Precision at threshold** | Of all sellers scored above threshold T, what fraction are fraudulent? |
| **Recall at threshold** | Of all fraudulent sellers, what fraction are scored above threshold T? |
| **F1 at threshold** | Harmonic mean of precision and recall at threshold T. |

Thresholds evaluated: `[0.30, 0.50, 0.70, 0.90]`.

### Precision@K / Recall@K

Simulates an analyst review queue: "If we hand the top K highest-scored sellers to our review team, how many are actually fraudulent?"

| K | Use case |
|---|---------|
| 100 | Very small team, must be highly precise |
| 500 | Mid-size team |
| 1000 | Large-scale automated review |

Precision@K and Recall@K are computed for each K value. A model with high Precision@100 is useful when investigator time is scarce.

---

## Evaluation Artefacts

Running `make evaluate` (or `scripts/evaluate.py`) saves the following to `data/processed/`:

| File | Contents |
|------|----------|
| `eval_baseline_val.json` | Baseline metrics on validation set |
| `eval_baseline_test.json` | Baseline metrics on test set |
| `eval_classifier_val.json` | XGBoost metrics on validation set |
| `eval_classifier_test.json` | XGBoost metrics on test set |
| `eval_ensemble_val.json` | Ensemble metrics on validation set |
| `eval_ensemble_test.json` | Ensemble metrics on test set |
| `shap_importance.csv` | Mean absolute SHAP per feature (global importance) |
| `shap_values_test.parquet` | Per-prediction SHAP values on test set |

### JSON Report Structure

```json
{
  "pr_auc": 0.89,
  "roc_auc": 0.95,
  "thresholds": {
    "0.30": {"precision": 0.71, "recall": 0.88, "f1": 0.79},
    "0.50": {"precision": 0.81, "recall": 0.79, "f1": 0.80},
    "0.70": {"precision": 0.90, "recall": 0.65, "f1": 0.76},
    "0.90": {"precision": 0.97, "recall": 0.42, "f1": 0.58}
  },
  "precision_at_k": {"100": 0.95, "500": 0.82, "1000": 0.64},
  "recall_at_k": {"100": 0.14, "500": 0.58, "1000": 0.91},
  "confusion_matrix": {"tp": 84, "fp": 9, "fn": 25, "tn": 732}
}
```

---

## Val vs. Test Set Usage

| Set | Used for |
|-----|---------|
| **Val** | Hyperparameter tuning, early stopping, model selection |
| **Test** | Final one-time evaluation reported to stakeholders |

The test set is intentionally held out until the end. Evaluating on it repeatedly during development would cause implicit overfitting to the test distribution.

---

## Running Evaluation

```bash
make evaluate
# or
poetry run python scripts/evaluate.py

# Skip SHAP (faster, for quick metric checks)
poetry run python scripts/evaluate.py --no-shap
```

---

## Interpreting Results

### Expected Performance Range

On the synthetic dataset with default `noise_level: 0.3`:

- Baseline PR-AUC: ~0.60–0.75
- XGBoost PR-AUC: ~0.85–0.93
- Ensemble PR-AUC: ~0.85–0.94

The ensemble gain over the classifier alone comes from the anomaly detector catching the `account_takeover` and `new_account_abuse` scenarios, where the supervised model has limited signal.

### Overfitting Check

Compare val metrics to test metrics. A large gap (> 5 pp PR-AUC) suggests the model has overfit to the validation time period. Remedies include:

- Reducing `max_depth` or `n_estimators`
- Increasing `early_stopping_rounds` tolerance
- Reducing `noise_level` to make the problem harder
