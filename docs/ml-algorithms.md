# ML Algorithms

## Overview

The system uses three distinct model types, each addressing a different aspect of the fraud detection problem:

| Model | Algorithm | Type | Role |
|-------|-----------|------|------|
| `BaselineClassifier` | Logistic Regression | Supervised | Reference benchmark |
| `FraudClassifier` | XGBoost | Supervised | Primary fraud detector |
| `AnomalyDetector` | Isolation Forest | Unsupervised | Novel pattern detector |

---

## 1. Logistic Regression Baseline

### What It Does

A linear classifier trained on the standardised feature matrix. It models the log-odds of fraud as a linear combination of features:

```
log P(fraud) / P(normal) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

### Implementation

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
```

`class_weight="balanced"` automatically adjusts weights inversely proportional to class frequencies, compensating for the 85/15 normal/fraud imbalance without manual oversampling.

### Why Include It?

1. **Sanity check** — if XGBoost barely beats logistic regression, the features are likely mostly linearly separable. A large gap indicates that non-linear interactions matter.
2. **Interpretability** — the signed feature coefficients provide a direct, human-readable indicator of direction (positive coefficient → pushes score toward fraud).
3. **Fast training** — provides rapid iteration during feature development.

### Limitations

- Cannot model non-linear feature interactions (e.g., "low price AND new account AND device sharing are all present simultaneously").
- Sensitive to outliers unless the StandardScaler is used.
- Assumes feature independence within the linear combination.

---

## 2. XGBoost Classifier

### What It Does

XGBoost (Extreme Gradient Boosting) is the primary supervised model. It builds an ensemble of decision trees sequentially, where each tree corrects the residual errors of all previous trees. The final prediction is the weighted sum of all tree outputs passed through a sigmoid function.

### Why XGBoost for Fraud Detection?

| Property | Benefit |
|----------|---------|
| Gradient boosting | Powerful at capturing non-linear interactions between features |
| Decision trees | Naturally handles heterogeneous feature scales (no normalisation needed) |
| Built-in regularisation | L1 (`alpha`) and L2 (`lambda`) penalties reduce overfitting |
| Robustness to outliers | Tree splits are based on rank, not magnitude |
| Class imbalance handling | `scale_pos_weight` adjusts the gradient for the minority class |
| Speed | Parallel tree construction, optimised C++ core |
| SHAP compatibility | TreeExplainer provides exact SHAP values efficiently |

### Configuration

```yaml
model:
  xgboost:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
    scale_pos_weight: 5
    eval_metric: "aucpr"
    early_stopping_rounds: 30
```

**`n_estimators: 300` with `learning_rate: 0.05`**
A small learning rate with many trees generally outperforms a large learning rate with few trees. The model adds 300 trees, each contributing a small correction.

**`max_depth: 6`**
Limits each tree to 6 levels deep. Deeper trees model higher-order interactions but risk overfitting. Depth 6 allows interactions between up to 6 features at once.

**`scale_pos_weight: 5`**
With 15 % fraud prevalence, there are roughly 5.7× as many normal sellers as fraudulent ones. Setting this to 5 tells XGBoost to weight gradient updates for the positive (fraud) class 5× more heavily. This prevents the model from predicting "normal" for everything and achieving 85 % accuracy trivially.

**`eval_metric: "aucpr"`**
Training uses PR-AUC as the evaluation metric on the validation set. This is more informative than log-loss or accuracy for imbalanced problems (see [evaluation.md](evaluation.md)).

**`early_stopping_rounds: 30`**
Training stops if the validation PR-AUC has not improved for 30 consecutive rounds, preventing overfitting and saving compute.

### Gradient Boosting — How It Works

1. Start with a constant prediction (log-odds of class prevalence).
2. Compute pseudo-residuals — the gradient of the loss with respect to the current prediction.
3. Fit a new shallow decision tree to the residuals.
4. Add the tree to the ensemble with weight `learning_rate`.
5. Repeat for `n_estimators` rounds (or until early stopping triggers).

The result is an additive model where each tree specialises on the patterns the previous trees missed.

---

## 3. Isolation Forest Anomaly Detector

### What It Does

Isolation Forest is an unsupervised anomaly detection algorithm. It detects anomalies by measuring how easily each data point can be "isolated" by random feature splits.

**Core idea:** Anomalies are rare and different. They require very few random splits to isolate from the rest of the dataset, resulting in a short path length from root to leaf. Normal points are dense and similar; they require many splits to isolate.

### Algorithm

1. Randomly sample a subset of training sellers.
2. Build a random tree by repeatedly selecting a random feature and a random split point within that feature's range.
3. Record the depth at which each seller is isolated (its path length).
4. Repeat for `n_estimators` trees.
5. Average path lengths. Short average → anomaly. Long average → normal.

The anomaly score is normalised so that 0.5 = expected score for normal data and values approaching 1 indicate strong anomalies.

### Configuration

```yaml
model:
  isolation_forest:
    n_estimators: 200
    contamination: 0.15
```

**`contamination: 0.15`**
Tells the model to expect 15 % of the training set to be anomalous. This sets the decision boundary such that approximately 15 % of sellers receive an anomaly score above the threshold.

**Training without labels**
Unlike XGBoost, Isolation Forest is trained on the full training set without using the fraud labels. This is intentional — it learns "what normal looks like" and flags deviations. The anomaly scores are normalised to [0, 1] using the min/max of training scores.

### Why Include It?

Supervised models can only detect fraud patterns they have seen before. Isolation Forest catches sellers whose overall behaviour is unusual in any way, even if that specific combination of features was not present in training data. This is especially valuable for:

- **Zero-day fraud patterns** — new fraud techniques the classifier was never trained on.
- **Cross-signal anomalies** — a seller who is slightly unusual across many features simultaneously, but not extreme in any single one.

### Limitations

- Prone to false positives for sellers in naturally sparse feature regions (e.g., niche product categories with few data points).
- Does not benefit from fraud labels; it cannot be directly optimised for precision/recall.
- Anomaly scores are not calibrated probabilities.

---

## Model Comparison

| Property | Logistic Regression | XGBoost | Isolation Forest |
|----------|--------------------|---------|--------------------|
| Supervised | Yes | Yes | No |
| Handles non-linearity | No | Yes | N/A |
| Feature scaling required | Yes | No | No |
| Handles class imbalance | Via `class_weight` | Via `scale_pos_weight` | N/A |
| Interpretability | Coefficients | SHAP | Path length |
| Detects novel fraud | No | Limited | Yes |
| Primary role | Baseline | Core scorer | Anomaly boost |

---

## Model Artefacts

After training, two artefacts are saved to `models/`:

```
models/
├── ensemble.pkl       # EnsembleRiskScorer (wraps all three models)
├── classifier.pkl     # FraudClassifier (XGBoost only)
└── feature_meta.json  # Feature names and schema
```

Models are serialised with `pickle`. The ensemble loads and wraps all sub-models at load time.
