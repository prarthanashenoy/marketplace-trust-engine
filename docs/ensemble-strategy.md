# Ensemble Strategy

## Why an Ensemble?

No single modelling approach fully covers the fraud problem:

| Approach | Strength | Weakness |
|----------|----------|----------|
| Supervised ML | High precision on known patterns | Blind to novel fraud |
| Anomaly detection | Catches new patterns | Many false positives |
| Rule-based | Transparent, auditable | Brittle, easily evaded |

The ensemble combines all three to get the benefits of each while limiting their individual weaknesses.

---

## Risk Score Formula

```
risk_score = clip(
    (0.60 × classifier_probability
   + 0.25 × anomaly_score
   + 0.15 × behavioral_risk) × 100,
    0, 100
)
```

The raw weighted sum is multiplied by 100 and clipped to [0, 100] to produce an integer risk score.

### Component Breakdown

#### Component 1: Classifier Probability (60 % weight)

- Source: `FraudClassifier.predict_proba()` — XGBoost sigmoid output in [0, 1].
- Reflects the model's calibrated estimate that this seller is fraudulent based on historical training data.
- Given the highest weight because it is the most directly trained signal.

#### Component 2: Anomaly Score (25 % weight)

- Source: `AnomalyDetector.score()` — Isolation Forest anomaly score normalised to [0, 1].
- Reflects how unusual this seller's feature vector is compared to typical sellers.
- Given a significant but secondary weight; complements the supervised signal by flagging sellers who are "weird" even if they don't match a known fraud pattern.

#### Component 3: Behavioral Risk (15 % weight)

- Source: Eight hard-coded rules that each contribute a partial score.
- Provides a transparent, auditable floor. Even if both ML models miss a seller, clear rule violations escalate the score.
- Kept at 15 % to avoid overriding the data-driven models with rigid heuristics.

---

## Behavioral Risk Rules

The behavioral risk component is the sum of eight rule contributions, clipped to [0, 1].

| Rule | Trigger | Max contribution |
|------|---------|-----------------|
| Device sharing | `sellers_sharing_device > 2` | 0.20 |
| IP abuse | `max_accounts_per_ip > 3` | 0.15 |
| High cancellation rate | `cancel_to_transaction_ratio > 0.3` | 0.15 |
| High refund rate | `refund_to_transaction_ratio > 0.2` | 0.15 |
| Rapid reviews | `rapid_review_rate > 0.3` | 0.10 |
| Deep underpricing | `min_price_to_market_ratio < 0.4` | 0.10 |
| Crypto payment dominance | `crypto_payment_rate > 0.5` | 0.10 |
| New-account velocity | `seller_age_days < 30 AND total_listings > 50` | 0.10 |

Contributions are scaled proportionally within each rule's range (not binary on/off), so a seller at twice the threshold gets a higher contribution than one just at the threshold.

**Why keep rules at all?**
Rules are interpretable and auditable. Compliance teams, analysts, and merchants understand "this seller has 7 accounts on the same device" far better than "the ensemble score is 81". The rule component ensures that clear violations always appear in the `top_risk_factors` explanation.

---

## Risk Level Classification

The scalar score is bucketed into four named levels:

| Score range | Risk level |
|-------------|-----------|
| 0–24 | LOW |
| 25–49 | MEDIUM |
| 50–74 | HIGH |
| 75–100 | CRITICAL |

These thresholds are configurable in `configs/config.yaml`. In production, teams would calibrate thresholds based on analyst capacity and acceptable false-positive rates.

---

## Ensemble Weights — How They Were Chosen

The current weights (0.60 / 0.25 / 0.15) are **hand-tuned** based on:

1. **Supervised dominance** — the XGBoost classifier, trained directly on fraud labels, is the most reliable signal and deserves the majority of the weight.
2. **Anomaly supplementation** — Isolation Forest adds value for novel fraud but is noisier; it contributes a quarter of the score.
3. **Rules as floor** — rules provide a minimum escalation signal and should not override the ML models.

### Known Limitation

Hand-tuned weights are a pragmatic starting point, not optimal. Better approaches include:

- **Platt scaling** — fit a logistic regression layer on top of the three component scores using the validation set, learning the optimal weights and calibration simultaneously.
- **Isotonic regression** — non-parametric calibration, more flexible but requires more validation data.
- **Stacking / meta-learner** — train a second-level model (e.g., a small logistic regression) whose inputs are the three component scores. This allows the meta-learner to discover non-linear interactions between components.

---

## Per-Prediction Output

`EnsembleRiskScorer.predict_single()` returns a `RiskPrediction` dataclass:

```python
@dataclass
class RiskPrediction:
    seller_id: str
    risk_score: int                  # 0–100
    risk_level: str                  # LOW / MEDIUM / HIGH / CRITICAL
    model_probability: float         # XGBoost P(fraud)
    anomaly_score: float             # Isolation Forest score [0,1]
    behavioral_risk: float           # Rule-based score [0,1]
    top_risk_factors: list[str]      # Human-readable explanations
```

### Top Risk Factors

Factors are derived from:
1. SHAP values of the top contributing features (from XGBoost).
2. Rule contributions that exceeded their thresholds.

They are rendered as plain-English sentences, for example:

```
"5 seller accounts share the same device"
"High cancellation rate (42% of transactions)"
"Listings priced 69% below market value"
"Seller account is only 14 days old with 87 listings"
```

---

## Inference Latency

The ensemble performs three operations per prediction:

1. XGBoost tree traversal — sub-millisecond for ~300 trees.
2. Isolation Forest path length computation — sub-millisecond for ~200 trees.
3. Rule evaluation — pure Python arithmetic, negligible.

Total inference latency is typically **< 5 ms** per seller on a standard CPU, excluding feature computation.
