# Explainability

## Why Explainability Matters in Fraud Detection

Raw risk scores are not enough. Analysts, compliance officers, and sellers themselves need to understand *why* a seller was flagged. Without explanations:

- Analysts cannot verify whether a flag is a true positive or a false alarm.
- Compliance teams cannot document enforcement decisions.
- Sellers who appeal cannot receive a fair hearing.

The system provides two layers of explainability:

1. **Per-prediction SHAP values** — which features pushed the XGBoost score up or down.
2. **Behavioral rules** — which rule thresholds were crossed.

---

## SHAP (SHapley Additive exPlanations)

### What SHAP Measures

SHAP values are grounded in cooperative game theory (Shapley values). For a given prediction, the SHAP value of feature `x_i` answers: "How much did this feature's value contribute to moving the model's prediction away from the average prediction?"

Formally, for a prediction `f(x)`:

```
f(x) = base_value + SHAP(x₁) + SHAP(x₂) + ... + SHAP(xₙ)
```

Where `base_value` is the model's average prediction across all training sellers.

**Properties:**
- SHAP values sum exactly to the difference between the prediction and the base value (local accuracy).
- Features are fairly credited even when correlated (unlike naive feature importance).
- Values are additive, making them easy to interpret: positive SHAP → pushed score up, negative SHAP → pushed score down.

### TreeExplainer

For XGBoost (and other tree-based models), SHAP provides `TreeExplainer`, which computes exact SHAP values in polynomial time by exploiting the tree structure. This is much faster than the model-agnostic `KernelExplainer`.

```python
import shap
explainer = shap.TreeExplainer(xgboost_model)
shap_values = explainer.shap_values(X_test)
# shap_values shape: (n_sellers, n_features)
```

### Per-Prediction Explanations

For any seller, `SHAPExplainer.explain()` returns the top 5 features by absolute SHAP value:

```
Seller sel_000042 — Risk Score: 78 (CRITICAL)

Top contributing features:
  ▲  cancel_to_transaction_ratio = 0.42    SHAP: +0.231   ↑ pushed score UP
  ▲  sellers_sharing_device = 5            SHAP: +0.198   ↑ pushed score UP
  ▲  min_price_to_market_ratio = 0.31      SHAP: +0.144   ↑ pushed score UP
  ▲  seller_age_days = 14                  SHAP: +0.089   ↑ pushed score UP
  ▽  avg_rating = 4.8                      SHAP: -0.051   ↓ pushed score DOWN
```

The Streamlit dashboard's **Seller Investigation** page renders this explanation interactively.

### Global Feature Importance

Across the entire test set, SHAP global importance is computed as the **mean absolute SHAP value** per feature:

```python
shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
```

This is saved to `data/processed/shap_importance.csv` and displayed on the **Model Performance** page.

Global SHAP importance differs from XGBoost's built-in `feature_importances_` (which uses gain or split count). SHAP importance is more reliable because:
- It accounts for feature interactions.
- It is not inflated by high-cardinality features.
- It has a consistent unit (model output scale).

---

## Rule-Based Explanations

The behavioral risk component (15 % of the ensemble score) produces human-readable sentences when a rule threshold is crossed:

| Rule | Example explanation |
|------|---------------------|
| Device sharing | `"5 seller accounts share the same device"` |
| IP abuse | `"IP address shared by 8 accounts"` |
| High cancellation | `"High cancellation rate (42% of transactions)"` |
| High refund rate | `"High refund rate (28% of transactions)"` |
| Rapid reviews | `"31% of reviews were posted within 1 hour"` |
| Underpricing | `"Listings priced 69% below market value"` |
| Crypto payments | `"73% of payments via crypto/wallet"` |
| New-account velocity | `"Seller is only 14 days old with 87 active listings"` |

These are included in the `top_risk_factors` field of every `RiskPrediction`, alongside the top SHAP features.

---

## Limitations and Caveats

### SHAP Explains the Model, Not Ground Truth

SHAP values explain **why the model assigned a high score**, not **why the seller is actually fraudulent**. If the model has a systematic bias — for example, over-weighting `crypto_payment_rate` for a legitimate seller operating in a crypto-native market — SHAP will faithfully surface this as a top factor even if it is wrong.

### Correlation vs. Causation

A high SHAP value for `cancel_to_transaction_ratio` means the model heavily relied on that feature for this prediction. It does not mean cancellations *caused* the fraud. Correlation in the training data drives the model.

### Explainability of Isolation Forest

Isolation Forest anomaly scores do not have SHAP explanations computed in the current implementation. Only the XGBoost component is explained. The behavioral rules are self-explanatory by design.

### Feature Coverage

SHAP explanations are only as good as the features. A fraud pattern that the feature pipeline does not capture will not appear in the SHAP output, even if the model assigns a high score via proxy features.

---

## Using Explainability in Practice

**Analyst workflow (Seller Investigation page):**
1. Search for a seller ID.
2. Review the risk score, level, and component breakdown.
3. Read the top risk factors (SHAP + rules).
4. Compare this seller's feature values to the normal seller baseline (shown as a radar/bar chart).
5. Make an enforcement decision with a documented justification.

**Model debugging workflow:**
1. Find a false positive (normal seller with high score).
2. Read its SHAP explanation — which features drove the score?
3. Investigate whether those features are correlated with fraud in training data for the wrong reason.
4. Update features, thresholds, or training data accordingly.
