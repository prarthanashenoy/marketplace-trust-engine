# API Reference

## Base URL

```
http://localhost:8000
```

Interactive documentation is available at `/docs` (Swagger UI) and `/redoc`.

---

## Authentication

No authentication is required in the current implementation. In production, add an API key header (e.g., `X-API-Key`) via FastAPI middleware.

---

## Endpoints

### GET `/health`

Liveness and model status check.

**Response 200:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "0.1.0"
}
```

`model_loaded` is `false` if the ensemble could not be loaded at startup. The API continues to serve `/health` and `/model-info` but returns 503 on `/predict`.

**Use case:** Docker health checks, load balancer liveness probes.

---

### GET `/model-info`

Returns metadata about the loaded model.

**Response 200:**

```json
{
  "model_type": "EnsembleRiskScorer",
  "n_features": 52,
  "feature_names": ["seller_age_days", "total_listings", "..."],
  "weights": {
    "classifier": 0.60,
    "anomaly": 0.25,
    "behavioral": 0.15
  },
  "risk_thresholds": {
    "LOW": [0, 24],
    "MEDIUM": [25, 49],
    "HIGH": [50, 74],
    "CRITICAL": [75, 100]
  }
}
```

**Response 503** (model not loaded):

```json
{
  "detail": "Model not loaded. Run training pipeline first."
}
```

---

### POST `/predict`

Score a single seller and return a risk prediction.

#### Request Body

```json
{
  "seller_id": "sel_001234",
  "features": {
    "seller_age_days": 14,
    "total_listings": 87,
    "cancel_to_transaction_ratio": 0.42,
    "sellers_sharing_device": 5,
    "min_price_to_market_ratio": 0.31
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `seller_id` | string | Yes | Unique seller identifier (for tracing) |
| `features` | object | Yes | Key-value map of feature names to values |

**Feature handling:**
- **Missing features** → filled with 0 (safe default for new sellers).
- **Extra features** → silently dropped (backward-compatible with new feature additions).
- **Invalid feature values** → rejected by Pydantic validators (e.g., `avg_rating` must be in [1.0, 5.0]).

You do not need to provide all 50+ features. Partial feature sets are acceptable; the model will treat missing features as 0.

#### Response 200

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

| Field | Type | Description |
|-------|------|-------------|
| `seller_id` | string | Echo of the input seller ID |
| `risk_score` | integer | Overall risk score 0–100 |
| `risk_level` | string | `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` |
| `model_probability` | float | XGBoost P(fraud) in [0, 1] |
| `anomaly_score` | float | Isolation Forest score in [0, 1] |
| `behavioral_risk` | float | Rule-based score in [0, 1] |
| `top_risk_factors` | array[string] | Human-readable explanations |

#### Response 422 (Validation Error)

Returned when Pydantic rejects an invalid field value:

```json
{
  "detail": [
    {
      "loc": ["body", "features", "avg_rating"],
      "msg": "ensure this value is less than or equal to 5.0",
      "type": "value_error.number.not_le"
    }
  ]
}
```

#### Response 503 (Model Not Loaded)

```json
{
  "detail": "Model not loaded. Run 'make train' before starting the API."
}
```

---

## Error Handling

| Status | Cause |
|--------|-------|
| 422 | Request body fails Pydantic validation |
| 503 | Ensemble model artefact not found at startup |
| 500 | Unexpected internal error (check server logs) |

---

## Starting the API

### Local

```bash
make serve
# or
poetry run uvicorn marketplace_trust.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker-compose up api
```

The model is loaded from `MODEL_DIR` (default: `models/`). Override via environment variable:

```bash
MODEL_DIR=/path/to/models uvicorn marketplace_trust.api.main:app
```

---

## Example: cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "seller_id": "sel_000042",
    "features": {
      "seller_age_days": 14,
      "total_listings": 120,
      "cancel_to_transaction_ratio": 0.45,
      "sellers_sharing_device": 7
    }
  }'
```

## Example: Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "seller_id": "sel_000042",
        "features": {
            "seller_age_days": 14,
            "total_listings": 120,
            "cancel_to_transaction_ratio": 0.45,
        }
    }
)
prediction = response.json()
print(prediction["risk_score"])       # e.g. 82
print(prediction["risk_level"])       # CRITICAL
print(prediction["top_risk_factors"]) # list of strings
```

---

## Design Notes

- **Model loaded once at startup** via FastAPI's lifespan context manager. No per-request overhead from loading the model.
- **Thread-safe** — the model is read-only during inference; no state is mutated per request.
- **CORS enabled** — the API accepts cross-origin requests to support the Streamlit dashboard and external integrations.
- **Schema evolution** — adding new features to the model does not break existing API clients; their missing features are filled with 0.
