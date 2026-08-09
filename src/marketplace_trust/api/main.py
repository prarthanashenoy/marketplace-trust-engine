"""
FastAPI inference service for the Marketplace Trust Engine.

Endpoints:
    POST /predict      — score a seller and return risk assessment
    GET  /health       — liveness check (used by Docker health checks)
    GET  /model-info   — metadata about the loaded model

Design notes:
- The Predictor is loaded once at startup via a lifespan context manager
  and shared across all requests (thread-safe for read-only inference).
- MODEL_DIR is read from the environment variable MODEL_DIR, defaulting
  to "models/".  This makes Docker deployment simple.
- When no trained model exists (e.g. first run before training), the API
  still starts but returns 503 on /predict with a clear error message.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from marketplace_trust.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from marketplace_trust.inference.predictor import Predictor
from marketplace_trust.models.ensemble import W_ANOMALY, W_BEHAVIORAL, W_CLASSIFIER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_predictor: Predictor | None = None
_model_load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model on startup; release resources on shutdown."""
    global _predictor, _model_load_error

    model_dir = Path(os.environ.get("MODEL_DIR", "models"))
    logger.info("Loading model artefacts from %s", model_dir)

    try:
        _predictor = Predictor.from_artefacts(model_dir)
        logger.info("Model loaded successfully (%d features)", len(_predictor.feature_names))
    except FileNotFoundError as e:
        _model_load_error = str(e)
        logger.warning(
            "Model artefacts not found (%s). "
            "Run 'make train' first. The API will return 503 on /predict.",
            e,
        )
    except Exception as e:
        _model_load_error = str(e)
        logger.error("Failed to load model: %s", e)

    yield  # application runs

    _predictor = None
    logger.info("Model released")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Marketplace Trust Engine",
    description=(
        "ML-powered risk scoring API for detecting suspicious sellers " "on online marketplaces."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """
    Liveness check.

    Returns 200 whether or not a model is loaded, so the container
    itself is considered healthy.  Use /model-info to check model status.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,
        version="0.1.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded. {_model_load_error or 'Run make train first.'}",
        )
    return ModelInfoResponse(
        model_type="EnsembleRiskScorer (XGBoost + IsolationForest + Rules)",
        n_features=len(_predictor.feature_names),
        feature_names=_predictor.feature_names,
        risk_levels={
            "0-24": "LOW",
            "25-49": "MEDIUM",
            "50-74": "HIGH",
            "75-100": "CRITICAL",
        },
        weights={
            "classifier_probability": W_CLASSIFIER,
            "anomaly_score": W_ANOMALY,
            "behavioral_risk": W_BEHAVIORAL,
        },
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Score a seller and return a risk assessment.

    The request body contains the seller's pre-computed feature values.
    Missing features default to 0 (safe for new sellers with no history).

    Example request:
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
    """
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded. {_model_load_error or 'Run make train first.'}",
        )

    # Convert Pydantic model to plain dict, excluding unset alias fields
    feature_dict = request.features.model_dump(
        exclude={"seller_age_days_velocity"},
        exclude_none=False,
    )

    try:
        prediction = _predictor.predict(
            features=feature_dict,
            seller_id=request.seller_id,
        )
    except Exception as e:
        logger.exception("Prediction failed for seller %s", request.seller_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {e}",
        )

    return PredictResponse(
        seller_id=prediction.seller_id,
        risk_score=prediction.risk_score,
        risk_level=prediction.risk_level,
        model_probability=round(prediction.classifier_probability, 4),
        anomaly_score=round(prediction.anomaly_score, 4),
        behavioral_risk=round(prediction.behavioral_risk, 4),
        top_risk_factors=prediction.top_risk_factors,
    )
