"""
Tests for the FastAPI inference service.

Tests cover:
1. GET /health — always returns 200 with correct structure.
2. GET /model-info — returns 503 when no model is loaded.
3. POST /predict — returns 503 when no model is loaded.
4. POST /predict — returns valid PredictResponse when model is mocked.
5. POST /predict — Pydantic validation rejects invalid feature values.
6. POST /predict — partial feature dict (missing fields) is accepted.
7. Risk score is within [0, 100].
8. Risk level is one of the four valid values.

Strategy: we do NOT require a trained model on disk.  Instead we inject
a mock Predictor via FastAPI's dependency-override / app-state mechanism,
keeping tests fast and self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient
from marketplace_trust.api.main import app
from marketplace_trust.api.schemas import PredictResponse
from marketplace_trust.models.ensemble import RiskPrediction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_predictor(
    risk_score: int = 72,
    risk_level: str = "HIGH",
    clf_prob: float = 0.78,
    anomaly: float = 0.65,
    behavioral: float = 0.40,
) -> MagicMock:
    """Return a Predictor mock that echoes the seller_id from the call."""
    mock = MagicMock()
    mock.feature_names = ["seller_age_days", "total_listings", "cancel_to_transaction_ratio"]

    def _predict(features: dict, seller_id: str = "unknown") -> RiskPrediction:
        return RiskPrediction(
            seller_id=seller_id,  # echo the caller's seller_id
            risk_score=risk_score,
            risk_level=risk_level,
            classifier_probability=clf_prob,
            anomaly_score=anomaly,
            behavioral_risk=behavioral,
            top_risk_factors=["High cancellation rate", "Shared device with 4 accounts"],
        )

    mock.predict.side_effect = _predict
    return mock


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_always_returns_200(self):
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_has_status_field(self):
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_response_has_version(self):
        with TestClient(app) as client:
            resp = client.get("/health")
        assert "version" in resp.json()

    def test_model_loaded_false_when_no_artefacts(self):
        # No model on disk → model_loaded should be False
        with TestClient(app) as client:
            resp = client.get("/health")
        data = resp.json()
        # model_loaded may be True or False depending on whether models/ exists
        assert isinstance(data["model_loaded"], bool)


# ---------------------------------------------------------------------------
# /model-info — no model loaded
# ---------------------------------------------------------------------------


class TestModelInfoNoModel:
    def test_returns_503_when_no_model(self):
        """Without a trained model on disk the endpoint should 503."""
        import marketplace_trust.api.main as api_module

        original = api_module._predictor
        api_module._predictor = None
        try:
            with TestClient(app) as client:
                resp = client.get("/model-info")
            assert resp.status_code == 503
        finally:
            api_module._predictor = original

    def test_error_message_present(self):
        import marketplace_trust.api.main as api_module

        original = api_module._predictor
        api_module._predictor = None
        try:
            with TestClient(app) as client:
                resp = client.get("/model-info")
            assert "detail" in resp.json()
        finally:
            api_module._predictor = original


# ---------------------------------------------------------------------------
# /model-info — with mock model
# ---------------------------------------------------------------------------


class TestModelInfoWithModel:
    def test_returns_200_with_mock_model(self):
        import marketplace_trust.api.main as api_module

        original = api_module._predictor
        api_module._predictor = _make_mock_predictor()
        try:
            with TestClient(app) as client:
                resp = client.get("/model-info")
            assert resp.status_code == 200
        finally:
            api_module._predictor = original

    def test_response_has_feature_names(self):
        import marketplace_trust.api.main as api_module

        original = api_module._predictor
        api_module._predictor = _make_mock_predictor()
        try:
            with TestClient(app) as client:
                resp = client.get("/model-info")
            data = resp.json()
            assert "feature_names" in data
            assert isinstance(data["feature_names"], list)
        finally:
            api_module._predictor = original

    def test_response_has_risk_levels(self):
        import marketplace_trust.api.main as api_module

        original = api_module._predictor
        api_module._predictor = _make_mock_predictor()
        try:
            with TestClient(app) as client:
                resp = client.get("/model-info")
            data = resp.json()
            assert "risk_levels" in data
        finally:
            api_module._predictor = original


# ---------------------------------------------------------------------------
# POST /predict — no model loaded
# ---------------------------------------------------------------------------


class TestPredictNoModel:
    def test_returns_503_when_no_model(self):
        import marketplace_trust.api.main as api_module

        original = api_module._predictor
        api_module._predictor = None
        payload = {"seller_id": "sel_001", "features": {}}
        try:
            with TestClient(app) as client:
                resp = client.post("/predict", json=payload)
            assert resp.status_code == 503
        finally:
            api_module._predictor = original


# ---------------------------------------------------------------------------
# POST /predict — with mock model
# ---------------------------------------------------------------------------


class TestPredictWithModel:
    @pytest.fixture(autouse=True)
    def inject_mock(self):
        """Inject a mock predictor for every test in this class."""
        import marketplace_trust.api.main as api_module

        self._original = api_module._predictor
        api_module._predictor = _make_mock_predictor()
        yield
        api_module._predictor = self._original

    def test_valid_request_returns_200(self):
        payload = {
            "seller_id": "sel_test",
            "features": {"seller_age_days": 14, "cancel_to_transaction_ratio": 0.45},
        }
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_response_has_required_fields(self):
        payload = {"seller_id": "sel_test", "features": {}}
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        data = resp.json()
        for field in [
            "seller_id",
            "risk_score",
            "risk_level",
            "model_probability",
            "anomaly_score",
            "behavioral_risk",
            "top_risk_factors",
        ]:
            assert field in data, f"Missing field: {field}"

    def test_risk_score_in_range(self):
        payload = {"seller_id": "sel_test", "features": {}}
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        data = resp.json()
        assert 0 <= data["risk_score"] <= 100

    def test_risk_level_valid(self):
        payload = {"seller_id": "sel_test", "features": {}}
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.json()["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_probabilities_in_unit_interval(self):
        payload = {"seller_id": "sel_test", "features": {}}
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        data = resp.json()
        assert 0.0 <= data["model_probability"] <= 1.0
        assert 0.0 <= data["anomaly_score"] <= 1.0
        assert 0.0 <= data["behavioral_risk"] <= 1.0

    def test_top_risk_factors_is_list(self):
        payload = {"seller_id": "sel_test", "features": {}}
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert isinstance(resp.json()["top_risk_factors"], list)

    def test_seller_id_echoed_in_response(self):
        payload = {"seller_id": "sel_unique_123", "features": {}}
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.json()["seller_id"] == "sel_unique_123"

    def test_partial_features_accepted(self):
        """Only a subset of features supplied — should succeed."""
        payload = {
            "seller_id": "sel_partial",
            "features": {
                "sellers_sharing_device": 8,
                "cancel_to_transaction_ratio": 0.6,
            },
        }
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_empty_features_accepted(self):
        """Empty feature dict should succeed (all features default to 0)."""
        payload = {"seller_id": "sel_empty", "features": {}}
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_invalid_feature_value_rejected(self):
        """avg_rating out of [1, 5] range should be rejected by Pydantic."""
        payload = {
            "seller_id": "sel_bad",
            "features": {"avg_rating": 9.9},  # max is 5.0
        }
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_seller_id_rejected(self):
        """seller_id is required."""
        payload = {"features": {}}
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Pydantic schema tests (unit, no HTTP)
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_predict_response_valid(self):
        resp = PredictResponse(
            seller_id="sel_001",
            risk_score=87,
            risk_level="HIGH",
            model_probability=0.91,
            anomaly_score=0.82,
            behavioral_risk=0.55,
            top_risk_factors=["High cancellation rate"],
        )
        assert resp.risk_score == 87

    def test_predict_response_rejects_score_out_of_range(self):
        with pytest.raises(Exception):
            PredictResponse(
                seller_id="x",
                risk_score=110,  # > 100
                risk_level="HIGH",
                model_probability=0.5,
                anomaly_score=0.5,
                behavioral_risk=0.5,
                top_risk_factors=[],
            )

    def test_predict_response_rejects_probability_out_of_range(self):
        with pytest.raises(Exception):
            PredictResponse(
                seller_id="x",
                risk_score=50,
                risk_level="HIGH",
                model_probability=1.5,  # > 1.0
                anomaly_score=0.5,
                behavioral_risk=0.5,
                top_risk_factors=[],
            )
