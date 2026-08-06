"""
Inference predictor — loads trained artefacts and scores a single seller.

The Predictor class is the single object the API keeps in memory.
It owns:
  - EnsembleRiskScorer (classifier + anomaly detector + behavioral rules)
  - SHAPExplainer      (for per-prediction explanations)
  - feature_names      (ordered list that the models were trained on)

Design decisions:
- The API receives pre-computed features in the request body rather than
  raw transaction tables.  This avoids coupling the inference service to
  the database and keeps latency low.
- Feature alignment: the incoming dict is aligned to the exact column
  order the model was trained on.  Missing columns are filled with 0.
  Extra columns are silently dropped.  This makes the API robust to
  partial payloads from older clients.
- SHAP is optional: if the explainer was not loaded the API still returns
  risk scores; the top_risk_factors come from the ensemble's behavioral
  rule engine instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from marketplace_trust.evaluation.explainability import SHAPExplainer
from marketplace_trust.models.classifier import FraudClassifier
from marketplace_trust.models.ensemble import EnsembleRiskScorer, RiskPrediction

logger = logging.getLogger(__name__)


class Predictor:
    """
    Wraps the ensemble scorer and SHAP explainer into a single
    inference object that can be called with a feature dict.
    """

    def __init__(
        self,
        ensemble: EnsembleRiskScorer,
        feature_names: list[str],
        explainer: SHAPExplainer | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.feature_names = feature_names
        self.explainer = explainer

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_artefacts(cls, models_dir: str | Path) -> "Predictor":
        """
        Load all artefacts from a directory produced by scripts/train.py.

        Expected files:
            ensemble.pkl       — EnsembleRiskScorer
            feature_meta.json  — {"feature_names": [...], ...}
            classifier.pkl     — FraudClassifier (for SHAP, optional)
        """
        models_dir = Path(models_dir)

        logger.info("Loading artefacts from %s", models_dir)
        ensemble = EnsembleRiskScorer.load(models_dir / "ensemble.pkl")
        logger.info("Loaded EnsembleRiskScorer")

        meta_path = models_dir / "feature_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"feature_meta.json not found in {models_dir}")
        with open(meta_path) as f:
            meta = json.load(f)
        feature_names: list[str] = meta["feature_names"]
        logger.info("Feature names loaded: %d features", len(feature_names))

        # SHAP explainer is optional — skip gracefully if classifier missing
        explainer: SHAPExplainer | None = None
        classifier_path = models_dir / "classifier.pkl"
        if classifier_path.exists():
            try:
                classifier = FraudClassifier.load(classifier_path)
                explainer = SHAPExplainer(classifier)
                logger.info("SHAP explainer initialised (call fit() before using)")
            except Exception as e:
                logger.warning("Could not load SHAP explainer: %s", e)

        return cls(ensemble=ensemble, feature_names=feature_names, explainer=explainer)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        features: dict[str, float],
        seller_id: str = "unknown",
    ) -> RiskPrediction:
        """
        Score a single seller given a dict of feature values.

        Missing features are filled with 0; extra features are dropped.
        This makes the API tolerant of schema evolution.
        """
        X = self._align_features(features)
        prediction = self.ensemble.predict_single(X, seller_id=seller_id)

        # If SHAP is available, augment with SHAP-derived top factors
        if self.explainer is not None and self.explainer._explainer is not None:
            try:
                explanation = self.explainer.explain(X, seller_id=seller_id, top_n=5)
                shap_factors = [
                    f"{name} = {fval:.3g}  (SHAP: {sv:+.3f})"
                    for name, sv, fval in explanation.top_factors
                    if abs(sv) > 1e-4
                ]
                if shap_factors:
                    prediction.top_risk_factors = shap_factors
            except Exception as e:
                logger.debug("SHAP explanation failed for %s: %s", seller_id, e)

        return prediction

    def _align_features(self, features: dict[str, float]) -> pd.DataFrame:
        """
        Build a single-row DataFrame aligned to the model's training columns.

        Columns present in training but absent from the request → 0.
        Columns in the request but not in training → dropped.
        """
        row = {col: features.get(col, 0.0) for col in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names)
