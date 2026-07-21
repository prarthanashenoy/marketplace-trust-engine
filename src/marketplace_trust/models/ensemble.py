"""
Ensemble risk scorer.

Combines three complementary signals into a single risk score:

1. classifier_probability  — XGBoost supervised fraud probability [0, 1]
2. anomaly_score           — Isolation Forest unsupervised anomaly score [0, 1]
3. behavioral_risk_signal  — hand-crafted rule-based risk from key features

Why combine all three:
- Classifier is strong when fraud resembles past labelled patterns.
- Anomaly detector catches novel patterns the classifier has not seen.
- Behavioral rules provide transparent, auditable escalation logic that
  does not depend on black-box model outputs.

Final risk_score is scaled to [0, 100] and mapped to risk levels:
    0–24    → LOW
    25–49   → MEDIUM
    50–74   → HIGH
    75–100  → CRITICAL
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from marketplace_trust.models.anomaly import AnomalyDetector
from marketplace_trust.models.baseline import BaselineModel
from marketplace_trust.models.classifier import FraudClassifier

logger = logging.getLogger(__name__)

# Risk level thresholds (applied to the 0-100 risk_score)
RISK_LEVELS = [
    (75, "CRITICAL"),
    (50, "HIGH"),
    (25, "MEDIUM"),
    (0, "LOW"),
]

# Ensemble weights — must sum to 1.0
W_CLASSIFIER = 0.60
W_ANOMALY = 0.25
W_BEHAVIORAL = 0.15


@dataclass
class RiskPrediction:
    """Per-seller risk prediction output."""

    seller_id: str
    risk_score: int  # 0–100
    risk_level: str  # LOW / MEDIUM / HIGH / CRITICAL
    classifier_probability: float
    anomaly_score: float
    behavioral_risk: float
    top_risk_factors: list[str]


class EnsembleRiskScorer:
    """
    Combines classifier, anomaly detector, and behavioral rules into
    a single calibrated risk score.
    """

    def __init__(
        self,
        classifier: FraudClassifier,
        anomaly_detector: AnomalyDetector,
        baseline: BaselineModel | None = None,
    ) -> None:
        self.classifier = classifier
        self.anomaly_detector = anomaly_detector
        self.baseline = baseline

    def score(
        self,
        X: pd.DataFrame,
        seller_ids: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Score a feature matrix and return a DataFrame with risk outputs.

        Columns: seller_id, risk_score, risk_level, classifier_probability,
                 anomaly_score, behavioral_risk.
        """
        clf_prob = self.classifier.predict_proba(X)
        anom_score = self.anomaly_detector.anomaly_score(X)
        beh_risk = self._behavioral_risk(X)

        combined = W_CLASSIFIER * clf_prob + W_ANOMALY * anom_score + W_BEHAVIORAL * beh_risk
        risk_scores = np.clip(combined * 100, 0, 100).astype(int)
        risk_levels = [_score_to_level(s) for s in risk_scores]

        result = pd.DataFrame(
            {
                "seller_id": seller_ids.values if seller_ids is not None else range(len(X)),
                "risk_score": risk_scores,
                "risk_level": risk_levels,
                "classifier_probability": np.round(clf_prob, 4),
                "anomaly_score": np.round(anom_score, 4),
                "behavioral_risk": np.round(beh_risk, 4),
            }
        )
        return result

    def predict_single(
        self,
        X_row: pd.DataFrame,
        seller_id: str = "unknown",
        feature_names: list[str] | None = None,
    ) -> RiskPrediction:
        """
        Score a single seller row and return a RiskPrediction with
        human-readable top risk factors.
        """
        scores_df = self.score(X_row, pd.Series([seller_id]))
        row = scores_df.iloc[0]

        top_factors = self._top_risk_factors(
            X_row,
            float(row["classifier_probability"]),
            float(row["anomaly_score"]),
            float(row["behavioral_risk"]),
        )

        return RiskPrediction(
            seller_id=seller_id,
            risk_score=int(row["risk_score"]),
            risk_level=str(row["risk_level"]),
            classifier_probability=float(row["classifier_probability"]),
            anomaly_score=float(row["anomaly_score"]),
            behavioral_risk=float(row["behavioral_risk"]),
            top_risk_factors=top_factors,
        )

    # ------------------------------------------------------------------
    # Behavioral risk rules
    # ------------------------------------------------------------------

    def _behavioral_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Rule-based behavioral risk signal in [0, 1].

        Each rule fires independently and contributes a fixed weight.
        Rules are designed so that no single rule can produce risk > 0.5;
        multiple signals must co-occur to push the score high.

        Important: these rules use the same feature columns as the model,
        so they never see the label.
        """
        scores = np.zeros(len(X))

        def col(name: str) -> np.ndarray:
            if name in X.columns:
                return X[name].fillna(0).values.astype(float)
            return np.zeros(len(X))

        # Device sharing: many sellers on same device
        scores += np.clip(col("sellers_sharing_device") / 10.0, 0, 0.20)

        # IP abuse: many accounts on same IP
        scores += np.clip(col("max_accounts_per_ip") / 20.0, 0, 0.15)

        # High cancellation rate
        scores += np.clip(col("cancel_to_transaction_ratio") * 0.5, 0, 0.15)

        # High refund rate
        scores += np.clip(col("refund_to_transaction_ratio") * 0.5, 0, 0.15)

        # Suspiciously fast reviews (fake review signal)
        scores += np.clip(col("reviews_within_1h_count") / 20.0, 0, 0.10)

        # Very low price-to-market ratio (scam listing signal)
        low_price = col("min_price_to_market_ratio") < 0.4
        scores += low_price.astype(float) * 0.10

        # High crypto/wallet payment rate
        scores += np.clip(col("crypto_wallet_rate") * 0.3, 0, 0.10)

        # Very new seller with many listings (mass listing abuse)
        new_seller = col("seller_age_days") < 30
        many_listings = col("total_listings") > 50
        scores += (new_seller & many_listings).astype(float) * 0.10

        return np.clip(scores, 0.0, 1.0)

    def _top_risk_factors(
        self,
        X_row: pd.DataFrame,
        clf_prob: float,
        anom_score: float,
        beh_risk: float,
    ) -> list[str]:
        """
        Generate human-readable risk factor descriptions for a single seller.

        Returns up to 5 factors sorted by estimated contribution.
        """
        factors: list[tuple[float, str]] = []

        def val(col: str) -> float:
            return float(X_row[col].iloc[0]) if col in X_row.columns else 0.0

        if val("sellers_sharing_device") >= 3:
            factors.append(
                (
                    val("sellers_sharing_device"),
                    f"{int(val('sellers_sharing_device'))} seller accounts share the same device",
                )
            )
        if val("max_accounts_per_ip") >= 5:
            factors.append(
                (
                    val("max_accounts_per_ip"),
                    f"Up to {int(val('max_accounts_per_ip'))} accounts share the same IP address",
                )
            )
        if val("cancel_to_transaction_ratio") > 0.25:
            pct = int(val("cancel_to_transaction_ratio") * 100)
            factors.append((pct, f"High cancellation rate ({pct}% of transactions)"))
        if val("refund_to_transaction_ratio") > 0.20:
            pct = int(val("refund_to_transaction_ratio") * 100)
            factors.append((pct, f"High refund rate ({pct}% of transactions)"))
        if val("reviews_within_1h_count") >= 5:
            factors.append(
                (
                    val("reviews_within_1h_count"),
                    f"{int(val('reviews_within_1h_count'))} reviews posted within 1 hour of purchase",
                )
            )
        if val("min_price_to_market_ratio") < 0.4:
            pct = int(val("min_price_to_market_ratio") * 100)
            factors.append((100 - pct, f"Listings priced {100 - pct}% below market value"))
        if val("txn_burst_ratio_1d_7d") > 0.5:
            factors.append(
                (
                    val("txn_burst_ratio_1d_7d") * 10,
                    "Unusually high transaction burst (spike in last 24h vs last 7 days)",
                )
            )
        if val("reviewer_concentration") > 3.0:
            factors.append(
                (
                    val("reviewer_concentration"),
                    f"High reviewer concentration (avg {val('reviewer_concentration'):.1f} reviews per unique reviewer)",
                )
            )
        if val("five_star_rate") > 0.95 and val("total_reviews") > 10:
            factors.append(
                (
                    val("five_star_rate") * 10,
                    f"Suspiciously high five-star review rate ({val('five_star_rate') * 100:.0f}%)",
                )
            )
        if anom_score > 0.7:
            factors.append((anom_score * 10, "Seller behaviour is statistically anomalous"))

        # Sort by score descending, return top 5 descriptions
        factors.sort(key=lambda t: t[0], reverse=True)
        return [desc for _, desc in factors[:5]] or ["No specific risk factors identified"]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("EnsembleRiskScorer saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleRiskScorer":
        with open(path, "rb") as f:
            return pickle.load(f)


def _score_to_level(score: int) -> str:
    for threshold, level in RISK_LEVELS:
        if score >= threshold:
            return level
    return "LOW"
