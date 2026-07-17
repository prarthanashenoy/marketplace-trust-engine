"""
Isolation Forest anomaly detector.

Why anomaly detection alongside supervised classification:
- Supervised models require labelled fraud examples. Genuinely novel fraud
  patterns that have never been labelled will be missed.
- Isolation Forest is unsupervised: it assigns anomaly scores based solely
  on how easily a point is isolated from the rest of the data.
- Sellers that score highly anomalous on Isolation Forest but have low
  classifier probability may represent new, unlabelled fraud patterns.

The anomaly score is combined with the classifier probability in the
ensemble risk scorer to produce the final risk score.

Score convention:
- sklearn's decision_function returns negative values; more negative = more
  anomalous. We negate and normalise to [0, 1] for interpretability.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from marketplace_trust.config import IsolationForestConfig

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Isolation Forest anomaly detector.

    Fitted on training data only. At inference time, returns a normalised
    anomaly score in [0, 1] where 1 = most anomalous.
    """

    def __init__(self, config: IsolationForestConfig) -> None:
        self.config = config
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        # Calibration constants set during fit for score normalisation
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def fit(self, X_train: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit the Isolation Forest on training data.

        A StandardScaler is applied first so that features with very
        different scales don't dominate the isolation criterion.
        """
        self.feature_names = list(X_train.columns)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = np.clip(X_scaled, -10, 10)

        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        logger.info("Fitting IsolationForest on %d samples", len(X_train))
        self.model.fit(X_scaled)

        # Store calibration range from training scores for normalisation
        raw_scores = self.model.decision_function(X_scaled)
        # Negate: sklearn returns higher values for normals
        neg_scores = -raw_scores
        self._score_min = float(neg_scores.min())
        self._score_max = float(neg_scores.max())
        logger.info(
            "Anomaly score range on train: [%.4f, %.4f]",
            self._score_min,
            self._score_max,
        )
        return self

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return normalised anomaly scores in [0, 1].

        0 = completely normal, 1 = maximally anomalous.
        Scores are clipped to [0, 1] for values outside the training range.
        """
        assert self.model is not None and self.scaler is not None
        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, -10, 10)
        raw_scores = self.model.decision_function(X_scaled)
        neg_scores = -raw_scores

        score_range = self._score_max - self._score_min
        if score_range < 1e-8:
            return np.zeros(len(X))

        normalised = (neg_scores - self._score_min) / score_range
        return np.clip(normalised, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary anomaly flags (1 = anomalous, 0 = normal)."""
        assert self.model is not None and self.scaler is not None
        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, -10, 10)
        # IsolationForest returns -1 for anomalies, 1 for normals
        preds = self.model.predict(X_scaled)
        return (preds == -1).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("AnomalyDetector saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        with open(path, "rb") as f:
            return pickle.load(f)
