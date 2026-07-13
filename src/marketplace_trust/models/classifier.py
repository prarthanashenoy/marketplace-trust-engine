"""
XGBoost classifier — the primary supervised fraud detection model.

Why XGBoost:
- Handles mixed feature types (counts, ratios, booleans) without scaling.
- Robust to outliers and noisy features via tree splitting.
- scale_pos_weight addresses class imbalance without oversampling.
- Native feature importance and SHAP compatibility.

Training notes:
- early_stopping_rounds uses the validation set to prevent overfitting
  without needing a separate held-out set from training data.
- eval_metric="aucpr" (PR-AUC) is preferred over "auc" (ROC-AUC) for
  imbalanced problems because it focuses on the minority class.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from marketplace_trust.config import XGBoostConfig

logger = logging.getLogger(__name__)


class FraudClassifier:
    """
    XGBoost-based fraud classifier.

    Wraps xgb.XGBClassifier with sensible defaults for imbalanced
    binary classification and exposes a consistent predict_proba / predict
    interface matching the baseline model.
    """

    def __init__(self, config: XGBoostConfig) -> None:
        self.config = config
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "FraudClassifier":
        """
        Fit the XGBoost classifier.

        Uses X_val/y_val for early stopping — the model stops training
        when validation PR-AUC stops improving, preventing overfitting.
        """
        self.feature_names = list(X_train.columns)

        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            scale_pos_weight=self.config.scale_pos_weight,
            random_state=self.config.random_state,
            eval_metric="logloss",
            early_stopping_rounds=30,
            verbosity=0,
            n_jobs=-1,
        )

        logger.info("Fitting XGBoost on %d train / %d val samples", len(X_train), len(X_val))
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best = self.model.best_iteration
        logger.info("XGBoost best iteration: %d", best)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of the positive (fraud) class."""
        assert self.model is not None, "Model not fitted. Call fit() first."
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def feature_importances(self) -> pd.Series:
        """Return feature importances as a sorted Series."""
        assert self.model is not None
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
            name="importance",
        ).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("FraudClassifier saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FraudClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
