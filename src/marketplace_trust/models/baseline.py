"""
Logistic Regression baseline model.

Purpose:
- Establishes a simple, interpretable lower bound on performance.
- If the tree-based model barely beats logistic regression, it suggests
  the problem is mostly linearly separable and simpler than expected.
- Useful as a sanity check: logistic regression should achieve non-trivial
  precision/recall if the features are meaningful.

Preprocessing:
- StandardScaler is required because logistic regression is sensitive to
  feature scale (unlike tree-based models).
- We use a Pipeline so that the scaler is fit only on training data and
  applied consistently to val/test, preventing data leakage.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from marketplace_trust.config import LogisticRegressionConfig

logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Logistic Regression baseline wrapped in a sklearn Pipeline.

    The pipeline applies StandardScaler before fitting, ensuring that
    features with large absolute values (e.g. transaction counts) don't
    dominate purely due to scale.
    """

    def __init__(self, config: LogisticRegressionConfig) -> None:
        self.config = config
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaselineModel":
        """Fit the scaler + logistic regression on training data."""
        self.feature_names = list(X_train.columns)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clipper", FunctionTransformer(np.clip, kw_args={"a_min": -10, "a_max": 10})),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                        class_weight="balanced",
                        solver="lbfgs",
                        C=self.config.C,
                    ),
                ),
            ]
        )
        logger.info("Fitting baseline LogisticRegression on %d samples", len(X_train))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
            self.pipeline.fit(X_train, y_train)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of the positive (fraud) class."""
        assert self.pipeline is not None, "Model not fitted. Call fit() first."
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
            return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at the given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Baseline model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        with open(path, "rb") as f:
            return pickle.load(f)
