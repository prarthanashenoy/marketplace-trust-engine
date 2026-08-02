"""
SHAP-based explainability for the XGBoost fraud classifier.

Why SHAP:
- SHAP (SHapley Additive exPlanations) provides consistent, theoretically
  grounded feature attributions based on game theory.
- Unlike permutation importance or built-in tree importance, SHAP values
  are per-prediction: each seller gets their own explanation.
- SHAP values are additive: sum of all SHAP values equals the model's
  log-odds output offset by the expected value.

Important caveat:
- SHAP values explain the model's predictions, not the ground truth.
- They tell you *why the model scored this seller high*, not *why this
  seller is actually fraudulent*.
- All explanations are labelled as "model-derived signals" to be clear.

Usage:
    explainer = SHAPExplainer(classifier)
    explainer.fit(X_train)
    explanation = explainer.explain(X_row, seller_id="sel_001234")
    print(explanation.human_readable())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from marketplace_trust.models.classifier import FraudClassifier

logger = logging.getLogger(__name__)


@dataclass
class SellerExplanation:
    """SHAP-based explanation for a single seller's risk score."""

    seller_id: str
    expected_value: float  # baseline model output (log-odds)
    shap_values: np.ndarray  # per-feature SHAP values
    feature_names: list[str]
    feature_values: np.ndarray  # actual feature values for this seller
    top_n: int = 5

    @property
    def top_factors(self) -> list[tuple[str, float, float]]:
        """
        Return the top-N most impactful features as (name, shap_value, feature_value).

        Sorted by absolute SHAP value descending — features that pushed the
        score up or down the most appear first.
        """
        order = np.argsort(np.abs(self.shap_values))[::-1][: self.top_n]
        return [
            (
                self.feature_names[i],
                float(self.shap_values[i]),
                float(self.feature_values[i]),
            )
            for i in order
        ]

    def human_readable(self) -> str:
        """
        Format a human-readable explanation block.

        Example output:
            Model-derived risk signals for seller sel_001234
            ─────────────────────────────────────────────────
            + sellers_sharing_device = 8.0   (SHAP: +0.412)  ← pushed score UP
            + cancel_to_transaction_ratio = 0.34  (SHAP: +0.231)
            - avg_rating = 4.8  (SHAP: -0.089)  ← pushed score DOWN
        """
        lines = [
            f"Model-derived risk signals for seller {self.seller_id}",
            "─" * 55,
        ]
        for name, sv, fv in self.top_factors:
            direction = "↑ pushed score UP  " if sv > 0 else "↓ pushed score DOWN"
            lines.append(
                f"  {'▲' if sv > 0 else '▽'} {name:<40s} "
                f"value={fv:>8.3f}  SHAP={sv:>+.4f}  {direction}"
            )
        lines.append("\nNote: these are model-derived signals, not causal explanations.")
        return "\n".join(lines)


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer for the XGBoost FraudClassifier.

    Must be fitted on training data before calling explain().
    The background dataset is used to compute expected SHAP values.
    """

    def __init__(self, classifier: FraudClassifier) -> None:
        assert classifier.model is not None, "Classifier must be fitted first."
        self.classifier = classifier
        self.feature_names: list[str] = classifier.feature_names
        self._explainer: shap.TreeExplainer | None = None

    def fit(self, X_train: pd.DataFrame) -> "SHAPExplainer":
        """
        Create the SHAP TreeExplainer using a background sample from training data.

        TreeExplainer is fast for tree-based models and does not require
        a large background dataset.
        """
        logger.info("Fitting SHAP TreeExplainer on %d training samples", len(X_train))
        # Use the model directly — TreeExplainer doesn't need a background for XGBoost
        self._explainer = shap.TreeExplainer(
            self.classifier.model,
            feature_perturbation="interventional",
        )
        return self

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for the given feature matrix.

        Returns an array of shape (n_samples, n_features).
        Positive values push toward fraud; negative values push toward normal.
        """
        assert self._explainer is not None, "Call fit() before shap_values()"
        sv = self._explainer.shap_values(X)
        # TreeExplainer for binary classification may return a list [neg_class, pos_class]
        if isinstance(sv, list):
            return sv[1]
        return sv

    def explain(
        self, X_row: pd.DataFrame, seller_id: str = "unknown", top_n: int = 5
    ) -> SellerExplanation:
        """
        Explain a single seller's prediction.

        Parameters
        ----------
        X_row:     Single-row DataFrame with the seller's features.
        seller_id: Identifier for the explanation.
        top_n:     Number of top features to include.
        """
        assert self._explainer is not None, "Call fit() before explain()"
        sv = self.shap_values(X_row)[0]  # shape (n_features,)
        expected = float(
            self._explainer.expected_value
            if not isinstance(self._explainer.expected_value, (list, np.ndarray))
            else self._explainer.expected_value[1]
        )

        return SellerExplanation(
            seller_id=seller_id,
            expected_value=expected,
            shap_values=sv,
            feature_names=list(X_row.columns),
            feature_values=X_row.values[0],
            top_n=top_n,
        )

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        """
        Compute mean absolute SHAP values across a dataset — global feature importance.

        More reliable than built-in tree feature importance because it
        accounts for feature interactions and is consistent across models.

        Returns a Series sorted by importance descending.
        """
        sv = self.shap_values(X)
        importance = np.abs(sv).mean(axis=0)
        return pd.Series(importance, index=list(X.columns), name="mean_abs_shap").sort_values(
            ascending=False
        )

    def summary_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with SHAP values for every seller.

        Shape: (n_samples, n_features). Columns match X.columns.
        Useful for the dashboard's SHAP summary plot.
        """
        sv = self.shap_values(X)
        return pd.DataFrame(sv, columns=list(X.columns), index=X.index)
