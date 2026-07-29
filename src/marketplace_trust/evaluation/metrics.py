"""
Evaluation metrics for fraud detection.

Why not accuracy?
Fraud detection is a heavily imbalanced problem. A model that predicts
"not fraud" for every seller would achieve >85% accuracy while being
completely useless. The metrics below all focus on model behaviour on
the minority (fraud) class.

Metric guide:
- Precision: of all sellers flagged as fraud, what fraction actually are?
  Low precision = many false positives = wasted analyst time.
- Recall: of all actual fraud sellers, what fraction did we catch?
  Low recall = missed fraud = real harm.
- F1: harmonic mean of precision and recall — single summary score.
- PR-AUC: area under the precision-recall curve across all thresholds.
  Better than ROC-AUC for imbalanced problems. Ranges [0, 1].
- ROC-AUC: area under the ROC curve. Less sensitive to imbalance but
  universally reported for comparability.
- Precision@K / Recall@K: if an analyst can only review the top K flagged
  sellers, how precise and complete is that review queue?
- Threshold analysis: shows the precision/recall tradeoff at specific
  operating thresholds so stakeholders can choose their risk tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ThresholdMetrics:
    """Metrics at a single classification threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    flagged_count: int
    flagged_rate: float


@dataclass
class TopKMetrics:
    """Precision and recall when reviewing the top-K highest-risk sellers."""

    k: int
    precision_at_k: float
    recall_at_k: float
    fraud_in_top_k: int


@dataclass
class ModelEvaluation:
    """Full evaluation results for one model on one dataset split."""

    model_name: str
    split_name: str  # "val" or "test"
    n_samples: int
    n_fraud: int
    fraud_rate: float
    pr_auc: float
    roc_auc: float
    # Metrics at each threshold from config
    threshold_metrics: list[ThresholdMetrics] = field(default_factory=list)
    # Precision@K and Recall@K
    top_k_metrics: list[TopKMetrics] = field(default_factory=list)
    # Curve data for plotting
    precision_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    recall_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    fpr_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr_curve: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        """Return a compact single-line summary."""
        return (
            f"{self.model_name} [{self.split_name}] | "
            f"PR-AUC: {self.pr_auc:.4f} | ROC-AUC: {self.roc_auc:.4f} | "
            f"fraud rate: {self.fraud_rate:.2%}"
        )


def evaluate_model(
    model_name: str,
    split_name: str,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
    top_k_values: list[int] | None = None,
) -> ModelEvaluation:
    """
    Compute a full suite of evaluation metrics for a binary classifier.

    Parameters
    ----------
    model_name:  Human-readable name for the model.
    split_name:  "val" or "test".
    y_true:      Ground-truth binary labels (0/1).
    y_prob:      Predicted probabilities for the positive class.
    thresholds:  List of thresholds for threshold analysis.
    top_k_values: List of K values for precision/recall@K.

    Returns
    -------
    ModelEvaluation dataclass with all computed metrics.
    """
    if thresholds is None:
        thresholds = [0.30, 0.50, 0.70, 0.90]
    if top_k_values is None:
        top_k_values = [100, 500, 1000]

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_samples = len(y_true)
    n_fraud = int(y_true.sum())
    fraud_rate = float(y_true.mean())

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    # PR and ROC curves for plotting
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)

    # Threshold analysis
    threshold_metrics = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        flagged = int(y_pred.sum())
        threshold_metrics.append(
            ThresholdMetrics(
                threshold=t,
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1=float(f1_score(y_true, y_pred, zero_division=0)),
                tp=int(tp),
                fp=int(fp),
                fn=int(fn),
                tn=int(tn),
                flagged_count=flagged,
                flagged_rate=flagged / n_samples,
            )
        )

    # Precision@K and Recall@K
    # Sort by descending probability, then check the top-K entries
    sorted_idx = np.argsort(y_prob)[::-1]
    top_k_metrics = []
    for k in top_k_values:
        if k > n_samples:
            continue
        top_k_labels = y_true[sorted_idx[:k]]
        fraud_in_top_k = int(top_k_labels.sum())
        top_k_metrics.append(
            TopKMetrics(
                k=k,
                precision_at_k=fraud_in_top_k / k,
                recall_at_k=fraud_in_top_k / max(n_fraud, 1),
                fraud_in_top_k=fraud_in_top_k,
            )
        )

    return ModelEvaluation(
        model_name=model_name,
        split_name=split_name,
        n_samples=n_samples,
        n_fraud=n_fraud,
        fraud_rate=fraud_rate,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        threshold_metrics=threshold_metrics,
        top_k_metrics=top_k_metrics,
        precision_curve=precision_curve,
        recall_curve=recall_curve,
        fpr_curve=fpr_curve,
        tpr_curve=tpr_curve,
    )


def compare_models(evaluations: list[ModelEvaluation]) -> pd.DataFrame:
    """
    Return a DataFrame comparing multiple model evaluations side-by-side.

    Columns: model_name, split, PR-AUC, ROC-AUC, and metrics at 0.50 threshold.
    """
    rows = []
    for ev in evaluations:
        row: dict[str, object] = {
            "model": ev.model_name,
            "split": ev.split_name,
            "pr_auc": round(ev.pr_auc, 4),
            "roc_auc": round(ev.roc_auc, 4),
            "n_samples": ev.n_samples,
            "fraud_rate": round(ev.fraud_rate, 4),
        }
        # Add threshold-specific columns
        for tm in ev.threshold_metrics:
            t = tm.threshold
            row[f"precision@{t}"] = round(tm.precision, 4)
            row[f"recall@{t}"] = round(tm.recall, 4)
            row[f"f1@{t}"] = round(tm.f1, 4)
        rows.append(row)
    return pd.DataFrame(rows)
