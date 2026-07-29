"""
Evaluation report generation.

Produces human-readable text reports and saves metric artefacts to disk.
These are used by:
- scripts/evaluate.py  (printed to stdout and saved as JSON)
- dashboard/app.py     (loaded for the Model Performance page)

Design principle: reports never embed hardcoded numbers. All figures
come from the actual ModelEvaluation objects produced by metrics.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from marketplace_trust.evaluation.metrics import ModelEvaluation, compare_models

logger = logging.getLogger(__name__)


def print_evaluation_report(evaluation: ModelEvaluation) -> None:
    """Print a formatted evaluation report to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  MODEL: {evaluation.model_name}  |  SPLIT: {evaluation.split_name.upper()}")
    print(sep)
    print(f"  Samples : {evaluation.n_samples:,}")
    print(f"  Fraud   : {evaluation.n_fraud:,} ({evaluation.fraud_rate:.2%})")
    print(f"  PR-AUC  : {evaluation.pr_auc:.4f}")
    print(f"  ROC-AUC : {evaluation.roc_auc:.4f}")

    print("\n  Threshold Analysis")
    print(
        f"  {'Thresh':>7}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  "
        f"{'Flagged':>7}  {'Flag%':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}"
    )
    print("  " + "-" * 70)
    for tm in evaluation.threshold_metrics:
        print(
            f"  {tm.threshold:>7.2f}  {tm.precision:>9.4f}  {tm.recall:>6.4f}  "
            f"{tm.f1:>6.4f}  {tm.flagged_count:>7,}  {tm.flagged_rate:>5.1%}  "
            f"{tm.tp:>5}  {tm.fp:>5}  {tm.fn:>5}"
        )

    if evaluation.top_k_metrics:
        print("\n  Precision@K / Recall@K")
        print(f"  {'K':>6}  {'Prec@K':>8}  {'Rec@K':>7}  {'Fraud in K':>10}")
        print("  " + "-" * 40)
        for km in evaluation.top_k_metrics:
            print(
                f"  {km.k:>6,}  {km.precision_at_k:>8.4f}  "
                f"{km.recall_at_k:>7.4f}  {km.fraud_in_top_k:>10,}"
            )
    print()


def print_comparison_table(evaluations: list[ModelEvaluation]) -> None:
    """Print a side-by-side model comparison table."""
    df = compare_models(evaluations)
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))
    print()


def save_evaluation_json(
    evaluation: ModelEvaluation,
    output_path: str | Path,
) -> None:
    """
    Serialise an evaluation to JSON for downstream consumption
    (dashboard, CI artefacts, MLflow logging).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_name": evaluation.model_name,
        "split_name": evaluation.split_name,
        "n_samples": evaluation.n_samples,
        "n_fraud": evaluation.n_fraud,
        "fraud_rate": evaluation.fraud_rate,
        "pr_auc": evaluation.pr_auc,
        "roc_auc": evaluation.roc_auc,
        "threshold_metrics": [
            {
                "threshold": tm.threshold,
                "precision": tm.precision,
                "recall": tm.recall,
                "f1": tm.f1,
                "tp": tm.tp,
                "fp": tm.fp,
                "fn": tm.fn,
                "tn": tm.tn,
                "flagged_count": tm.flagged_count,
                "flagged_rate": tm.flagged_rate,
            }
            for tm in evaluation.threshold_metrics
        ],
        "top_k_metrics": [
            {
                "k": km.k,
                "precision_at_k": km.precision_at_k,
                "recall_at_k": km.recall_at_k,
                "fraud_in_top_k": km.fraud_in_top_k,
            }
            for km in evaluation.top_k_metrics
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved evaluation → %s", output_path)


def load_evaluation_json(path: str | Path) -> dict:
    """Load a previously saved evaluation JSON file."""
    with open(path) as f:
        return json.load(f)


def save_shap_importance(
    importance: pd.Series,
    output_path: str | Path,
    top_n: int = 30,
) -> None:
    """
    Save global SHAP feature importance to CSV.

    Parameters
    ----------
    importance:  Series from SHAPExplainer.global_importance().
    output_path: Destination file path.
    top_n:       Save only the top-N features.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = importance.head(top_n).reset_index()
    top.columns = ["feature", "mean_abs_shap"]
    top.to_csv(output_path, index=False)
    logger.info("Saved SHAP importance (%d features) → %s", len(top), output_path)
