"""Evaluation package."""

from marketplace_trust.evaluation.explainability import SellerExplanation, SHAPExplainer
from marketplace_trust.evaluation.metrics import (
    ModelEvaluation,
    ThresholdMetrics,
    TopKMetrics,
    compare_models,
    evaluate_model,
)
from marketplace_trust.evaluation.reports import (
    print_comparison_table,
    print_evaluation_report,
    save_evaluation_json,
    save_shap_importance,
)

__all__ = [
    "evaluate_model",
    "compare_models",
    "ModelEvaluation",
    "ThresholdMetrics",
    "TopKMetrics",
    "SHAPExplainer",
    "SellerExplanation",
    "print_evaluation_report",
    "print_comparison_table",
    "save_evaluation_json",
    "save_shap_importance",
]
