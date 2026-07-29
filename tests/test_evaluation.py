"""
Tests for evaluation metrics, SHAP explainability, and reports.

Verifies:
1. evaluate_model() returns correct shapes and value ranges.
2. Threshold metrics are self-consistent (TP + FP + FN + TN = N).
3. PR-AUC and ROC-AUC are in [0, 1].
4. Precision@K and Recall@K are in [0, 1].
5. compare_models() returns a DataFrame with expected columns.
6. SHAP values have the right shape and are finite.
7. SellerExplanation.human_readable() returns a non-empty string.
8. Global SHAP importance is sorted descending.
9. save_evaluation_json() produces valid JSON that round-trips.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from marketplace_trust.config import (
    DataConfig,
    LogisticRegressionConfig,
    TemporalSplitConfig,
    XGBoostConfig,
)
from marketplace_trust.data.generator import MarketplaceDataGenerator
from marketplace_trust.evaluation.explainability import SHAPExplainer
from marketplace_trust.evaluation.metrics import (
    compare_models,
    evaluate_model,
)
from marketplace_trust.evaluation.reports import (
    load_evaluation_json,
    save_evaluation_json,
    save_shap_importance,
)
from marketplace_trust.features.pipeline import build_feature_matrix
from marketplace_trust.models.baseline import BaselineModel
from marketplace_trust.models.classifier import FraudClassifier
from marketplace_trust.models.splitter import temporal_split

# ---------------------------------------------------------------------------
# Shared fixtures — module-scoped to generate data only once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_dataset():
    cfg = DataConfig(
        random_seed=99,
        n_users=400,
        n_sellers=120,
        n_listings=600,
        n_transactions=1200,
        n_reviews=600,
        n_devices=200,
        n_events=2000,
    )
    return MarketplaceDataGenerator(cfg).generate()


@pytest.fixture(scope="module")
def split(raw_dataset):
    fm = build_feature_matrix(
        raw_dataset.sellers,
        raw_dataset.listings,
        raw_dataset.transactions,
        raw_dataset.reviews,
        raw_dataset.users,
        raw_dataset.events,
    )
    return temporal_split(
        X=fm.X,
        y=fm.y,
        seller_ids=fm.seller_ids,
        transactions=raw_dataset.transactions,
        config=TemporalSplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15),
    ), fm


@pytest.fixture(scope="module")
def fitted_classifier(split):
    ds, fm = split
    model = FraudClassifier(
        XGBoostConfig(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.15,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5.0,
            random_state=99,
        )
    )
    model.fit(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
    return model


@pytest.fixture(scope="module")
def fitted_baseline(split):
    ds, _ = split
    model = BaselineModel(LogisticRegressionConfig(max_iter=300, random_state=99))
    model.fit(ds.X_train, ds.y_train)
    return model


@pytest.fixture(scope="module")
def eval_classifier(split, fitted_classifier):
    ds, _ = split
    proba = fitted_classifier.predict_proba(ds.X_test)
    return evaluate_model(
        model_name="classifier",
        split_name="test",
        y_true=ds.y_test,
        y_prob=proba,
        thresholds=[0.30, 0.50, 0.70],
        top_k_values=[10, 30],
    )


@pytest.fixture(scope="module")
def eval_baseline(split, fitted_baseline):
    ds, _ = split
    proba = fitted_baseline.predict_proba(ds.X_test)
    return evaluate_model(
        model_name="baseline",
        split_name="test",
        y_true=ds.y_test,
        y_prob=proba,
        thresholds=[0.30, 0.50, 0.70],
        top_k_values=[10, 30],
    )


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_pr_auc_in_unit_interval(self, eval_classifier):
        assert 0.0 <= eval_classifier.pr_auc <= 1.0

    def test_roc_auc_in_unit_interval(self, eval_classifier):
        assert 0.0 <= eval_classifier.roc_auc <= 1.0

    def test_threshold_metrics_count(self, eval_classifier):
        assert len(eval_classifier.threshold_metrics) == 3  # 0.30, 0.50, 0.70

    def test_confusion_matrix_totals(self, eval_classifier):
        """TP + FP + FN + TN must equal total sample count."""
        for tm in eval_classifier.threshold_metrics:
            total = tm.tp + tm.fp + tm.fn + tm.tn
            assert total == eval_classifier.n_samples, (
                f"Threshold {tm.threshold}: confusion matrix sums to {total}, "
                f"expected {eval_classifier.n_samples}"
            )

    def test_threshold_precision_in_unit_interval(self, eval_classifier):
        for tm in eval_classifier.threshold_metrics:
            assert 0.0 <= tm.precision <= 1.0
            assert 0.0 <= tm.recall <= 1.0
            assert 0.0 <= tm.f1 <= 1.0

    def test_flagged_rate_in_unit_interval(self, eval_classifier):
        for tm in eval_classifier.threshold_metrics:
            assert 0.0 <= tm.flagged_rate <= 1.0

    def test_top_k_metrics_present(self, eval_classifier):
        assert len(eval_classifier.top_k_metrics) >= 1

    def test_precision_at_k_in_unit_interval(self, eval_classifier):
        for km in eval_classifier.top_k_metrics:
            assert 0.0 <= km.precision_at_k <= 1.0

    def test_recall_at_k_in_unit_interval(self, eval_classifier):
        for km in eval_classifier.top_k_metrics:
            assert 0.0 <= km.recall_at_k <= 1.0

    def test_pr_curve_shapes_match(self, eval_classifier):
        assert len(eval_classifier.precision_curve) == len(eval_classifier.recall_curve)

    def test_roc_curve_shapes_match(self, eval_classifier):
        assert len(eval_classifier.fpr_curve) == len(eval_classifier.tpr_curve)

    def test_n_fraud_consistent(self, eval_classifier, split):
        ds, _ = split
        assert eval_classifier.n_fraud == int(ds.y_test.sum())

    def test_higher_threshold_lower_flagged_rate(self, eval_classifier):
        """As threshold rises, fewer sellers should be flagged."""
        rates = [tm.flagged_rate for tm in eval_classifier.threshold_metrics]
        assert rates == sorted(
            rates, reverse=True
        ), "Flagged rate should decrease as threshold increases"


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_returns_dataframe(self, eval_classifier, eval_baseline):
        df = compare_models([eval_classifier, eval_baseline])
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_evaluation(self, eval_classifier, eval_baseline):
        df = compare_models([eval_classifier, eval_baseline])
        assert len(df) == 2

    def test_contains_required_columns(self, eval_classifier, eval_baseline):
        df = compare_models([eval_classifier, eval_baseline])
        assert "model" in df.columns
        assert "pr_auc" in df.columns
        assert "roc_auc" in df.columns


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------


class TestSHAPExplainer:
    @pytest.fixture(scope="class")
    def explainer(self, split, fitted_classifier):
        ds, _ = split
        exp = SHAPExplainer(fitted_classifier)
        exp.fit(ds.X_train)
        return exp, ds

    def test_shap_values_shape(self, explainer):
        exp, ds = explainer
        sv = exp.shap_values(ds.X_test)
        assert sv.shape == (len(ds.X_test), ds.X_test.shape[1])

    def test_shap_values_finite(self, explainer):
        exp, ds = explainer
        sv = exp.shap_values(ds.X_test)
        assert np.isfinite(sv).all(), "SHAP values contain NaN or Inf"

    def test_global_importance_sorted(self, explainer):
        exp, ds = explainer
        imp = exp.global_importance(ds.X_test)
        values = imp.values
        assert (values[:-1] >= values[1:]).all(), "Global importance not sorted descending"

    def test_global_importance_non_negative(self, explainer):
        exp, ds = explainer
        imp = exp.global_importance(ds.X_test)
        assert (imp >= 0).all()

    def test_explain_single_returns_explanation(self, explainer):
        from marketplace_trust.evaluation.explainability import SellerExplanation

        exp, ds = explainer
        row = ds.X_test.iloc[[0]]
        explanation = exp.explain(row, seller_id="sel_test_001")
        assert isinstance(explanation, SellerExplanation)
        assert explanation.seller_id == "sel_test_001"

    def test_human_readable_non_empty(self, explainer):
        exp, ds = explainer
        row = ds.X_test.iloc[[0]]
        explanation = exp.explain(row, seller_id="sel_test_001")
        text = explanation.human_readable()
        assert len(text) > 0
        assert "sel_test_001" in text

    def test_top_factors_count(self, explainer):
        exp, ds = explainer
        row = ds.X_test.iloc[[0]]
        explanation = exp.explain(row, top_n=5)
        assert len(explanation.top_factors) <= 5

    def test_summary_df_shape(self, explainer):
        exp, ds = explainer
        df = exp.summary_df(ds.X_test)
        assert df.shape == ds.X_test.shape


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class TestReports:
    def test_save_and_load_evaluation_json(self, eval_classifier, tmp_path):
        path = tmp_path / "eval.json"
        save_evaluation_json(eval_classifier, path)
        assert path.exists()
        data = load_evaluation_json(path)
        assert data["model_name"] == "classifier"
        assert data["split_name"] == "test"
        assert "pr_auc" in data
        assert "threshold_metrics" in data
        assert len(data["threshold_metrics"]) == 3

    def test_json_pr_auc_matches(self, eval_classifier, tmp_path):
        path = tmp_path / "eval2.json"
        save_evaluation_json(eval_classifier, path)
        data = load_evaluation_json(path)
        assert abs(data["pr_auc"] - eval_classifier.pr_auc) < 1e-6

    def test_save_shap_importance(self, split, fitted_classifier, tmp_path):
        exp = SHAPExplainer(fitted_classifier)
        ds, _ = split
        exp.fit(ds.X_train)
        importance = exp.global_importance(ds.X_test)
        path = tmp_path / "shap.csv"
        save_shap_importance(importance, path, top_n=10)
        assert path.exists()
        df = pd.read_csv(path)
        assert "feature" in df.columns
        assert "mean_abs_shap" in df.columns
        assert len(df) <= 10
