"""
Tests for model training modules.

Verifies:
1. Baseline model fits and produces valid probabilities.
2. XGBoost classifier fits with early stopping and produces valid probabilities.
3. Anomaly detector produces normalised scores in [0, 1].
4. Ensemble risk scorer produces risk scores in [0, 100].
5. Risk levels are one of the four expected values.
6. Temporal split preserves all rows and has no overlap.
7. All models can be saved and reloaded from disk.
8. Prediction output shape matches input.
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
    IsolationForestConfig,
    LogisticRegressionConfig,
    TemporalSplitConfig,
    XGBoostConfig,
)
from marketplace_trust.data.generator import MarketplaceDataGenerator
from marketplace_trust.features.pipeline import FeatureMatrix, build_feature_matrix
from marketplace_trust.models.anomaly import AnomalyDetector
from marketplace_trust.models.baseline import BaselineModel
from marketplace_trust.models.classifier import FraudClassifier
from marketplace_trust.models.ensemble import EnsembleRiskScorer, RiskPrediction
from marketplace_trust.models.splitter import temporal_split

# ---------------------------------------------------------------------------
# Fixtures — module-scoped so we only generate data once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_dataset():
    cfg = DataConfig(
        random_seed=42,
        n_users=400,
        n_sellers=100,
        n_listings=500,
        n_transactions=1000,
        n_reviews=500,
        n_devices=200,
        n_events=2000,
    )
    return MarketplaceDataGenerator(cfg).generate()


@pytest.fixture(scope="module")
def feature_matrix(raw_dataset) -> FeatureMatrix:
    ds = raw_dataset
    return build_feature_matrix(
        ds.sellers,
        ds.listings,
        ds.transactions,
        ds.reviews,
        ds.users,
        ds.events,
    )


@pytest.fixture(scope="module")
def data_split(feature_matrix, raw_dataset):
    return temporal_split(
        X=feature_matrix.X,
        y=feature_matrix.y,
        seller_ids=feature_matrix.seller_ids,
        transactions=raw_dataset.transactions,
        config=TemporalSplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
        ),
    )


@pytest.fixture(scope="module")
def fitted_baseline(data_split):
    model = BaselineModel(LogisticRegressionConfig(max_iter=200, random_state=42))
    model.fit(data_split.X_train, data_split.y_train)
    return model


@pytest.fixture(scope="module")
def fitted_classifier(data_split):
    model = FraudClassifier(
        XGBoostConfig(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5.0,
            random_state=42,
        )
    )
    model.fit(
        data_split.X_train,
        data_split.y_train,
        data_split.X_val,
        data_split.y_val,
    )
    return model


@pytest.fixture(scope="module")
def fitted_anomaly(data_split):
    model = AnomalyDetector(
        IsolationForestConfig(
            n_estimators=50,
            contamination=0.15,
            random_state=42,
        )
    )
    model.fit(data_split.X_train)
    return model


@pytest.fixture(scope="module")
def fitted_ensemble(fitted_classifier, fitted_anomaly, fitted_baseline):
    return EnsembleRiskScorer(
        classifier=fitted_classifier,
        anomaly_detector=fitted_anomaly,
        baseline=fitted_baseline,
    )


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    def test_total_rows_preserved(self, feature_matrix, data_split):
        total = len(data_split.X_train) + len(data_split.X_val) + len(data_split.X_test)
        assert total == len(feature_matrix.X)

    def test_no_seller_overlap_between_splits(self, data_split):
        train_ids = set(data_split.seller_ids_train)
        val_ids = set(data_split.seller_ids_val)
        test_ids = set(data_split.seller_ids_test)
        assert train_ids.isdisjoint(val_ids), "Train/val overlap"
        assert train_ids.isdisjoint(test_ids), "Train/test overlap"
        assert val_ids.isdisjoint(test_ids), "Val/test overlap"

    def test_train_is_largest_split(self, data_split):
        assert len(data_split.X_train) >= len(data_split.X_val)
        assert len(data_split.X_train) >= len(data_split.X_test)

    def test_fraud_rates_non_negative(self, data_split):
        assert data_split.train_fraud_rate >= 0.0
        assert data_split.val_fraud_rate >= 0.0
        assert data_split.test_fraud_rate >= 0.0


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------


class TestBaselineModel:
    def test_predict_proba_shape(self, fitted_baseline, data_split):
        proba = fitted_baseline.predict_proba(data_split.X_val)
        assert proba.shape == (len(data_split.X_val),)

    def test_probabilities_in_unit_interval(self, fitted_baseline, data_split):
        proba = fitted_baseline.predict_proba(data_split.X_val)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predictions_are_binary(self, fitted_baseline, data_split):
        preds = fitted_baseline.predict(data_split.X_val)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_save_and_load(self, fitted_baseline, data_split, tmp_path):
        path = tmp_path / "baseline.pkl"
        fitted_baseline.save(path)
        loaded = BaselineModel.load(path)
        orig = fitted_baseline.predict_proba(data_split.X_val)
        reloaded = loaded.predict_proba(data_split.X_val)
        np.testing.assert_array_almost_equal(orig, reloaded)


# ---------------------------------------------------------------------------
# XGBoost classifier
# ---------------------------------------------------------------------------


class TestFraudClassifier:
    def test_predict_proba_shape(self, fitted_classifier, data_split):
        proba = fitted_classifier.predict_proba(data_split.X_val)
        assert proba.shape == (len(data_split.X_val),)

    def test_probabilities_in_unit_interval(self, fitted_classifier, data_split):
        proba = fitted_classifier.predict_proba(data_split.X_val)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_feature_importances_sum_to_one(self, fitted_classifier):
        imp = fitted_classifier.feature_importances
        assert abs(imp.sum() - 1.0) < 1e-4

    def test_feature_importances_all_non_negative(self, fitted_classifier):
        assert (fitted_classifier.feature_importances >= 0).all()

    def test_save_and_load(self, fitted_classifier, data_split, tmp_path):
        path = tmp_path / "classifier.pkl"
        fitted_classifier.save(path)
        loaded = FraudClassifier.load(path)
        orig = fitted_classifier.predict_proba(data_split.X_val)
        reloaded = loaded.predict_proba(data_split.X_val)
        np.testing.assert_array_almost_equal(orig, reloaded)


# ---------------------------------------------------------------------------
# Anomaly detector
# ---------------------------------------------------------------------------


class TestAnomalyDetector:
    def test_anomaly_scores_in_unit_interval(self, fitted_anomaly, data_split):
        scores = fitted_anomaly.anomaly_score(data_split.X_val)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_predict_binary(self, fitted_anomaly, data_split):
        preds = fitted_anomaly.predict(data_split.X_val)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomaly_score_shape(self, fitted_anomaly, data_split):
        scores = fitted_anomaly.anomaly_score(data_split.X_val)
        assert scores.shape == (len(data_split.X_val),)

    def test_save_and_load(self, fitted_anomaly, data_split, tmp_path):
        path = tmp_path / "anomaly.pkl"
        fitted_anomaly.save(path)
        loaded = AnomalyDetector.load(path)
        orig = fitted_anomaly.anomaly_score(data_split.X_val)
        reloaded = loaded.anomaly_score(data_split.X_val)
        np.testing.assert_array_almost_equal(orig, reloaded)


# ---------------------------------------------------------------------------
# Ensemble risk scorer
# ---------------------------------------------------------------------------


class TestEnsembleRiskScorer:
    def test_risk_scores_in_range(self, fitted_ensemble, data_split):
        result = fitted_ensemble.score(data_split.X_val, data_split.seller_ids_val)
        assert (result["risk_score"] >= 0).all()
        assert (result["risk_score"] <= 100).all()

    def test_risk_levels_valid(self, fitted_ensemble, data_split):
        result = fitted_ensemble.score(data_split.X_val, data_split.seller_ids_val)
        valid_levels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert set(result["risk_level"].unique()).issubset(valid_levels)

    def test_output_has_expected_columns(self, fitted_ensemble, data_split):
        result = fitted_ensemble.score(data_split.X_val, data_split.seller_ids_val)
        expected = {
            "seller_id",
            "risk_score",
            "risk_level",
            "classifier_probability",
            "anomaly_score",
            "behavioral_risk",
        }
        assert expected.issubset(set(result.columns))

    def test_output_row_count(self, fitted_ensemble, data_split):
        result = fitted_ensemble.score(data_split.X_val, data_split.seller_ids_val)
        assert len(result) == len(data_split.X_val)

    def test_predict_single_returns_risk_prediction(self, fitted_ensemble, data_split):
        row = data_split.X_val.iloc[[0]]
        pred = fitted_ensemble.predict_single(row, seller_id="sel_000001")
        assert isinstance(pred, RiskPrediction)
        assert 0 <= pred.risk_score <= 100
        assert pred.risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert len(pred.top_risk_factors) >= 1

    def test_save_and_load(self, fitted_ensemble, data_split, tmp_path):
        path = tmp_path / "ensemble.pkl"
        fitted_ensemble.save(path)
        loaded = EnsembleRiskScorer.load(path)
        orig = fitted_ensemble.score(data_split.X_val, data_split.seller_ids_val)
        reloaded = loaded.score(data_split.X_val, data_split.seller_ids_val)
        pd.testing.assert_series_equal(orig["risk_score"], reloaded["risk_score"])
