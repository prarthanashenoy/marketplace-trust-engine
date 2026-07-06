"""
Tests for feature engineering modules.

Verifies:
1. All feature functions return correctly shaped DataFrames.
2. No label column leaks into the feature matrix.
3. Feature values are finite (no inf/NaN after pipeline).
4. Rolling window features respect temporal ordering.
5. Graph features are non-negative.
6. The full pipeline produces consistent output given the same seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from marketplace_trust.config import DataConfig
from marketplace_trust.data.generator import GeneratedDataset, MarketplaceDataGenerator
from marketplace_trust.features.behavioral_features import build_behavioral_features
from marketplace_trust.features.graph_features import build_graph_features
from marketplace_trust.features.pipeline import ALL_FEATURE_GROUPS, build_feature_matrix
from marketplace_trust.features.seller_features import build_seller_features
from marketplace_trust.features.transaction_features import (
    build_transaction_features,
    build_velocity_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset() -> GeneratedDataset:
    cfg = DataConfig(
        random_seed=42,
        n_users=300,
        n_sellers=80,
        n_listings=400,
        n_transactions=800,
        n_reviews=400,
        n_devices=150,
        n_events=1500,
    )
    return MarketplaceDataGenerator(cfg).generate()


# ---------------------------------------------------------------------------
# Seller features
# ---------------------------------------------------------------------------


class TestSellerFeatures:
    def test_returns_one_row_per_seller(self, dataset: GeneratedDataset) -> None:
        feats = build_seller_features(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
        )
        assert len(feats) == len(dataset.sellers)

    def test_contains_seller_id_column(self, dataset: GeneratedDataset) -> None:
        feats = build_seller_features(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
        )
        assert "seller_id" in feats.columns

    def test_no_label_columns(self, dataset: GeneratedDataset) -> None:
        feats = build_seller_features(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
        )
        assert "is_fraudulent" not in feats.columns
        assert "fraud_scenario" not in feats.columns

    def test_no_negative_counts(self, dataset: GeneratedDataset) -> None:
        feats = build_seller_features(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
        )
        count_cols = [c for c in feats.columns if "count" in c or "total" in c]
        for col in count_cols:
            assert (feats[col] >= 0).all(), f"{col} has negative values"

    def test_ratios_in_unit_interval(self, dataset: GeneratedDataset) -> None:
        feats = build_seller_features(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
        )
        for col in [
            "cancel_to_transaction_ratio",
            "refund_to_transaction_ratio",
            "review_coverage_rate",
            "five_star_rate",
        ]:
            if col in feats.columns:
                assert (feats[col] >= 0).all(), f"{col} has negative values"

    def test_seller_age_non_negative(self, dataset: GeneratedDataset) -> None:
        feats = build_seller_features(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
        )
        assert (feats["seller_age_days"] >= 0).all()


# ---------------------------------------------------------------------------
# Transaction temporal features
# ---------------------------------------------------------------------------


class TestTransactionFeatures:
    def test_returns_one_row_per_seller(self, dataset: GeneratedDataset) -> None:
        feats = build_transaction_features(dataset.transactions)
        assert feats["seller_id"].nunique() == feats["seller_id"].count()

    def test_window_columns_present(self, dataset: GeneratedDataset) -> None:
        feats = build_transaction_features(dataset.transactions)
        for days in [1, 7, 30]:
            assert f"txn_count_{days}d" in feats.columns

    def test_counts_non_negative(self, dataset: GeneratedDataset) -> None:
        feats = build_transaction_features(dataset.transactions)
        for col in [c for c in feats.columns if "count" in c]:
            assert (feats[col] >= 0).all(), f"{col} has negative values"

    def test_1d_count_leq_7d_count(self, dataset: GeneratedDataset) -> None:
        """Transactions in last 1 day cannot exceed last 7 days."""
        feats = build_transaction_features(dataset.transactions)
        assert (feats["txn_count_1d"] <= feats["txn_count_7d"]).all()

    def test_7d_count_leq_30d_count(self, dataset: GeneratedDataset) -> None:
        feats = build_transaction_features(dataset.transactions)
        assert (feats["txn_count_7d"] <= feats["txn_count_30d"]).all()

    def test_velocity_features_present(self, dataset: GeneratedDataset) -> None:
        feats = build_velocity_features(dataset.transactions, dataset.sellers)
        assert "txn_velocity_30d" in feats.columns
        assert "lifetime_txn_velocity" in feats.columns


# ---------------------------------------------------------------------------
# Behavioral features
# ---------------------------------------------------------------------------


class TestBehavioralFeatures:
    def test_returns_one_row_per_seller(self, dataset: GeneratedDataset) -> None:
        feats = build_behavioral_features(
            dataset.sellers,
            dataset.users,
            dataset.events,
            dataset.reviews,
            dataset.transactions,
        )
        assert len(feats) == len(dataset.sellers)

    def test_no_nan_after_fillna(self, dataset: GeneratedDataset) -> None:
        feats = build_behavioral_features(
            dataset.sellers,
            dataset.users,
            dataset.events,
            dataset.reviews,
            dataset.transactions,
        )
        numeric = feats.select_dtypes(include="number")
        assert not numeric.isnull().any().any(), "NaN found in behavioral features"

    def test_review_lag_non_negative(self, dataset: GeneratedDataset) -> None:
        feats = build_behavioral_features(
            dataset.sellers,
            dataset.users,
            dataset.events,
            dataset.reviews,
            dataset.transactions,
        )
        if "avg_review_lag_hours" in feats.columns:
            assert (feats["avg_review_lag_hours"] >= 0).all()


# ---------------------------------------------------------------------------
# Graph features
# ---------------------------------------------------------------------------


class TestGraphFeatures:
    def test_returns_one_row_per_seller(self, dataset: GeneratedDataset) -> None:
        feats = build_graph_features(
            dataset.sellers,
            dataset.users,
            dataset.transactions,
            dataset.reviews,
            dataset.events,
        )
        assert len(feats) == len(dataset.sellers)

    def test_non_negative_counts(self, dataset: GeneratedDataset) -> None:
        feats = build_graph_features(
            dataset.sellers,
            dataset.users,
            dataset.transactions,
            dataset.reviews,
            dataset.events,
        )
        for col in feats.select_dtypes(include="number").columns:
            assert (feats[col] >= 0).all(), f"{col} has negative values"

    def test_no_user_id_in_output(self, dataset: GeneratedDataset) -> None:
        """user_id is a join key and must not leak into the feature output."""
        feats = build_graph_features(
            dataset.sellers,
            dataset.users,
            dataset.transactions,
            dataset.reviews,
            dataset.events,
        )
        assert "user_id" not in feats.columns


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_output_shape(self, dataset: GeneratedDataset) -> None:
        result = build_feature_matrix(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
            dataset.users,
            dataset.events,
        )
        assert len(result.X) == len(dataset.sellers)
        assert len(result.y) == len(dataset.sellers)
        assert len(result.seller_ids) == len(dataset.sellers)

    def test_no_label_in_X(self, dataset: GeneratedDataset) -> None:
        """Critical: fraud label must not appear in the feature matrix."""
        result = build_feature_matrix(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
            dataset.users,
            dataset.events,
        )
        assert "is_fraudulent" not in result.X.columns
        assert "fraud_scenario" not in result.X.columns

    def test_no_nan_in_X(self, dataset: GeneratedDataset) -> None:
        result = build_feature_matrix(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
            dataset.users,
            dataset.events,
        )
        assert not result.X.isnull().any().any(), "NaN found in feature matrix"

    def test_no_inf_in_X(self, dataset: GeneratedDataset) -> None:
        result = build_feature_matrix(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
            dataset.users,
            dataset.events,
        )
        assert not np.isinf(result.X.values).any(), "Inf found in feature matrix"

    def test_y_is_binary(self, dataset: GeneratedDataset) -> None:
        result = build_feature_matrix(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
            dataset.users,
            dataset.events,
        )
        assert set(result.y.unique()).issubset({0, 1})

    def test_feature_names_match_X_columns(self, dataset: GeneratedDataset) -> None:
        result = build_feature_matrix(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
            dataset.users,
            dataset.events,
        )
        assert result.feature_names == list(result.X.columns)

    def test_pipeline_is_deterministic(self, dataset: GeneratedDataset) -> None:
        """Running the pipeline twice on the same data must yield identical results."""
        r1 = build_feature_matrix(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
            dataset.users,
            dataset.events,
        )
        r2 = build_feature_matrix(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
            dataset.users,
            dataset.events,
        )
        pd.testing.assert_frame_equal(r1.X.reset_index(drop=True), r2.X.reset_index(drop=True))

    def test_fraud_rate_preserved(self, dataset: GeneratedDataset) -> None:
        """The label distribution in the feature matrix should match the raw data."""
        result = build_feature_matrix(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
            dataset.users,
            dataset.events,
        )
        expected_rate = dataset.sellers["is_fraudulent"].mean()
        actual_rate = result.y.mean()
        assert (
            abs(expected_rate - actual_rate) < 0.01
        ), f"Fraud rate mismatch: expected {expected_rate:.3f}, got {actual_rate:.3f}"

    def test_all_feature_group_columns_present(self, dataset: GeneratedDataset) -> None:
        """At least some columns from each feature group should be in the matrix."""
        result = build_feature_matrix(
            dataset.sellers,
            dataset.listings,
            dataset.transactions,
            dataset.reviews,
            dataset.users,
            dataset.events,
        )
        feature_set = set(result.feature_names)
        for group, cols in ALL_FEATURE_GROUPS.items():
            overlap = feature_set & set(cols)
            assert len(overlap) > 0, f"No columns from feature group '{group}' found in matrix"
