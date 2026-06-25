"""
Tests for the synthetic data generator.

These tests verify:
1. Deterministic generation (same seed -> same data)
2. Correct schema (all required columns present)
3. Value ranges (no invalid data)
4. Referential integrity (foreign keys are valid)
5. Label generation (fraud labels come from scenarios)
6. No target leakage in label assignment
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest
from marketplace_trust.config import DataConfig
from marketplace_trust.data.generator import GeneratedDataset, MarketplaceDataGenerator
from marketplace_trust.data.validation import validate_dataset


@pytest.fixture(scope="module")
def small_config() -> DataConfig:
    """Small dataset config for fast tests."""
    return DataConfig(
        random_seed=42,
        n_users=500,
        n_sellers=100,
        n_listings=500,
        n_transactions=1000,
        n_reviews=500,
        n_devices=200,
        n_events=2000,
        fraud_prevalence=0.15,
        noise_level=0.3,
    )


@pytest.fixture(scope="module")
def dataset(small_config: DataConfig) -> GeneratedDataset:
    """Generate a small dataset for testing."""
    generator = MarketplaceDataGenerator(small_config)
    return generator.generate()


class TestDeterminism:
    """Data generation must be deterministic given the same seed."""

    def test_same_seed_same_sellers(self, small_config: DataConfig) -> None:
        gen1 = MarketplaceDataGenerator(small_config)
        gen2 = MarketplaceDataGenerator(small_config)
        ds1 = gen1.generate()
        ds2 = gen2.generate()
        pd.testing.assert_frame_equal(ds1.sellers, ds2.sellers)

    def test_same_seed_same_transactions(self, small_config: DataConfig) -> None:
        gen1 = MarketplaceDataGenerator(small_config)
        gen2 = MarketplaceDataGenerator(small_config)
        ds1 = gen1.generate()
        ds2 = gen2.generate()
        pd.testing.assert_frame_equal(ds1.transactions, ds2.transactions)

    def test_different_seed_different_data(self, small_config: DataConfig) -> None:
        config2 = small_config.model_copy(update={"random_seed": 999})
        gen1 = MarketplaceDataGenerator(small_config)
        gen2 = MarketplaceDataGenerator(config2)
        ds1 = gen1.generate()
        ds2 = gen2.generate()
        # Seller IDs will be the same, but behavior should differ
        assert not ds1.sellers["average_rating"].equals(ds2.sellers["average_rating"])


class TestSchema:
    """All required columns must be present."""

    def test_users_schema(self, dataset: GeneratedDataset) -> None:
        required = [
            "user_id",
            "account_created_at",
            "country",
            "device_id",
            "account_age_days",
            "verification_status",
        ]
        for col in required:
            assert col in dataset.users.columns, f"users missing column: {col}"

    def test_sellers_schema(self, dataset: GeneratedDataset) -> None:
        required = [
            "seller_id",
            "user_id",
            "seller_created_at",
            "category",
            "average_rating",
            "number_of_listings",
            "number_of_transactions",
            "cancellation_rate",
            "fraud_scenario",
            "is_fraudulent",
        ]
        for col in required:
            assert col in dataset.sellers.columns, f"sellers missing column: {col}"

    def test_listings_schema(self, dataset: GeneratedDataset) -> None:
        required = [
            "listing_id",
            "seller_id",
            "category",
            "price",
            "market_price",
            "created_at",
            "description_length",
            "image_count",
            "listing_status",
        ]
        for col in required:
            assert col in dataset.listings.columns, f"listings missing column: {col}"

    def test_transactions_schema(self, dataset: GeneratedDataset) -> None:
        required = [
            "transaction_id",
            "buyer_id",
            "seller_id",
            "listing_id",
            "amount",
            "timestamp",
            "payment_method",
            "status",
            "cancellation_flag",
            "refund_flag",
        ]
        for col in required:
            assert col in dataset.transactions.columns, f"transactions missing column: {col}"

    def test_reviews_schema(self, dataset: GeneratedDataset) -> None:
        required = [
            "review_id",
            "buyer_id",
            "seller_id",
            "transaction_id",
            "rating",
            "timestamp",
            "text_length",
        ]
        for col in required:
            assert col in dataset.reviews.columns, f"reviews missing column: {col}"


class TestValueRanges:
    """All values must be within valid ranges."""

    def test_seller_ratings_in_range(self, dataset: GeneratedDataset) -> None:
        ratings = dataset.sellers["average_rating"]
        assert (ratings >= 1.0).all(), "Ratings below 1.0 found"
        assert (ratings <= 5.0).all(), "Ratings above 5.0 found"

    def test_cancellation_rate_in_range(self, dataset: GeneratedDataset) -> None:
        rates = dataset.sellers["cancellation_rate"]
        assert (rates >= 0.0).all()
        assert (rates <= 1.0).all()

    def test_listing_prices_positive(self, dataset: GeneratedDataset) -> None:
        assert (dataset.listings["price"] > 0).all()
        assert (dataset.listings["market_price"] > 0).all()

    def test_transaction_amounts_positive(self, dataset: GeneratedDataset) -> None:
        assert (dataset.transactions["amount"] > 0).all()

    def test_review_ratings_in_range(self, dataset: GeneratedDataset) -> None:
        ratings = dataset.reviews["rating"]
        assert (ratings >= 1.0).all()
        assert (ratings <= 5.0).all()


class TestReferentialIntegrity:
    """Foreign keys must reference valid primary keys."""

    def test_sellers_reference_valid_users(self, dataset: GeneratedDataset) -> None:
        valid_user_ids = set(dataset.users["user_id"])
        seller_user_ids = set(dataset.sellers["user_id"])
        assert seller_user_ids.issubset(valid_user_ids), "Some sellers reference non-existent users"

    def test_listings_reference_valid_sellers(self, dataset: GeneratedDataset) -> None:
        valid_seller_ids = set(dataset.sellers["seller_id"])
        listing_seller_ids = set(dataset.listings["seller_id"])
        assert listing_seller_ids.issubset(
            valid_seller_ids
        ), "Some listings reference non-existent sellers"

    def test_transactions_reference_valid_sellers(self, dataset: GeneratedDataset) -> None:
        valid_seller_ids = set(dataset.sellers["seller_id"])
        txn_seller_ids = set(dataset.transactions["seller_id"])
        assert txn_seller_ids.issubset(valid_seller_ids)

    def test_transactions_reference_valid_buyers(self, dataset: GeneratedDataset) -> None:
        valid_user_ids = set(dataset.users["user_id"])
        txn_buyer_ids = set(dataset.transactions["buyer_id"])
        assert txn_buyer_ids.issubset(valid_user_ids)


class TestFraudLabels:
    """Fraud labels must be consistent with scenarios."""

    def test_normal_sellers_not_fraudulent(self, dataset: GeneratedDataset) -> None:
        sellers = dataset.sellers
        normal = sellers[sellers["fraud_scenario"] == "normal"]
        assert not normal["is_fraudulent"].any(), "Normal sellers should not be marked fraudulent"

    def test_fraud_scenario_sellers_are_fraudulent(self, dataset: GeneratedDataset) -> None:
        sellers = dataset.sellers
        non_normal = sellers[sellers["fraud_scenario"] != "normal"]
        assert non_normal[
            "is_fraudulent"
        ].all(), "All non-normal scenario sellers should be marked fraudulent"

    def test_fraud_rate_reasonable(self, dataset: GeneratedDataset) -> None:
        fraud_rate = dataset.sellers["is_fraudulent"].mean()
        # Should be between 5% and 50% given default weights
        assert (
            0.05 <= fraud_rate <= 0.50
        ), f"Fraud rate {fraud_rate:.2%} outside expected range [5%, 50%]"

    def test_multiple_scenarios_present(self, dataset: GeneratedDataset) -> None:
        scenarios = dataset.sellers["fraud_scenario"].unique()
        assert len(scenarios) >= 3, f"Expected at least 3 scenarios, got: {scenarios}"

    def test_fraud_label_not_in_features(self, dataset: GeneratedDataset) -> None:
        """
        The is_fraudulent column must NOT be a direct function of any
        single observable feature (basic leakage check).

        We verify that fraudulent and normal sellers have overlapping
        distributions for key features (not perfectly separable).
        """
        sellers = dataset.sellers
        fraud = sellers[sellers["is_fraudulent"]]["average_rating"]
        normal = sellers[~sellers["is_fraudulent"]]["average_rating"]

        # The ranges should overlap: fraud sellers don't all have rating=1
        assert fraud.max() > normal.min(), "Ratings appear perfectly separated (possible leakage)"
        assert normal.max() > fraud.min(), "Ratings appear perfectly separated (possible leakage)"


class TestValidation:
    """The validation layer should catch data quality issues."""

    def test_full_dataset_passes_validation(self, dataset: GeneratedDataset) -> None:
        result = validate_dataset(
            users=dataset.users,
            sellers=dataset.sellers,
            listings=dataset.listings,
            transactions=dataset.transactions,
            reviews=dataset.reviews,
            devices=dataset.devices,
            events=dataset.events,
        )
        assert result.passed, f"Validation failed: {result.errors}"

    def test_missing_column_fails_validation(self, dataset: GeneratedDataset) -> None:
        bad_sellers = dataset.sellers.drop(columns=["average_rating"])
        result = validate_dataset(
            users=dataset.users,
            sellers=bad_sellers,
            listings=dataset.listings,
            transactions=dataset.transactions,
            reviews=dataset.reviews,
            devices=dataset.devices,
            events=dataset.events,
        )
        assert not result.passed
        assert any("average_rating" in e for e in result.errors)

    def test_invalid_rating_fails_validation(self, dataset: GeneratedDataset) -> None:
        bad_sellers = dataset.sellers.copy()
        bad_sellers.loc[bad_sellers.index[0], "average_rating"] = 6.0
        result = validate_dataset(
            users=dataset.users,
            sellers=bad_sellers,
            listings=dataset.listings,
            transactions=dataset.transactions,
            reviews=dataset.reviews,
            devices=dataset.devices,
            events=dataset.events,
        )
        assert not result.passed
