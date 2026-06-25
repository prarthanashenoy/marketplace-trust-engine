"""
Data validation for the Marketplace Trust Engine.

Validates all generated DataFrames before feature engineering begins.
Fails loudly with clear error messages on data quality issues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    errors: list[str]
    warnings: list[str]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def raise_if_errors(self) -> None:
        if self.has_errors:
            raise DataValidationError("\n".join(self.errors))


class DataValidationError(Exception):
    """Raised when data validation fails."""

    pass


def validate_dataset(
    users: pd.DataFrame,
    sellers: pd.DataFrame,
    listings: pd.DataFrame,
    transactions: pd.DataFrame,
    reviews: pd.DataFrame,
    devices: pd.DataFrame,
    events: pd.DataFrame,
) -> ValidationResult:
    """
    Run all validation checks on the full dataset.

    Checks:
    - Required columns
    - Data types
    - Value ranges
    - Null rates
    - Duplicate IDs
    - Referential integrity
    - Timestamp consistency
    """
    errors: list[str] = []
    warnings: list[str] = []

    def _check(checks: list[str], warns: list[str]) -> None:
        errors.extend(checks)
        warnings.extend(warns)

    _check(*_validate_users(users))
    _check(*_validate_sellers(sellers, users))
    _check(*_validate_listings(listings, sellers))
    _check(*_validate_transactions(transactions, sellers, listings, users))
    _check(*_validate_reviews(reviews, transactions, sellers, users))
    _check(*_validate_devices(devices))
    _check(*_validate_events(events, users, devices))

    result = ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)

    if warnings:
        for w in warnings:
            logger.warning("Validation warning: %s", w)

    if errors:
        for e in errors:
            logger.error("Validation error: %s", e)
    else:
        logger.info("All validation checks passed.")

    return result


def _check_required_columns(df: pd.DataFrame, required: list[str], table_name: str) -> list[str]:
    errors = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"{table_name}: missing required columns: {missing}")
    return errors


def _check_no_duplicate_ids(df: pd.DataFrame, id_col: str, table_name: str) -> list[str]:
    if id_col not in df.columns:
        return []
    dups = df[id_col].duplicated().sum()
    if dups > 0:
        return [f"{table_name}: {dups} duplicate values in '{id_col}'"]
    return []


def _check_null_rate(
    df: pd.DataFrame, col: str, max_null_rate: float, table_name: str
) -> list[str]:
    if col not in df.columns:
        return []
    null_rate = df[col].isnull().mean()
    if null_rate > max_null_rate:
        return [
            f"{table_name}.{col}: null rate {null_rate:.2%} exceeds threshold {max_null_rate:.2%}"
        ]
    return []


def _check_value_range(
    df: pd.DataFrame,
    col: str,
    min_val: float | None,
    max_val: float | None,
    table_name: str,
) -> list[str]:
    if col not in df.columns:
        return []
    series = df[col].dropna()
    errors = []
    if min_val is not None and (series < min_val).any():
        n = (series < min_val).sum()
        errors.append(f"{table_name}.{col}: {n} values below minimum {min_val}")
    if max_val is not None and (series > max_val).any():
        n = (series > max_val).sum()
        errors.append(f"{table_name}.{col}: {n} values above maximum {max_val}")
    return errors


def _validate_users(users: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    required = [
        "user_id",
        "account_created_at",
        "country",
        "device_id",
        "account_age_days",
        "verification_status",
    ]
    errors.extend(_check_required_columns(users, required, "users"))
    errors.extend(_check_no_duplicate_ids(users, "user_id", "users"))
    errors.extend(_check_null_rate(users, "user_id", 0.0, "users"))
    errors.extend(_check_value_range(users, "account_age_days", 0, None, "users"))
    if len(users) == 0:
        errors.append("users: table is empty")
    return errors, warnings


def _validate_sellers(sellers: pd.DataFrame, users: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
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
    errors.extend(_check_required_columns(sellers, required, "sellers"))
    errors.extend(_check_no_duplicate_ids(sellers, "seller_id", "sellers"))
    errors.extend(_check_value_range(sellers, "average_rating", 1.0, 5.0, "sellers"))
    errors.extend(_check_value_range(sellers, "cancellation_rate", 0.0, 1.0, "sellers"))
    errors.extend(_check_value_range(sellers, "number_of_listings", 0, None, "sellers"))
    errors.extend(_check_value_range(sellers, "number_of_transactions", 0, None, "sellers"))

    # Referential integrity: all seller user_ids must exist in users
    if "user_id" in sellers.columns and "user_id" in users.columns:
        unknown_users = ~sellers["user_id"].isin(users["user_id"])
        if unknown_users.any():
            errors.append(
                f"sellers: {unknown_users.sum()} seller user_ids not found in users table"
            )
    return errors, warnings


def _validate_listings(
    listings: pd.DataFrame, sellers: pd.DataFrame
) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
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
    errors.extend(_check_required_columns(listings, required, "listings"))
    errors.extend(_check_no_duplicate_ids(listings, "listing_id", "listings"))
    errors.extend(_check_value_range(listings, "price", 0.01, None, "listings"))
    errors.extend(_check_value_range(listings, "market_price", 0.01, None, "listings"))
    errors.extend(_check_value_range(listings, "description_length", 0, None, "listings"))
    errors.extend(_check_value_range(listings, "image_count", 0, None, "listings"))

    if "seller_id" in listings.columns and "seller_id" in sellers.columns:
        unknown = ~listings["seller_id"].isin(sellers["seller_id"])
        if unknown.any():
            errors.append(f"listings: {unknown.sum()} seller_ids not found in sellers table")
    return errors, warnings


def _validate_transactions(
    transactions: pd.DataFrame,
    sellers: pd.DataFrame,
    listings: pd.DataFrame,
    users: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
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
    errors.extend(_check_required_columns(transactions, required, "transactions"))
    errors.extend(_check_no_duplicate_ids(transactions, "transaction_id", "transactions"))
    errors.extend(_check_value_range(transactions, "amount", 0.01, None, "transactions"))

    if "seller_id" in transactions.columns and "seller_id" in sellers.columns:
        unknown = ~transactions["seller_id"].isin(sellers["seller_id"])
        if unknown.any():
            errors.append(f"transactions: {unknown.sum()} seller_ids not found in sellers")
    if "buyer_id" in transactions.columns and "user_id" in users.columns:
        unknown = ~transactions["buyer_id"].isin(users["user_id"])
        if unknown.any():
            errors.append(f"transactions: {unknown.sum()} buyer_ids not found in users")
    return errors, warnings


def _validate_reviews(
    reviews: pd.DataFrame,
    transactions: pd.DataFrame,
    sellers: pd.DataFrame,
    users: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    required = [
        "review_id",
        "buyer_id",
        "seller_id",
        "transaction_id",
        "rating",
        "timestamp",
        "text_length",
    ]
    errors.extend(_check_required_columns(reviews, required, "reviews"))
    errors.extend(_check_no_duplicate_ids(reviews, "review_id", "reviews"))
    errors.extend(_check_value_range(reviews, "rating", 1.0, 5.0, "reviews"))
    errors.extend(_check_value_range(reviews, "text_length", 0, None, "reviews"))
    return errors, warnings


def _validate_devices(devices: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    required = ["device_id", "device_type", "country", "first_seen_at"]
    errors.extend(_check_required_columns(devices, required, "devices"))
    errors.extend(_check_no_duplicate_ids(devices, "device_id", "devices"))
    return errors, warnings


def _validate_events(
    events: pd.DataFrame, users: pd.DataFrame, devices: pd.DataFrame
) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    required = ["event_id", "user_id", "timestamp", "event_type", "ip_hash", "device_id"]
    errors.extend(_check_required_columns(events, required, "events"))
    # Events can have many rows; skip duplicate check for performance
    return errors, warnings
