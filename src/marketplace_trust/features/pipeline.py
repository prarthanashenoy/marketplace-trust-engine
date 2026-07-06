"""
Unified feature pipeline.

Orchestrates all feature engineering modules into a single function call
that produces the final feature matrix used for model training.

Output columns are divided into four groups (documented below) so that
models, SHAP explainability, and the dashboard can refer to them by group.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd

from marketplace_trust.features.behavioral_features import build_behavioral_features
from marketplace_trust.features.graph_features import build_graph_features
from marketplace_trust.features.seller_features import build_seller_features
from marketplace_trust.features.transaction_features import build_transaction_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature group definitions — used by explainability module
# ---------------------------------------------------------------------------

SELLER_PROFILE_FEATURES = [
    "seller_age_days",
    "total_listings",
    "total_transactions",
    "total_reviews",
    "avg_rating",
    "std_rating",
    "min_rating",
    "five_star_rate",
    "reviewer_concentration",
    "avg_review_text_length",
    "listings_per_day",
    "transactions_per_listing",
    "review_coverage_rate",
]

PRICE_FEATURES = [
    "avg_price_to_market_ratio",
    "min_price_to_market_ratio",
    "std_price_to_market_ratio",
    "listings_below_50pct_market",
    "scam_price_rate",
    "avg_description_length",
    "avg_image_count",
    "min_image_count",
    "removed_listing_rate",
]

TRANSACTION_FEATURES = [
    "avg_transaction_amount",
    "std_transaction_amount",
    "max_transaction_amount",
    "cancel_to_transaction_ratio",
    "refund_to_transaction_ratio",
    "dispute_to_transaction_ratio",
    "crypto_wallet_rate",
    "buyer_concentration",
    "unique_buyers",
    "txn_count_1d",
    "txn_count_7d",
    "txn_count_30d",
    "txn_amount_sum_30d",
    "txn_velocity_30d",
    "txn_burst_ratio_1d_7d",
    "days_since_last_transaction",
    "payment_method_diversity_30d",
    "crypto_wallet_count_30d",
]

BEHAVIORAL_FEATURES = [
    "num_event_devices",
    "num_event_ips",
    "device_changed",
    "avg_review_lag_hours",
    "review_lag_std_hours",
    "reviews_within_1h_count",
    "login_count",
    "total_event_count",
    "event_type_diversity",
    "profile_update_count",
]

GRAPH_FEATURES = [
    "sellers_sharing_device",
    "distinct_event_devices",
    "distinct_ips",
    "max_accounts_per_ip",
    "seller_buyer_overlap_count",
    "avg_buyer_seller_count",
    "shared_reviewer_count",
    "avg_reviewer_seller_count",
    "seller_degree_centrality",
]

ALL_FEATURE_GROUPS: dict[str, list[str]] = {
    "seller_profile": SELLER_PROFILE_FEATURES,
    "price": PRICE_FEATURES,
    "transaction": TRANSACTION_FEATURES,
    "behavioral": BEHAVIORAL_FEATURES,
    "graph": GRAPH_FEATURES,
}


class FeatureMatrix(NamedTuple):
    """Output of the feature pipeline."""

    X: pd.DataFrame  # Feature matrix (no labels)
    y: pd.Series  # Binary fraud label (is_fraudulent)
    seller_ids: pd.Series  # Seller IDs aligned to X rows
    feature_names: list[str]  # Ordered list of feature column names


def build_feature_matrix(
    sellers: pd.DataFrame,
    listings: pd.DataFrame,
    transactions: pd.DataFrame,
    reviews: pd.DataFrame,
    users: pd.DataFrame,
    events: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> FeatureMatrix:
    """
    Run the full feature engineering pipeline and return a clean feature matrix.

    This is the single entry point for feature engineering used by training
    scripts, the inference API, and the evaluation pipeline.

    Parameters
    ----------
    sellers, listings, transactions, reviews, users, events:
        Raw tables from the data generator or loaded from disk.
    snapshot_date:
        The 'as-of' date for rolling window calculations.
        Defaults to the latest transaction timestamp.

    Returns
    -------
    FeatureMatrix namedtuple with X, y, seller_ids, feature_names.
    """
    logger.info("Running full feature pipeline")

    if snapshot_date is None and len(transactions) > 0:
        snapshot_date = pd.to_datetime(transactions["timestamp"]).max()

    # Build each feature group
    seller_feats = build_seller_features(sellers, listings, transactions, reviews)
    txn_feats = build_transaction_features(transactions, snapshot_date)
    behavioral_feats = build_behavioral_features(sellers, users, events, reviews, transactions)
    graph_feats = build_graph_features(sellers, users, transactions, reviews, events)

    # Merge all groups on seller_id
    combined = seller_feats.copy()
    for feats_df in [txn_feats, behavioral_feats, graph_feats]:
        combined = combined.merge(feats_df, on="seller_id", how="left")

    # Merge labels (kept separate until this point to make leakage impossible)
    combined = combined.merge(sellers[["seller_id", "is_fraudulent"]], on="seller_id", how="left")

    # Drop non-numeric / metadata columns before building X
    drop_cols = [
        "seller_id",
        "seller_created_at",
        "category",
        "user_id",  # from behavioral/graph join artifacts
        "is_fraudulent",
    ]
    drop_cols = [c for c in drop_cols if c in combined.columns]

    X = combined.drop(columns=drop_cols)

    # Ensure all remaining columns are numeric; coerce anything else to NaN → 0
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Convert boolean columns to int (e.g. device_changed)
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    y = combined["is_fraudulent"].astype(int)
    seller_ids = combined["seller_id"]

    logger.info(
        "Feature matrix: %d rows × %d features | fraud rate: %.2f%%",
        len(X),
        X.shape[1],
        y.mean() * 100,
    )

    return FeatureMatrix(
        X=X,
        y=y,
        seller_ids=seller_ids,
        feature_names=list(X.columns),
    )


def get_feature_group(feature_name: str) -> str:
    """Return the group name for a given feature, or 'other' if not found."""
    for group, names in ALL_FEATURE_GROUPS.items():
        if feature_name in names:
            return group
    return "other"
