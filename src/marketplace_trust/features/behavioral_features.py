"""
Behavioral feature engineering from event and review streams.

Captures signals that are harder to fake: login patterns, review timing,
device/IP switching behaviour, and session-level activity.

Why these features matter:
- Account takeovers show sudden device/country changes after stable history.
- Fake review networks show unusually rapid, clustered review submissions.
- Fraud rings show correlated IP and device usage across multiple accounts.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_behavioral_features(
    sellers: pd.DataFrame,
    users: pd.DataFrame,
    events: pd.DataFrame,
    reviews: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-seller behavioral features from event and review logs.

    Parameters
    ----------
    sellers:      Sellers table (seller_id, user_id).
    users:        Users table (user_id, device_id, country).
    events:       Behavioral events (user_id, timestamp, device_id, ip_hash, ...).
    reviews:      Reviews (seller_id, buyer_id, timestamp, rating).
    transactions: Transactions (seller_id, buyer_id, timestamp).

    Returns
    -------
    DataFrame with one row per seller and behavioral feature columns.
    """
    logger.info("Building behavioral features")

    device_feats = _device_change_features(sellers, users, events)
    review_timing_feats = _review_timing_features(reviews, transactions)
    event_feats = _event_activity_features(sellers, events)

    feats = sellers[["seller_id"]].copy()
    feats = feats.merge(device_feats, on="seller_id", how="left")
    feats = feats.merge(review_timing_feats, on="seller_id", how="left")
    feats = feats.merge(event_feats, on="seller_id", how="left")

    numeric_cols = feats.select_dtypes(include="number").columns
    feats[numeric_cols] = feats[numeric_cols].fillna(0)

    logger.info("Behavioral features shape: %s", feats.shape)
    return feats


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _device_change_features(
    sellers: pd.DataFrame,
    users: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect device and country switching relative to a user's baseline.

    Account takeover signals:
    - num_distinct_devices: many devices used over account lifetime.
    - num_distinct_countries: activity from many different countries.
    - device_changed: device used in events differs from registration device.
    """
    # Join sellers → users to get baseline device/country
    seller_user = sellers[["seller_id", "user_id"]].merge(
        users[["user_id", "device_id", "country"]], on="user_id", how="left"
    )

    # Join user events: get distinct devices and IPs per user
    if len(events) == 0:
        seller_user["num_event_devices"] = 0
        seller_user["num_event_countries"] = 0
        seller_user["num_event_ips"] = 0
        seller_user["device_changed"] = False
        return seller_user[
            [
                "seller_id",
                "num_event_devices",
                "num_event_countries",
                "num_event_ips",
                "device_changed",
            ]
        ]

    events_agg = (
        events.groupby("user_id")
        .agg(
            num_event_devices=("device_id", "nunique"),
            num_event_ips=("ip_hash", "nunique"),
        )
        .reset_index()
    )

    # Country is encoded in the device table; use ip_hash diversity as proxy
    result = seller_user.merge(events_agg, on="user_id", how="left")
    result["num_event_devices"] = result["num_event_devices"].fillna(0)
    result["num_event_ips"] = result["num_event_ips"].fillna(0)

    # device_changed: True if the seller used >1 distinct device in events
    result["device_changed"] = result["num_event_devices"] > 1

    return result[["seller_id", "num_event_devices", "num_event_ips", "device_changed"]]


def _review_timing_features(
    reviews: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Capture suspicious review timing patterns.

    Fake review signals:
    - avg_review_lag_hours: time between transaction and review.
      Very short lags (< 1 hour) are suspicious.
    - review_lag_std_hours: unusually low variance in review timing.
    - reviews_within_1h: count of reviews posted within 1 hour of transaction.
    """
    if len(reviews) == 0 or len(transactions) == 0:
        return pd.DataFrame(
            columns=[
                "seller_id",
                "avg_review_lag_hours",
                "review_lag_std_hours",
                "reviews_within_1h_count",
            ]
        )

    rev = reviews.copy()
    txn = transactions[["transaction_id", "timestamp"]].copy()
    rev["review_ts"] = pd.to_datetime(rev["timestamp"])
    txn["txn_ts"] = pd.to_datetime(txn["timestamp"])

    merged = rev.merge(txn, on="transaction_id", how="left")
    merged["review_lag_hours"] = (merged["review_ts"] - merged["txn_ts"]).dt.total_seconds() / 3600
    # Negative lags (review before transaction) are data noise; clip to 0
    merged["review_lag_hours"] = merged["review_lag_hours"].clip(lower=0)

    agg = (
        merged.groupby("seller_id")
        .agg(
            avg_review_lag_hours=("review_lag_hours", "mean"),
            review_lag_std_hours=("review_lag_hours", "std"),
            reviews_within_1h_count=("review_lag_hours", lambda x: (x < 1.0).sum()),
        )
        .reset_index()
    )
    agg["review_lag_std_hours"] = agg["review_lag_std_hours"].fillna(0)
    return agg


def _event_activity_features(
    sellers: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate event-stream features per seller (via their user_id).

    Covers: login frequency, event diversity, session-level activity.
    """
    if len(events) == 0:
        return sellers[["seller_id"]].assign(
            login_count=0,
            total_event_count=0,
            event_type_diversity=0,
            profile_update_count=0,
        )

    seller_user = sellers[["seller_id", "user_id"]]

    event_agg = (
        events.groupby("user_id")
        .agg(
            login_count=("event_type", lambda x: (x == "login").sum()),
            total_event_count=("event_id", "count"),
            event_type_diversity=("event_type", "nunique"),
            profile_update_count=("event_type", lambda x: (x == "profile_update").sum()),
        )
        .reset_index()
    )

    result = seller_user.merge(event_agg, on="user_id", how="left")
    return result[
        [
            "seller_id",
            "login_count",
            "total_event_count",
            "event_type_diversity",
            "profile_update_count",
        ]
    ]
