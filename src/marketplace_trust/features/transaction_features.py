"""
Transaction-level temporal feature engineering.

Computes rolling-window velocity features over time, which are critical
for detecting sudden behavioural changes such as account takeovers and
fraud rings that burst into activity.

Key design decision: features are computed per-seller using time-based
rolling windows anchored on each seller's most-recent observation date.
This avoids look-ahead bias — we only use information available at the
time of the 'snapshot'.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Rolling window sizes in days
WINDOWS = [1, 7, 30]


def build_transaction_features(
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build per-seller temporal transaction features.

    For each seller, computes transaction counts and amounts over
    1-day, 7-day, and 30-day rolling windows ending at snapshot_date.

    Parameters
    ----------
    transactions:  Raw transactions table with timestamp and seller_id.
    snapshot_date: The 'as-of' date. Defaults to max timestamp in data.

    Returns
    -------
    DataFrame with one row per seller and rolling-window feature columns.
    """
    logger.info("Building transaction temporal features")

    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if snapshot_date is None:
        snapshot_date = df["timestamp"].max()

    feature_rows = []
    for seller_id, grp in df.groupby("seller_id"):
        row: dict[str, object] = {"seller_id": seller_id}
        grp = grp.sort_values("timestamp")

        for days in WINDOWS:
            cutoff = snapshot_date - pd.Timedelta(days=days)
            window = grp[grp["timestamp"] >= cutoff]

            row[f"txn_count_{days}d"] = len(window)
            row[f"txn_amount_sum_{days}d"] = float(window["amount"].sum())
            row[f"txn_amount_mean_{days}d"] = float(
                window["amount"].mean() if len(window) > 0 else 0.0
            )
            row[f"unique_buyers_{days}d"] = int(window["buyer_id"].nunique())
            row[f"cancel_count_{days}d"] = int(window["cancellation_flag"].sum())
            row[f"refund_count_{days}d"] = int(window["refund_flag"].sum())

        # Transaction velocity: transactions per day over 30-day window
        row["txn_velocity_30d"] = float(row["txn_count_30d"]) / 30.0  # type: ignore[arg-type]

        # Burst ratio: ratio of last 1-day activity to 7-day activity
        # High burst ratio indicates sudden activity spike
        row["txn_burst_ratio_1d_7d"] = float(row["txn_count_1d"]) / max(  # type: ignore[arg-type]
            float(row["txn_count_7d"]),
            1.0,  # type: ignore[arg-type]
        )

        # Recency of last transaction in days before snapshot
        if len(grp) > 0:
            last_txn = grp["timestamp"].max()
            row["days_since_last_transaction"] = float(
                (snapshot_date - last_txn).total_seconds() / 86400
            )
        else:
            row["days_since_last_transaction"] = float(
                (snapshot_date - df["timestamp"].min()).total_seconds() / 86400
            )

        # Payment method diversity in last 30 days
        w30 = grp[grp["timestamp"] >= snapshot_date - pd.Timedelta(days=30)]
        row["payment_method_diversity_30d"] = int(w30["payment_method"].nunique())
        row["crypto_wallet_count_30d"] = int(w30["payment_method"].isin(["crypto", "wallet"]).sum())

        feature_rows.append(row)

    result = pd.DataFrame(feature_rows)
    logger.info("Transaction temporal features shape: %s", result.shape)
    return result


def build_velocity_features(
    transactions: pd.DataFrame,
    sellers: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compute normalised velocity features: activity relative to seller age.

    Returns a DataFrame with seller_id + normalised velocity columns.
    These capture whether a seller is abnormally active *given* their age.
    """
    txn_feats = build_transaction_features(transactions, snapshot_date)

    age_map = sellers.set_index("seller_id")["seller_created_at"].pipe(pd.to_datetime)
    if snapshot_date is None:
        snapshot_date = pd.to_datetime(transactions["timestamp"]).max()

    txn_feats["seller_age_days"] = (
        txn_feats["seller_id"]
        .map(age_map)
        .pipe(lambda s: (snapshot_date - pd.to_datetime(s)).dt.days)
        .clip(lower=1)
    )

    # Normalised velocity: total transactions divided by seller age
    txn_feats["lifetime_txn_velocity"] = txn_feats["txn_count_30d"] / txn_feats["seller_age_days"]

    return txn_feats
