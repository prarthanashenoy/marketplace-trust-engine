"""
Temporal train/validation/test split.

Why temporal splitting matters for fraud detection:
- Random splitting leaks future information into training, making models
  appear better than they are in production.
- In production a model trained on data up to date T is evaluated on
  data after T. Temporal splitting replicates this setup.
- Fraud patterns evolve over time; temporal splitting forces the model
  to generalise to unseen time periods, not just unseen random samples.

The split is based on the seller's first transaction timestamp.
Sellers with no transactions fall into the training set by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from marketplace_trust.config import TemporalSplitConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Holds train/val/test indices aligned to a feature matrix."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    seller_ids_train: pd.Series
    seller_ids_val: pd.Series
    seller_ids_test: pd.Series

    @property
    def train_fraud_rate(self) -> float:
        return float(self.y_train.mean())

    @property
    def val_fraud_rate(self) -> float:
        return float(self.y_val.mean())

    @property
    def test_fraud_rate(self) -> float:
        return float(self.y_test.mean())


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    seller_ids: pd.Series,
    transactions: pd.DataFrame,
    config: TemporalSplitConfig,
) -> DataSplit:
    """
    Split data into train/val/test sets using temporal ordering.

    Sellers are ordered by their earliest transaction timestamp.
    The earliest 70% go to train, next 15% to val, final 15% to test.
    Sellers with no transactions are assigned to train.

    Parameters
    ----------
    X:            Feature matrix (one row per seller).
    y:            Fraud labels.
    seller_ids:   Seller IDs aligned to X rows.
    transactions: Raw transactions table used to derive timestamps.
    config:       TemporalSplitConfig with train/val/test fractions.
    """
    # Earliest transaction per seller
    txn_ts = pd.to_datetime(transactions["timestamp"])
    first_txn = (
        transactions.assign(timestamp=txn_ts)
        .groupby("seller_id")["timestamp"]
        .min()
        .reset_index()
        .rename(columns={"timestamp": "first_transaction_at"})
    )

    # Build ordering frame
    order_df = pd.DataFrame({"seller_id": seller_ids.values}).merge(
        first_txn, on="seller_id", how="left"
    )

    # Sellers with no transactions get the earliest possible date so they
    # end up in the training set
    order_df["first_transaction_at"] = order_df["first_transaction_at"].fillna(txn_ts.min())
    order_df = order_df.sort_values("first_transaction_at").reset_index(drop=True)

    n = len(order_df)
    train_end = int(n * config.train_fraction)
    val_end = train_end + int(n * config.val_fraction)

    train_sellers = set(order_df.iloc[:train_end]["seller_id"])
    val_sellers = set(order_df.iloc[train_end:val_end]["seller_id"])
    test_sellers = set(order_df.iloc[val_end:]["seller_id"])

    idx = seller_ids.reset_index(drop=True)
    train_mask = idx.isin(train_sellers)
    val_mask = idx.isin(val_sellers)
    test_mask = idx.isin(test_sellers)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    split = DataSplit(
        X_train=X[train_mask].reset_index(drop=True),
        X_val=X[val_mask].reset_index(drop=True),
        X_test=X[test_mask].reset_index(drop=True),
        y_train=y[train_mask].reset_index(drop=True),
        y_val=y[val_mask].reset_index(drop=True),
        y_test=y[test_mask].reset_index(drop=True),
        seller_ids_train=idx[train_mask].reset_index(drop=True),
        seller_ids_val=idx[val_mask].reset_index(drop=True),
        seller_ids_test=idx[test_mask].reset_index(drop=True),
    )

    logger.info(
        "Temporal split — train: %d (%.1f%% fraud) | val: %d (%.1f%% fraud) | test: %d (%.1f%% fraud)",
        len(split.X_train),
        split.train_fraud_rate * 100,
        len(split.X_val),
        split.val_fraud_rate * 100,
        len(split.X_test),
        split.test_fraud_rate * 100,
    )
    return split
