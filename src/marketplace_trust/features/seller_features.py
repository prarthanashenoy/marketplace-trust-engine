"""
Seller-level feature engineering.

Aggregates transaction, listing, and review history into per-seller
features that serve as inputs to the fraud detection models.

All features are computed from raw tables — no labels are used here,
which prevents target leakage.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_seller_features(
    sellers: pd.DataFrame,
    listings: pd.DataFrame,
    transactions: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a feature table with one row per seller.

    Parameters
    ----------
    sellers:      Raw sellers table (seller_id, seller_created_at, category, ...)
    listings:     Raw listings table
    transactions: Raw transactions table
    reviews:      Raw reviews table

    Returns
    -------
    DataFrame indexed by seller_id containing all seller-level features.
    Does NOT include the fraud label columns (fraud_scenario, is_fraudulent).
    """
    logger.info("Building seller features for %d sellers", len(sellers))

    feats = sellers[["seller_id", "seller_created_at", "category"]].copy()

    # Compute seller age relative to the last observed date in the dataset
    reference_date = pd.to_datetime(sellers["seller_created_at"]).max()
    feats["seller_age_days"] = (
        reference_date - pd.to_datetime(feats["seller_created_at"])
    ).dt.days.clip(lower=0)

    feats = feats.merge(_listing_features(listings), on="seller_id", how="left")
    feats = feats.merge(_transaction_features(transactions), on="seller_id", how="left")
    feats = feats.merge(_review_features(reviews), on="seller_id", how="left")
    feats = feats.merge(_price_features(listings), on="seller_id", how="left")

    # Fill NaN for sellers with no activity (new accounts with zero history)
    numeric_cols = feats.select_dtypes(include="number").columns
    feats[numeric_cols] = feats[numeric_cols].fillna(0)

    # Derived ratio features — safe to compute now that NaN are filled
    feats["cancel_to_transaction_ratio"] = feats["cancelled_transaction_count"] / feats[
        "total_transactions"
    ].clip(lower=1)
    feats["refund_to_transaction_ratio"] = feats["refunded_transaction_count"] / feats[
        "total_transactions"
    ].clip(lower=1)
    feats["dispute_to_transaction_ratio"] = feats["disputed_transaction_count"] / feats[
        "total_transactions"
    ].clip(lower=1)
    feats["review_coverage_rate"] = feats["total_reviews"] / feats["total_transactions"].clip(
        lower=1
    )
    feats["listings_per_day"] = feats["total_listings"] / feats["seller_age_days"].clip(lower=1)
    feats["transactions_per_listing"] = feats["total_transactions"] / feats["total_listings"].clip(
        lower=1
    )

    logger.info("Seller features shape: %s", feats.shape)
    return feats


# ---------------------------------------------------------------------------
# Private helpers — each returns a DataFrame with seller_id + feature columns
# ---------------------------------------------------------------------------


def _listing_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Per-seller listing statistics."""
    agg = (
        listings.groupby("seller_id")
        .agg(
            total_listings=("listing_id", "count"),
            avg_description_length=("description_length", "mean"),
            avg_image_count=("image_count", "mean"),
            min_image_count=("image_count", "min"),
            removed_listing_count=("listing_status", lambda x: (x == "removed").sum()),
        )
        .reset_index()
    )
    agg["removed_listing_rate"] = agg["removed_listing_count"] / agg["total_listings"].clip(lower=1)
    return agg


def _transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Per-seller transaction statistics."""
    agg = (
        transactions.groupby("seller_id")
        .agg(
            total_transactions=("transaction_id", "count"),
            avg_transaction_amount=("amount", "mean"),
            std_transaction_amount=("amount", "std"),
            max_transaction_amount=("amount", "max"),
            cancelled_transaction_count=("cancellation_flag", "sum"),
            refunded_transaction_count=("refund_flag", "sum"),
            disputed_transaction_count=("status", lambda x: (x == "disputed").sum()),
            unique_buyers=("buyer_id", "nunique"),
            crypto_wallet_count=(
                "payment_method",
                lambda x: x.isin(["crypto", "wallet"]).sum(),
            ),
        )
        .reset_index()
    )
    agg["std_transaction_amount"] = agg["std_transaction_amount"].fillna(0)
    agg["crypto_wallet_rate"] = agg["crypto_wallet_count"] / agg["total_transactions"].clip(lower=1)
    # High buyer concentration (few unique buyers relative to transactions) is a
    # signal of coordinated/shill purchasing.
    agg["buyer_concentration"] = agg["total_transactions"] / agg["unique_buyers"].clip(lower=1)
    return agg


def _review_features(reviews: pd.DataFrame) -> pd.DataFrame:
    """Per-seller review statistics."""
    agg = (
        reviews.groupby("seller_id")
        .agg(
            total_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            std_rating=("rating", "std"),
            min_rating=("rating", "min"),
            avg_review_text_length=("text_length", "mean"),
            five_star_count=("rating", lambda x: (x >= 4.9).sum()),
            unique_reviewers=("buyer_id", "nunique"),
        )
        .reset_index()
    )
    agg["std_rating"] = agg["std_rating"].fillna(0)
    agg["five_star_rate"] = agg["five_star_count"] / agg["total_reviews"].clip(lower=1)
    # Many reviews from the same buyer = review manipulation signal
    agg["reviewer_concentration"] = agg["total_reviews"] / agg["unique_reviewers"].clip(lower=1)
    return agg


def _price_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Per-seller price-to-market statistics."""
    df = listings.copy()
    df["price_to_market_ratio"] = df["price"] / df["market_price"].clip(lower=0.01)

    agg = (
        df.groupby("seller_id")
        .agg(
            avg_price_to_market_ratio=("price_to_market_ratio", "mean"),
            min_price_to_market_ratio=("price_to_market_ratio", "min"),
            std_price_to_market_ratio=("price_to_market_ratio", "std"),
            listings_below_50pct_market=("price_to_market_ratio", lambda x: (x < 0.5).sum()),
        )
        .reset_index()
    )
    agg["std_price_to_market_ratio"] = agg["std_price_to_market_ratio"].fillna(0)
    agg["scam_price_rate"] = (
        agg["listings_below_50pct_market"] / agg["total_listings"]
        if "total_listings" in agg.columns
        else agg["listings_below_50pct_market"].clip(upper=1)
    )
    # Simpler: fraction of this seller's listings priced below 50% of market
    agg["scam_price_rate"] = agg["listings_below_50pct_market"] / agg.groupby("seller_id")[
        "listings_below_50pct_market"
    ].transform("count").clip(lower=1)
    return agg
