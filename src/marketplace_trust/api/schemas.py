"""
Pydantic schemas for the inference API.

Two concerns are kept separate:
- PredictRequest: what the caller must supply (raw seller behaviour signals).
  These match the engineered features directly so no re-computation is
  required at inference time.  All fields are optional with sensible
  defaults so partial payloads are accepted gracefully.
- PredictResponse: what the API returns — risk score, level, component
  scores, and human-readable risk factors.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class SellerFeatures(BaseModel):
    """
    Raw seller behaviour features for a single seller.

    Field names match the feature columns produced by the feature pipeline.
    All fields are optional; missing values default to 0 (no activity).
    This allows the API to score new sellers with limited history.
    """

    # --- Seller profile ---
    seller_age_days: float = Field(
        default=0, ge=0, description="Days since seller account was created"
    )
    total_listings: float = Field(default=0, ge=0, description="Total number of listings created")
    total_transactions: float = Field(
        default=0, ge=0, description="Total completed + cancelled transactions"
    )
    total_reviews: float = Field(default=0, ge=0, description="Total reviews received")
    avg_rating: float = Field(default=4.0, ge=1.0, le=5.0, description="Average review rating")
    std_rating: float = Field(
        default=0.0, ge=0.0, description="Standard deviation of review ratings"
    )
    min_rating: float = Field(default=4.0, ge=1.0, le=5.0, description="Lowest received rating")
    five_star_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of reviews rated ≥4.9"
    )
    reviewer_concentration: float = Field(
        default=1.0, ge=0.0, description="Avg reviews per unique reviewer"
    )
    avg_review_text_length: float = Field(
        default=100.0, ge=0.0, description="Average review text length in characters"
    )
    listings_per_day: float = Field(
        default=0.0, ge=0.0, description="Listings created per day of seller age"
    )
    transactions_per_listing: float = Field(
        default=0.0, ge=0.0, description="Transactions per listing"
    )
    review_coverage_rate: float = Field(default=0.0, ge=0.0, description="Reviews per transaction")

    # --- Price signals ---
    avg_price_to_market_ratio: float = Field(
        default=0.9, ge=0.0, description="Average price as fraction of market price"
    )
    min_price_to_market_ratio: float = Field(
        default=0.9, ge=0.0, description="Minimum price as fraction of market price"
    )
    std_price_to_market_ratio: float = Field(
        default=0.0, ge=0.0, description="Std dev of price-to-market ratio"
    )
    listings_below_50pct_market: float = Field(
        default=0.0, ge=0.0, description="Count of listings priced below 50% of market"
    )
    scam_price_rate: float = Field(
        default=0.0, ge=0.0, description="Rate of listings priced below 50% of market"
    )
    avg_description_length: float = Field(
        default=200.0, ge=0.0, description="Average listing description length"
    )
    avg_image_count: float = Field(default=3.0, ge=0.0, description="Average images per listing")
    min_image_count: float = Field(
        default=1.0, ge=0.0, description="Minimum images across listings"
    )
    removed_listing_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of listings that were removed"
    )

    # --- Transaction signals ---
    avg_transaction_amount: float = Field(
        default=50.0, ge=0.0, description="Average transaction value (USD)"
    )
    std_transaction_amount: float = Field(
        default=0.0, ge=0.0, description="Std dev of transaction values"
    )
    max_transaction_amount: float = Field(
        default=50.0, ge=0.0, description="Largest single transaction"
    )
    cancel_to_transaction_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Cancellations / total transactions"
    )
    refund_to_transaction_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Refunds / total transactions"
    )
    dispute_to_transaction_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Disputes / total transactions"
    )
    crypto_wallet_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction paid via crypto or wallet"
    )
    buyer_concentration: float = Field(
        default=1.0, ge=0.0, description="Transactions per unique buyer"
    )
    unique_buyers: float = Field(default=0.0, ge=0.0, description="Number of distinct buyers")

    # --- Rolling window (temporal) ---
    txn_count_1d: float = Field(default=0.0, ge=0.0, description="Transactions in the last 1 day")
    txn_count_7d: float = Field(default=0.0, ge=0.0, description="Transactions in the last 7 days")
    txn_count_30d: float = Field(
        default=0.0, ge=0.0, description="Transactions in the last 30 days"
    )
    txn_amount_sum_30d: float = Field(
        default=0.0, ge=0.0, description="Total transaction value in last 30 days"
    )
    txn_amount_mean_1d: float = Field(default=0.0, ge=0.0)
    txn_amount_mean_7d: float = Field(default=0.0, ge=0.0)
    txn_amount_mean_30d: float = Field(default=0.0, ge=0.0)
    unique_buyers_1d: float = Field(default=0.0, ge=0.0)
    unique_buyers_7d: float = Field(default=0.0, ge=0.0)
    unique_buyers_30d: float = Field(default=0.0, ge=0.0)
    cancel_count_1d: float = Field(default=0.0, ge=0.0)
    cancel_count_7d: float = Field(default=0.0, ge=0.0)
    cancel_count_30d: float = Field(default=0.0, ge=0.0)
    refund_count_1d: float = Field(default=0.0, ge=0.0)
    refund_count_7d: float = Field(default=0.0, ge=0.0)
    refund_count_30d: float = Field(default=0.0, ge=0.0)
    txn_velocity_30d: float = Field(
        default=0.0, ge=0.0, description="Transactions per day (30-day window)"
    )
    txn_burst_ratio_1d_7d: float = Field(
        default=0.0, ge=0.0, description="Ratio of 1-day to 7-day transaction count"
    )
    days_since_last_transaction: float = Field(
        default=999.0, ge=0.0, description="Days elapsed since most recent transaction"
    )
    payment_method_diversity_30d: float = Field(
        default=1.0, ge=0.0, description="Distinct payment methods used in last 30 days"
    )
    crypto_wallet_count_30d: float = Field(
        default=0.0, ge=0.0, description="Crypto/wallet transactions in last 30 days"
    )
    lifetime_txn_velocity: float = Field(default=0.0, ge=0.0)
    seller_age_days_velocity: float = Field(
        default=0.0, ge=0.0, description="Alias field for velocity calc", alias="seller_age_days"
    )

    # --- Behavioral ---
    num_event_devices: float = Field(
        default=1.0, ge=0.0, description="Distinct devices in event log"
    )
    num_event_ips: float = Field(default=1.0, ge=0.0, description="Distinct IP hashes in event log")
    device_changed: float = Field(
        default=0.0, ge=0.0, le=1.0, description="1 if seller used more than one device"
    )
    avg_review_lag_hours: float = Field(
        default=48.0, ge=0.0, description="Average hours between purchase and review"
    )
    review_lag_std_hours: float = Field(
        default=0.0, ge=0.0, description="Std dev of review lag in hours"
    )
    reviews_within_1h_count: float = Field(
        default=0.0, ge=0.0, description="Reviews posted within 1 hour of transaction"
    )
    login_count: float = Field(default=0.0, ge=0.0)
    total_event_count: float = Field(default=0.0, ge=0.0)
    event_type_diversity: float = Field(default=0.0, ge=0.0)
    profile_update_count: float = Field(default=0.0, ge=0.0)

    # --- Graph features ---
    sellers_sharing_device: float = Field(
        default=1.0, ge=0.0, description="Seller accounts sharing the same device"
    )
    distinct_event_devices: float = Field(default=1.0, ge=0.0)
    distinct_ips: float = Field(
        default=1.0, ge=0.0, description="Distinct IPs associated with this seller"
    )
    max_accounts_per_ip: float = Field(
        default=1.0, ge=0.0, description="Max accounts on any single IP this seller used"
    )
    seller_buyer_overlap_count: float = Field(default=0.0, ge=0.0)
    avg_buyer_seller_count: float = Field(default=1.0, ge=0.0)
    shared_reviewer_count: float = Field(default=0.0, ge=0.0)
    avg_reviewer_seller_count: float = Field(default=1.0, ge=0.0)
    seller_degree_centrality: float = Field(default=0.0, ge=0.0, le=1.0)
    removed_listing_count: float = Field(default=0.0, ge=0.0)
    five_star_count: float = Field(default=0.0, ge=0.0)
    unique_reviewers: float = Field(default=0.0, ge=0.0)
    crypto_wallet_count: float = Field(default=0.0, ge=0.0)

    model_config = {"populate_by_name": True}


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    seller_id: str = Field(description="Unique seller identifier")
    features: SellerFeatures = Field(description="Seller behaviour feature vector")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    seller_id: str
    risk_score: Annotated[int, Field(ge=0, le=100, description="Overall risk score 0–100")]
    risk_level: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    model_probability: Annotated[
        float, Field(ge=0.0, le=1.0, description="XGBoost fraud probability")
    ]
    anomaly_score: Annotated[
        float, Field(ge=0.0, le=1.0, description="Isolation Forest anomaly score")
    ]
    behavioral_risk: Annotated[
        float, Field(ge=0.0, le=1.0, description="Rule-based behavioral risk signal")
    ]
    top_risk_factors: list[str] = Field(description="Human-readable risk factor descriptions")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    n_features: int
    feature_names: list[str]
    risk_levels: dict[str, str]
    weights: dict[str, float]
