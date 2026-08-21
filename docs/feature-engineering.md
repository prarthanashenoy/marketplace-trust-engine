# Feature Engineering

## Overview

The feature pipeline transforms seven raw marketplace tables into a flat seller-level feature matrix. All features are computed **before** fraud labels are merged into the matrix, preventing any form of target leakage.

The pipeline produces **50+ features** grouped into five categories:

| Group | Features | Source tables |
|-------|----------|--------------|
| Seller Profile | 13 | sellers, listings, reviews, transactions |
| Price Signals | 9 | listings |
| Transaction Signals | 18 | transactions |
| Behavioural Signals | 10 | devices, events, reviews |
| Graph / Network | 9 | devices, sellers |

## Group 1: Seller Profile (13 features)

These features describe basic account health and activity patterns.

| Feature | Description |
|---------|-------------|
| `seller_age_days` | Days since account creation |
| `total_listings` | Total number of listings posted |
| `active_listings` | Listings with `status == "active"` |
| `avg_rating` | Mean rating across all reviews |
| `rating_count` | Total number of ratings received |
| `rating_std` | Standard deviation of ratings (high std → inconsistent reputation) |
| `unique_reviewers` | Count of distinct reviewer user IDs |
| `reviewer_concentration` | `rating_count / unique_reviewers` — high value → reviews from a small pool |
| `listings_per_day` | `total_listings / max(seller_age_days, 1)` — listing velocity |
| `review_coverage` | `rating_count / max(total_transactions, 1)` — fraction of transactions that received a review |
| `verified_seller` | Binary flag from the sellers table |
| `country_risk_score` | Heuristic country-level risk bucket (0 / 0.3 / 0.7 / 1.0) |
| `days_since_last_listing` | Recency of last listing — inactivity signal |

## Group 2: Price Signals (9 features)

These features compare a seller's pricing behaviour against the simulated market price.

| Feature | Description |
|---------|-------------|
| `avg_price_to_market_ratio` | `mean(price / market_price)` across listings |
| `min_price_to_market_ratio` | `min(price / market_price)` — catches extreme underpricing |
| `price_std` | Standard deviation of absolute prices |
| `price_to_market_std` | Standard deviation of `price / market_price` ratios |
| `fraction_below_50pct_market` | Proportion of listings priced below 50 % of market price |
| `scam_price_rate` | Proportion of listings priced below 30 % of market price |
| `avg_image_count` | Mean images per listing (low → low-effort scam listings) |
| `min_image_count` | Minimum images per listing |
| `avg_description_length` | Mean character length of listing descriptions |

**Design note:** Market price is a synthetic baseline in the listings table. Real deployments would substitute a median price from historical sales or a price oracle.

## Group 3: Transaction Signals (18 features)

These features capture fulfilment quality, payment patterns, and temporal velocity.

| Feature | Description |
|---------|-------------|
| `total_transactions` | Total completed or attempted transactions |
| `cancel_to_transaction_ratio` | Cancellations / total transactions |
| `refund_to_transaction_ratio` | Refunds / total transactions |
| `dispute_rate` | Disputes / total transactions |
| `crypto_payment_rate` | Fraction of transactions using crypto / wallet payments |
| `unique_buyers` | Count of distinct buyer IDs |
| `buyer_concentration` | `total_transactions / max(unique_buyers, 1)` — high value → same few buyers repeatedly |
| `avg_transaction_amount` | Mean transaction value |
| `transactions_last_1d` | Transactions in the last 1 day (velocity) |
| `transactions_last_7d` | Transactions in the last 7 days |
| `transactions_last_30d` | Transactions in the last 30 days |
| `amount_last_1d` | Total amount transacted in last 1 day |
| `amount_last_7d` | Total amount transacted in last 7 days |
| `amount_last_30d` | Total amount transacted in last 30 days |
| `transaction_velocity` | `transactions_last_7d / max(transactions_last_30d, 1)` |
| `burst_ratio` | `transactions_last_1d / max(transactions_last_7d, 1)` — detects sudden spikes |
| `high_value_transaction_rate` | Fraction of transactions above a high-value threshold |
| `payment_method_diversity` | Entropy of payment methods used |

## Group 4: Behavioural Signals (10 features)

These features model account lifecycle behaviour including device switching, login patterns, and review timing.

| Feature | Description |
|---------|-------------|
| `device_switch_count` | Number of distinct devices used (frequent switching → account takeover signal) |
| `ip_diversity` | Count of distinct IP addresses |
| `event_count` | Total recorded platform events |
| `login_event_rate` | Fraction of events that are login events |
| `avg_review_lag_hours` | Mean time (hours) between transaction and corresponding review |
| `review_lag_std` | Standard deviation of review lag (very low → bot-like review timing) |
| `reviews_within_1h` | Count of reviews posted within 1 hour of each other |
| `rapid_review_rate` | `reviews_within_1h / max(rating_count, 1)` |
| `profile_update_count` | Number of profile update events |
| `unusual_hour_event_rate` | Fraction of events occurring outside 06:00–22:00 local time |

## Group 5: Graph / Network Signals (9 features)

These features detect coordinated fraud rings by modelling seller relationships through shared infrastructure.

| Feature | Description |
|---------|-------------|
| `sellers_sharing_device` | Number of other sellers sharing the same device |
| `max_accounts_per_ip` | Maximum seller accounts sharing the same IP address |
| `shared_device_count` | Total shared-device relationships for this seller |
| `shared_ip_count` | Total shared-IP relationships |
| `buyer_overlap_score` | Fraction of this seller's buyers who also bought from co-device sellers |
| `shared_reviewer_count` | Count of reviewers shared with co-device sellers (fake-review signal) |
| `fraud_ring_size` | Size of the connected component this seller belongs to in the device graph |
| `degree_centrality` | NetworkX degree centrality in the seller–device bipartite graph |
| `is_isolated` | Binary flag: 1 if seller has no shared devices or IPs |

**Why graph features matter:** Fraud rings are impossible to detect from a single seller's features in isolation. Centrality and shared-infrastructure features give the model visibility into coordination.

## Missing Values and Scaling

- Features that cannot be computed (e.g., a seller with no transactions) are filled with **0**.
- The XGBoost model does not require feature scaling — tree splits are scale-invariant.
- The Logistic Regression baseline uses a `StandardScaler` inside its sklearn `Pipeline`.

## Feature Selection

No explicit feature selection step is applied. XGBoost's gradient boosting naturally down-weights irrelevant features. SHAP global importance (see [explainability.md](explainability.md)) is used post-training to understand which features actually drove predictions.

## Running the Feature Pipeline

The feature pipeline is run automatically as part of training:

```bash
make train
```

To inspect the saved feature matrix:

```python
import pandas as pd
features = pd.read_parquet("data/processed/features.parquet")
print(features.shape)   # (n_sellers, 50+)
```
