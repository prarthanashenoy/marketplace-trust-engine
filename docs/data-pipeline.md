# Data Pipeline

## Why Synthetic Data?

The project uses fully synthetic marketplace data for three reasons:

1. **Privacy** — no real user or transaction records are needed.
2. **Reproducibility** — a fixed random seed produces the exact same dataset every time.
3. **Controlled experiments** — fraud prevalence, noise level, and scenario mix are all configurable parameters.

## Database Schema

Seven relational tables are generated:

| Table | Default rows | Key columns |
|-------|-------------|-------------|
| `users` | 10,000 | `user_id`, `created_at`, `country`, `is_verified` |
| `sellers` | 1,000 | `seller_id`, `user_id`, `created_at`, `is_fraudulent`, `fraud_scenario` |
| `listings` | 10,000 | `listing_id`, `seller_id`, `price`, `market_price`, `created_at` |
| `transactions` | 50,000 | `transaction_id`, `seller_id`, `buyer_id`, `amount`, `status`, `payment_method` |
| `reviews` | 30,000 | `review_id`, `seller_id`, `reviewer_id`, `rating`, `created_at` |
| `devices` | 5,000 | `device_id`, `user_id`, `seller_id`, `ip_address` |
| `events` | 100,000 | `event_id`, `seller_id`, `event_type`, `device_id`, `created_at` |

All tables are serialised as **Parquet** files in `data/raw/`.

## Fraud Injection Strategy

### Key Design Principle: No Target Leakage

Fraud labels are assigned **at seller creation time**, before any feature is computed. Features are then derived from the table rows that were generated for that seller. This mirrors production: you observe behaviour and then infer fraud probability — you never know the label when computing features.

### Fraud Prevalence

15 % of sellers are flagged as fraudulent by default (`fraud_prevalence: 0.15` in `configs/config.yaml`).

### Noise Level

`noise_level: 0.3` controls overlap between the fraud and normal distributions. A higher value makes the fraud signals noisier and the problem harder, which prevents models from achieving unrealistically perfect scores.

### Nine Fraud Scenarios

Each fraudulent seller is assigned exactly one scenario. The scenario controls which synthetic signals are injected when generating their listings, transactions, reviews, and device records.

| Scenario | Key signals injected |
|----------|---------------------|
| `new_account_abuse` | Very new account, many listings, high cancellations |
| `mass_listing_abuse` | Burst of near-identical listings priced far below market |
| `scam_listings` | Unrealistically low prices, poor image counts, short descriptions |
| `payment_refund_abuse` | High refund and dispute rates, heavy cryptocurrency payments |
| `account_takeover` | Sudden device / IP changes, unusual transaction burst |
| `fraud_ring` | Multiple sellers sharing devices and IP addresses |
| `fake_review_network` | Reviews posted within minutes of each other, reviewer concentration |
| `device_sharing` | Multiple seller accounts registered to the same device |
| `mixed_suspicious` | Combination of signals from multiple scenarios above |

## Data Validation

After generation, a validation step checks:

- **Referential integrity** — no orphaned foreign keys
- **Schema correctness** — expected columns and dtypes are present
- **Label consistency** — `is_fraudulent` matches `fraud_scenario` (fraudulent ↔ non-null scenario)
- **Range constraints** — prices > 0, ratings in [1, 5], etc.

## Generating Data

```bash
make generate-data
# or
poetry run python scripts/generate_data.py
```

Options:

```
--config PATH          Path to config YAML (default: configs/config.yaml)
--output-dir PATH      Directory for parquet output (default: data/raw)
--seed INT             Override random seed
```

## Configuration

All sizes are controlled via `configs/config.yaml`:

```yaml
data:
  random_seed: 42
  n_users: 10000
  n_sellers: 1000
  n_listings: 10000
  n_transactions: 50000
  n_reviews: 30000
  n_devices: 5000
  n_events: 100000
  fraud_prevalence: 0.15
  noise_level: 0.3
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
```
