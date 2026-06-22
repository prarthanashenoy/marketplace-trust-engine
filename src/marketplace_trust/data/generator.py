"""
Synthetic Marketplace Data Generator.

Generates realistic-looking marketplace data with embedded fraud scenarios.
Fraud labels are derived from behavioral scenarios, not directly from
feature values, to avoid target leakage.

Design principles:
- All randomness is seeded for reproducibility.
- Distributions are realistic and overlapping (fraud is not trivially obvious).
- Each fraud scenario affects multiple correlated features.
- The noise_level parameter controls how much the distributions overlap.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from marketplace_trust.config import DataConfig

logger = logging.getLogger(__name__)

# Realistic country distribution (ISO codes weighted by marketplace activity)
COUNTRIES = [
    "US",
    "GB",
    "DE",
    "FR",
    "CA",
    "AU",
    "NL",
    "SE",
    "IT",
    "ES",
    "BR",
    "MX",
    "IN",
    "SG",
    "HK",
    "PL",
    "BE",
    "CH",
    "AT",
    "DK",
]
# Weights normalized to sum exactly to 1.0
_RAW_COUNTRY_WEIGHTS = [
    0.30,
    0.12,
    0.09,
    0.07,
    0.06,
    0.05,
    0.04,
    0.03,
    0.03,
    0.03,
    0.03,
    0.02,
    0.02,
    0.02,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
]
_total = sum(_RAW_COUNTRY_WEIGHTS)
COUNTRY_WEIGHTS = [w / _total for w in _RAW_COUNTRY_WEIGHTS]

SELLER_CATEGORIES = [
    "electronics",
    "fashion",
    "home",
    "sports",
    "collectibles",
    "books",
    "automotive",
    "other",
]
CATEGORY_WEIGHTS = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.05]

# Market price ranges per category (mean, std) in USD
CATEGORY_PRICE_PARAMS: dict[str, tuple[float, float]] = {
    "electronics": (250.0, 150.0),
    "fashion": (60.0, 40.0),
    "home": (80.0, 60.0),
    "sports": (90.0, 70.0),
    "collectibles": (120.0, 100.0),
    "books": (20.0, 15.0),
    "automotive": (200.0, 180.0),
    "other": (50.0, 40.0),
}


@dataclass
class GeneratedDataset:
    """Container for all generated marketplace entities."""

    users: pd.DataFrame
    sellers: pd.DataFrame
    listings: pd.DataFrame
    transactions: pd.DataFrame
    reviews: pd.DataFrame
    devices: pd.DataFrame
    events: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class MarketplaceDataGenerator:
    """
    Generates synthetic marketplace data with realistic fraud scenarios.

    The generator creates a self-consistent dataset where:
    - Users own seller accounts
    - Sellers create listings
    - Buyers transact on listings
    - Reviews follow transactions
    - Events track user behavior

    Fraud scenarios are injected by modifying the behavioral parameters
    of a subset of sellers, not by post-hoc label assignment.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self._simulation_start = datetime(2023, 1, 1)
        self._simulation_end = datetime(2024, 6, 30)
        self._sim_days = (self._simulation_end - self._simulation_start).days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> GeneratedDataset:
        """Generate the full dataset. Returns a GeneratedDataset."""
        logger.info("Starting data generation (seed=%d)", self.config.random_seed)

        devices = self._generate_devices()
        users = self._generate_users(devices)
        # Pass a copy so the original users df is not mutated across calls
        users_work = users.copy()
        sellers, scenario_map = self._generate_sellers(users_work)
        listings = self._generate_listings(sellers, scenario_map)
        transactions = self._generate_transactions(sellers, listings, users_work, scenario_map)
        reviews = self._generate_reviews(transactions, sellers, scenario_map)
        events = self._generate_events(users_work, devices, scenario_map)

        dataset = GeneratedDataset(
            users=users_work,
            sellers=sellers,
            listings=listings,
            transactions=transactions,
            reviews=reviews,
            devices=devices,
            events=events,
            metadata={
                "random_seed": self.config.random_seed,
                "simulation_start": self._simulation_start.isoformat(),
                "simulation_end": self._simulation_end.isoformat(),
                "n_fraud_sellers": int(sellers["is_fraudulent"].sum()),
                "fraud_rate": float(sellers["is_fraudulent"].mean()),
            },
        )

        self._log_summary(dataset)
        return dataset

    # ------------------------------------------------------------------
    # Device generation
    # ------------------------------------------------------------------

    def _generate_devices(self) -> pd.DataFrame:
        n = self.config.n_devices
        logger.info("Generating %d devices", n)

        device_ids = [f"dev_{i:06d}" for i in range(n)]
        device_types = self.rng.choice(
            ["mobile", "desktop", "tablet"],
            size=n,
            p=[0.55, 0.35, 0.10],
        )
        countries = self.rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
        days_offset = self.rng.integers(0, self._sim_days - 30, size=n)
        first_seen = [self._simulation_start + timedelta(days=int(d)) for d in days_offset]

        return pd.DataFrame(
            {
                "device_id": device_ids,
                "device_type": device_types,
                "country": countries,
                "first_seen_at": first_seen,
            }
        )

    # ------------------------------------------------------------------
    # User generation
    # ------------------------------------------------------------------

    def _generate_users(self, devices: pd.DataFrame) -> pd.DataFrame:
        n = self.config.n_users
        logger.info("Generating %d users", n)

        device_ids = devices["device_id"].tolist()
        user_devices = self.rng.choice(device_ids, size=n)

        # Account creation: spread over simulation period
        days_offset = self.rng.integers(0, self._sim_days - 10, size=n)
        created_at = [self._simulation_start + timedelta(days=int(d)) for d in days_offset]
        reference_date = self._simulation_end
        account_age_days = [(reference_date - dt).days for dt in created_at]

        countries = self.rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
        verification_options = ["unverified", "email", "phone", "full"]
        verification_weights = [0.20, 0.40, 0.25, 0.15]
        verification = self.rng.choice(verification_options, size=n, p=verification_weights)

        return pd.DataFrame(
            {
                "user_id": [f"usr_{i:07d}" for i in range(n)],
                "account_created_at": created_at,
                "country": countries,
                "device_id": user_devices,
                "account_age_days": account_age_days,
                "verification_status": verification,
                "is_seller": False,
            }
        )

    # ------------------------------------------------------------------
    # Seller generation + fraud scenario assignment
    # ------------------------------------------------------------------

    def _assign_scenarios(self, n_sellers: int) -> list[str]:
        """
        Assign a fraud scenario to each seller.

        Uses the configured scenario weights from config.yaml.
        Returns a list of scenario names of length n_sellers.
        """
        weights = self.config.scenario_weights
        if not weights:
            weights = {
                "normal": 0.70,
                "new_account_abuse": 0.04,
                "mass_listing_abuse": 0.03,
                "scam_listings": 0.04,
                "payment_refund_abuse": 0.04,
                "account_takeover": 0.03,
                "fraud_ring": 0.04,
                "fake_review_network": 0.03,
                "device_sharing": 0.03,
                "mixed_suspicious": 0.02,
            }

        scenario_names = list(weights.keys())
        scenario_probs = np.array(list(weights.values()), dtype=float)
        scenario_probs /= scenario_probs.sum()  # normalize

        return self.rng.choice(scenario_names, size=n_sellers, p=scenario_probs).tolist()

    def _generate_sellers(self, users: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Generate seller accounts, assigned to a subset of users.

        Returns:
            sellers DataFrame
            scenario_map: {seller_id -> scenario_name}
        """
        n = self.config.n_sellers
        logger.info("Generating %d sellers", n)

        # Pick users to be sellers (no overlap)
        seller_user_indices = self.rng.choice(len(users), size=n, replace=False)
        seller_users = users.iloc[seller_user_indices].copy()

        # Mark those users as sellers in the passed-in users df
        users.loc[seller_user_indices, "is_seller"] = True

        scenarios = self._assign_scenarios(n)
        is_fraudulent = [s != "normal" for s in scenarios]

        seller_ids = [f"sel_{i:06d}" for i in range(n)]
        categories = self.rng.choice(SELLER_CATEGORIES, size=n, p=CATEGORY_WEIGHTS)

        # Seller creation: same time as or after user creation
        days_extra = self.rng.integers(0, 30, size=n)
        seller_created_at = [
            seller_users.iloc[i]["account_created_at"] + timedelta(days=int(days_extra[i]))
            for i in range(n)
        ]

        # Base ratings: normal sellers cluster around 4.2-4.8
        ratings = self._generate_seller_ratings(scenarios)
        cancellation_rates = self._generate_cancellation_rates(scenarios)
        n_listings = self._generate_listing_counts(scenarios)
        n_transactions = self._generate_transaction_counts(scenarios, n_listings)

        sellers_df = pd.DataFrame(
            {
                "seller_id": seller_ids,
                "user_id": seller_users["user_id"].tolist(),
                "seller_created_at": seller_created_at,
                "category": categories,
                "average_rating": ratings,
                "number_of_listings": n_listings,
                "number_of_transactions": n_transactions,
                "cancellation_rate": cancellation_rates,
                "fraud_scenario": scenarios,
                "is_fraudulent": is_fraudulent,
            }
        )

        scenario_map = dict(zip(seller_ids, scenarios))
        return sellers_df, scenario_map

    def _generate_seller_ratings(self, scenarios: list[str]) -> list[float]:
        """Generate realistic ratings that partially overlap between fraud and normal."""
        noise = self.config.noise_level
        ratings = []
        for scenario in scenarios:
            if scenario == "normal":
                # Genuine ratings: beta-ish distribution centered around 4.3
                r = self.rng.beta(8, 2) * 4 + 1  # range [1, 5], skewed high
            elif scenario == "fake_review_network":
                # Artificially inflated: very high but suspiciously uniform
                r = self.rng.normal(4.8, 0.15 * (1 + noise))
            elif scenario in ("fraud_ring", "new_account_abuse"):
                # New accounts: fewer reviews, more variable
                r = self.rng.normal(4.4, 0.5 * (1 + noise))
            elif scenario == "account_takeover":
                # Used to be good, now degraded
                r = self.rng.normal(3.8, 0.7 * (1 + noise))
            else:
                r = self.rng.normal(4.0, 0.6 * (1 + noise))
            ratings.append(float(np.clip(r, 1.0, 5.0)))
        return ratings

    def _generate_cancellation_rates(self, scenarios: list[str]) -> list[float]:
        """Generate cancellation rates. Fraud scenarios tend to have higher rates."""
        noise = self.config.noise_level
        rates = []
        for scenario in scenarios:
            if scenario == "normal":
                r = self.rng.beta(1.5, 15)  # mostly low, ~5-10%
            elif scenario in ("payment_refund_abuse", "scam_listings"):
                r = self.rng.beta(5, 8) * (1 + noise * 0.3)
            elif scenario == "fraud_ring":
                r = self.rng.beta(3, 10) * (1 + noise * 0.2)
            else:
                r = self.rng.beta(2, 12) * (1 + noise * 0.1)
            rates.append(float(np.clip(r, 0.0, 1.0)))
        return rates

    def _generate_listing_counts(self, scenarios: list[str]) -> list[int]:
        """Generate number of listings. Mass listers have many more."""
        noise = self.config.noise_level
        counts = []
        for scenario in scenarios:
            if scenario == "normal":
                c = max(1, int(self.rng.lognormal(2.0, 0.8)))  # median ~7
            elif scenario == "mass_listing_abuse":
                c = max(50, int(self.rng.lognormal(5.0, 0.5) * (1 + noise * 0.5)))
            elif scenario == "fraud_ring":
                c = max(20, int(self.rng.lognormal(3.5, 0.6) * (1 + noise * 0.3)))
            elif scenario == "new_account_abuse":
                c = max(10, int(self.rng.lognormal(3.0, 0.5)))
            else:
                c = max(1, int(self.rng.lognormal(2.5, 0.9)))
            counts.append(c)
        return counts

    def _generate_transaction_counts(
        self, scenarios: list[str], listing_counts: list[int]
    ) -> list[int]:
        """Transaction counts are correlated with listing counts."""
        counts = []
        for scenario, n_list in zip(scenarios, listing_counts):
            ratio = self.rng.uniform(0.3, 0.7)
            if scenario in ("payment_refund_abuse", "fraud_ring"):
                ratio = self.rng.uniform(0.6, 1.2)
            elif scenario == "scam_listings":
                ratio = self.rng.uniform(0.05, 0.25)
            c = max(0, int(n_list * ratio + self.rng.normal(0, 2)))
            counts.append(c)
        return counts

    # ------------------------------------------------------------------
    # Listings generation
    # ------------------------------------------------------------------

    def _generate_listings(
        self,
        sellers: pd.DataFrame,
        scenario_map: dict[str, str],
    ) -> pd.DataFrame:
        n = self.config.n_listings
        logger.info("Generating %d listings", n)

        listing_weights = sellers["number_of_listings"].values.astype(float)
        listing_weights /= listing_weights.sum()

        seller_assignments = self.rng.choice(
            sellers["seller_id"].tolist(), size=n, p=listing_weights
        )

        seller_category_map = dict(zip(sellers["seller_id"], sellers["category"]))
        seller_created_map = dict(zip(sellers["seller_id"], sellers["seller_created_at"]))

        records = []
        for i, seller_id in enumerate(seller_assignments):
            scenario = scenario_map[seller_id]
            category = seller_category_map[seller_id]
            seller_created = seller_created_map[seller_id]

            market_mean, market_std = CATEGORY_PRICE_PARAMS[category]
            market_price = max(1.0, self.rng.normal(market_mean, market_std))

            price_ratio = self._sample_price_ratio(scenario)
            price = max(0.50, market_price * price_ratio)

            days_after = self.rng.integers(0, max(1, self._sim_days - 60))
            created_at = seller_created + timedelta(days=int(days_after))
            if created_at > self._simulation_end:
                created_at = self._simulation_end - timedelta(days=1)

            desc_length = self._sample_description_length(scenario)
            image_count = self._sample_image_count(scenario)

            status_options = ["active", "sold", "removed", "expired"]
            if scenario == "scam_listings":
                status_weights = [0.4, 0.1, 0.4, 0.1]
            else:
                status_weights = [0.3, 0.5, 0.1, 0.1]
            status = self.rng.choice(status_options, p=status_weights)

            records.append(
                {
                    "listing_id": f"lst_{i:07d}",
                    "seller_id": seller_id,
                    "category": category,
                    "price": round(float(price), 2),
                    "market_price": round(float(market_price), 2),
                    "created_at": created_at,
                    "description_length": int(desc_length),
                    "image_count": int(image_count),
                    "listing_status": status,
                }
            )

        return pd.DataFrame(records)

    def _sample_price_ratio(self, scenario: str) -> float:
        """Sample price-to-market ratio. Scam listings are unusually cheap."""
        noise = self.config.noise_level
        if scenario == "normal":
            return float(self.rng.normal(0.92, 0.12 * (1 + noise)))
        elif scenario == "scam_listings":
            return float(self.rng.normal(0.45, 0.15 * (1 + noise)))
        elif scenario == "mass_listing_abuse":
            return float(self.rng.normal(0.75, 0.15 * (1 + noise)))
        elif scenario == "fraud_ring":
            return float(self.rng.normal(0.70, 0.18 * (1 + noise)))
        else:
            return float(self.rng.normal(0.85, 0.15 * (1 + noise)))

    def _sample_description_length(self, scenario: str) -> int:
        if scenario in ("scam_listings", "new_account_abuse"):
            return max(10, int(self.rng.lognormal(3.0, 0.8)))
        elif scenario == "normal":
            return max(50, int(self.rng.lognormal(5.5, 0.6)))
        else:
            return max(20, int(self.rng.lognormal(4.5, 0.8)))

    def _sample_image_count(self, scenario: str) -> int:
        if scenario in ("scam_listings", "new_account_abuse"):
            return int(self.rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3]))
        elif scenario == "normal":
            return int(max(1, self.rng.integers(2, 8)))
        else:
            return int(max(1, self.rng.integers(1, 5)))

    # ------------------------------------------------------------------
    # Transactions generation
    # ------------------------------------------------------------------

    def _generate_transactions(
        self,
        sellers: pd.DataFrame,
        listings: pd.DataFrame,
        users: pd.DataFrame,
        scenario_map: dict[str, str],
    ) -> pd.DataFrame:
        n = self.config.n_transactions
        logger.info("Generating %d transactions", n)

        buyer_ids = users["user_id"].tolist()

        active_listings = listings[listings["listing_status"].isin(["sold", "active"])].copy()
        if len(active_listings) == 0:
            active_listings = listings.copy()

        listing_weights = 1.0 / (active_listings["price"].values + 1.0)
        listing_weights /= listing_weights.sum()

        sampled_listings = active_listings.sample(
            n=n,
            replace=True,
            weights=listing_weights,
            random_state=int(self.config.random_seed),
        )

        payment_methods = ["card", "paypal", "bank_transfer", "crypto", "wallet"]
        payment_weights_normal = [0.50, 0.25, 0.15, 0.03, 0.07]
        payment_weights_fraud = [0.30, 0.15, 0.05, 0.25, 0.25]

        records = []
        for i, (_, listing) in enumerate(sampled_listings.iterrows()):
            seller_id = listing["seller_id"]
            scenario = scenario_map.get(seller_id, "normal")

            if scenario == "fraud_ring":
                if self.rng.random() < 0.5:
                    pool_size = min(20, len(buyer_ids))
                    buyer_id = self.rng.choice(buyer_ids[:pool_size])
                else:
                    buyer_id = self.rng.choice(buyer_ids)
            else:
                buyer_id = self.rng.choice(buyer_ids)

            listing_created = listing["created_at"]
            days_after = max(0, int(self.rng.exponential(14)))
            ts = listing_created + timedelta(days=days_after, hours=int(self.rng.integers(0, 24)))
            if ts > self._simulation_end:
                ts = self._simulation_end - timedelta(hours=1)

            amount = listing["price"] * self.rng.uniform(0.95, 1.05)

            if scenario in ("fraud_ring", "payment_refund_abuse", "account_takeover"):
                payment = self.rng.choice(payment_methods, p=payment_weights_fraud)
            else:
                payment = self.rng.choice(payment_methods, p=payment_weights_normal)

            cancel_flag, refund_flag, status = self._sample_transaction_outcome(scenario)

            records.append(
                {
                    "transaction_id": f"txn_{i:08d}",
                    "buyer_id": buyer_id,
                    "seller_id": seller_id,
                    "listing_id": listing["listing_id"],
                    "amount": round(float(amount), 2),
                    "timestamp": ts,
                    "payment_method": payment,
                    "status": status,
                    "cancellation_flag": cancel_flag,
                    "refund_flag": refund_flag,
                }
            )

        return pd.DataFrame(records)

    def _sample_transaction_outcome(self, scenario: str) -> tuple[bool, bool, str]:
        """Return (cancellation_flag, refund_flag, status)."""
        r = self.rng.random()
        if scenario == "normal":
            if r < 0.88:
                return False, False, "completed"
            elif r < 0.95:
                return True, False, "cancelled"
            else:
                return False, True, "refunded"
        elif scenario == "payment_refund_abuse":
            if r < 0.45:
                return False, True, "refunded"
            elif r < 0.65:
                return True, False, "cancelled"
            elif r < 0.75:
                return False, False, "disputed"
            else:
                return False, False, "completed"
        elif scenario == "scam_listings":
            if r < 0.50:
                return True, False, "cancelled"
            elif r < 0.70:
                return False, False, "disputed"
            else:
                return False, False, "completed"
        else:
            if r < 0.80:
                return False, False, "completed"
            elif r < 0.92:
                return True, False, "cancelled"
            else:
                return False, True, "refunded"

    # ------------------------------------------------------------------
    # Reviews generation
    # ------------------------------------------------------------------

    def _generate_reviews(
        self,
        transactions: pd.DataFrame,
        sellers: pd.DataFrame,
        scenario_map: dict[str, str],
    ) -> pd.DataFrame:
        n = self.config.n_reviews
        logger.info("Generating %d reviews", n)

        completed = transactions[transactions["status"] == "completed"].copy()
        if len(completed) == 0:
            completed = transactions.copy()

        sample_size = min(n, len(completed))
        sampled = completed.sample(
            n=sample_size,
            replace=(sample_size > len(completed)),
            random_state=int(self.config.random_seed),
        )

        records = []
        for i, (_, txn) in enumerate(sampled.iterrows()):
            seller_id = txn["seller_id"]
            scenario = scenario_map.get(seller_id, "normal")

            hours_after = max(1, int(self.rng.exponential(48)))
            if scenario == "fake_review_network":
                hours_after = max(1, int(self.rng.exponential(4)))
            ts = txn["timestamp"] + timedelta(hours=hours_after)
            if ts > self._simulation_end:
                ts = self._simulation_end - timedelta(hours=1)

            rating = self._sample_review_rating(scenario)
            text_length = self._sample_review_text_length(scenario)

            records.append(
                {
                    "review_id": f"rev_{i:08d}",
                    "buyer_id": txn["buyer_id"],
                    "seller_id": seller_id,
                    "transaction_id": txn["transaction_id"],
                    "rating": float(rating),
                    "timestamp": ts,
                    "text_length": int(text_length),
                }
            )

        return pd.DataFrame(records)

    def _sample_review_rating(self, scenario: str) -> float:
        noise = self.config.noise_level
        if scenario == "normal":
            r = self.rng.beta(6, 2) * 4 + 1
        elif scenario == "fake_review_network":
            r = self.rng.normal(4.9, 0.1 * (1 + noise))
        elif scenario == "account_takeover":
            r = self.rng.normal(2.5, 1.0 * (1 + noise))
        else:
            r = self.rng.normal(4.0, 0.8 * (1 + noise))
        return float(np.clip(r, 1.0, 5.0))

    def _sample_review_text_length(self, scenario: str) -> int:
        if scenario == "fake_review_network":
            return max(5, int(self.rng.lognormal(2.0, 0.5)))
        elif scenario == "normal":
            return max(20, int(self.rng.lognormal(4.5, 0.8)))
        else:
            return max(10, int(self.rng.lognormal(3.5, 0.8)))

    # ------------------------------------------------------------------
    # Events generation
    # ------------------------------------------------------------------

    def _generate_events(
        self,
        users: pd.DataFrame,
        devices: pd.DataFrame,
        scenario_map: dict[str, str],
    ) -> pd.DataFrame:
        n = self.config.n_events
        logger.info("Generating %d events", n)

        user_ids = users["user_id"].tolist()
        device_ids = devices["device_id"].tolist()

        event_types = [
            "login",
            "listing_view",
            "search",
            "purchase",
            "message",
            "review",
            "profile_update",
            "logout",
        ]
        event_weights = [0.15, 0.30, 0.25, 0.10, 0.08, 0.04, 0.03, 0.05]

        selected_users = self.rng.choice(user_ids, size=n)
        selected_devices = self.rng.choice(device_ids, size=n)
        selected_events = self.rng.choice(event_types, size=n, p=event_weights)
        days_offsets = self.rng.integers(0, self._sim_days, size=n)
        hour_offsets = self.rng.integers(0, 24, size=n)
        minute_offsets = self.rng.integers(0, 60, size=n)

        n_ips = max(100, n // 100)
        ip_pool = [f"ip_{hashlib.md5(f'ip_{i}'.encode()).hexdigest()[:8]}" for i in range(n_ips)]
        selected_ips = self.rng.choice(ip_pool, size=n)

        records = {
            "event_id": [f"evt_{i:09d}" for i in range(n)],
            "user_id": selected_users.tolist(),
            "timestamp": [
                self._simulation_start
                + timedelta(
                    days=int(days_offsets[i]),
                    hours=int(hour_offsets[i]),
                    minutes=int(minute_offsets[i]),
                )
                for i in range(n)
            ],
            "event_type": selected_events.tolist(),
            "ip_hash": selected_ips.tolist(),
            "device_id": selected_devices.tolist(),
        }

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _log_summary(self, dataset: GeneratedDataset) -> None:
        sellers = dataset.sellers
        logger.info("=== Data Generation Summary ===")
        logger.info("Users:        %d", len(dataset.users))
        logger.info("Sellers:      %d", len(sellers))
        logger.info(
            "  - Normal:      %d (%.1f%%)",
            (sellers["fraud_scenario"] == "normal").sum(),
            (sellers["fraud_scenario"] == "normal").mean() * 100,
        )
        logger.info(
            "  - Fraudulent:  %d (%.1f%%)",
            sellers["is_fraudulent"].sum(),
            sellers["is_fraudulent"].mean() * 100,
        )
        logger.info("Listings:     %d", len(dataset.listings))
        logger.info("Transactions: %d", len(dataset.transactions))
        logger.info("Reviews:      %d", len(dataset.reviews))
        logger.info("Devices:      %d", len(dataset.devices))
        logger.info("Events:       %d", len(dataset.events))
        for scenario in sellers["fraud_scenario"].unique():
            count = (sellers["fraud_scenario"] == scenario).sum()
            logger.info("  Scenario %-25s: %d", scenario, count)
