"""
Script to generate synthetic marketplace data and save to disk.

Usage:
    poetry run python scripts/generate_data.py
    poetry run python scripts/generate_data.py --config configs/config.yaml
    poetry run python scripts/generate_data.py --seed 123 --n-sellers 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is in the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from marketplace_trust.config import load_config
from marketplace_trust.data.generator import MarketplaceDataGenerator
from marketplace_trust.data.validation import validate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic marketplace data")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--n-sellers", type=int, help="Override number of sellers")
    parser.add_argument("--n-transactions", type=int, help="Override number of transactions")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.seed is not None:
        config.data.random_seed = args.seed
    if args.n_sellers is not None:
        config.data.n_sellers = args.n_sellers
    if args.n_transactions is not None:
        config.data.n_transactions = args.n_transactions

    output_dir = Path(args.output_dir or config.data.raw_data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating data with seed=%d", config.data.random_seed)
    generator = MarketplaceDataGenerator(config.data)
    dataset = generator.generate()

    # Validate before saving
    logger.info("Validating generated data...")
    result = validate_dataset(
        users=dataset.users,
        sellers=dataset.sellers,
        listings=dataset.listings,
        transactions=dataset.transactions,
        reviews=dataset.reviews,
        devices=dataset.devices,
        events=dataset.events,
    )
    result.raise_if_errors()

    # Save all tables
    tables = {
        "users": dataset.users,
        "sellers": dataset.sellers,
        "listings": dataset.listings,
        "transactions": dataset.transactions,
        "reviews": dataset.reviews,
        "devices": dataset.devices,
        "events": dataset.events,
    }

    for name, df in tables.items():
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %s: %d rows -> %s", name, len(df), path)

    # Save metadata
    import json

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(dataset.metadata, f, indent=2)
    logger.info("Saved metadata -> %s", meta_path)

    logger.info("Data generation complete.")


if __name__ == "__main__":
    main()
