"""
CLI script to score a seller from the command line.

Useful for:
- Quick manual checks during development
- Batch scoring from a script
- Verifying that trained artefacts work end-to-end

Usage:
    # Score with default zero-feature baseline
    poetry run python scripts/predict.py --seller-id sel_000001

    # Supply specific features
    poetry run python scripts/predict.py \\
        --seller-id sel_000042 \\
        --features '{"seller_age_days": 7, "total_listings": 120, "cancel_to_transaction_ratio": 0.45}'

    # Score from a parquet feature file (e.g. produced by generate_data + feature pipeline)
    poetry run python scripts/predict.py --from-file data/processed/features.parquet --seller-id sel_000042
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marketplace_trust.inference.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a seller with the Trust Engine")
    parser.add_argument("--seller-id", default="unknown", help="Seller ID to score")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing trained model artefacts",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="JSON string of feature values, e.g. '{\"seller_age_days\": 14}'",
    )
    parser.add_argument(
        "--from-file",
        default=None,
        help="Parquet file with feature rows; --seller-id selects the row",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictor = Predictor.from_artefacts(args.models_dir)

    # Build feature dict
    features: dict[str, float] = {}

    if args.from_file:
        import pandas as pd

        df = pd.read_parquet(args.from_file)
        if "seller_id" in df.columns:
            row = df[df["seller_id"] == args.seller_id]
            if row.empty:
                logger.error("Seller %s not found in %s", args.seller_id, args.from_file)
                sys.exit(1)
            features = row.iloc[0].drop("seller_id", errors="ignore").to_dict()
        else:
            features = df.iloc[0].to_dict()

    if args.features:
        features.update(json.loads(args.features))

    prediction = predictor.predict(features=features, seller_id=args.seller_id)

    print(f"\n{'=' * 50}")
    print(f"  Seller:       {prediction.seller_id}")
    print(f"  Risk Score:   {prediction.risk_score} / 100")
    print(f"  Risk Level:   {prediction.risk_level}")
    print(f"  Classifier P: {prediction.classifier_probability:.4f}")
    print(f"  Anomaly:      {prediction.anomaly_score:.4f}")
    print(f"  Behavioral:   {prediction.behavioral_risk:.4f}")
    print("\n  Top Risk Factors:")
    for factor in prediction.top_risk_factors:
        print(f"    • {factor}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
