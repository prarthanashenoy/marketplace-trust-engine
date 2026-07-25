"""
Training script for the Marketplace Trust Engine.

Runs the full training pipeline:
  1. Generate (or load) synthetic data
  2. Build the feature matrix
  3. Temporal train/val/test split
  4. Fit baseline, classifier, anomaly detector
  5. Assemble the ensemble risk scorer
  6. Save all artefacts
  7. Log everything to MLflow

Usage:
    poetry run python scripts/train.py
    poetry run python scripts/train.py --config configs/config.yaml
    poetry run python scripts/train.py --no-mlflow   # skip MLflow logging
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from marketplace_trust.config import load_config
from marketplace_trust.data.generator import MarketplaceDataGenerator
from marketplace_trust.data.validation import validate_dataset
from marketplace_trust.features.pipeline import build_feature_matrix
from marketplace_trust.models.anomaly import AnomalyDetector
from marketplace_trust.models.baseline import BaselineModel
from marketplace_trust.models.classifier import FraudClassifier
from marketplace_trust.models.ensemble import EnsembleRiskScorer
from marketplace_trust.models.splitter import temporal_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Marketplace Trust Engine models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Load data from this directory instead of generating fresh data",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save model artefacts",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow experiment tracking",
    )
    return parser.parse_args()


def load_or_generate_data(config, data_dir: str | None) -> dict[str, pd.DataFrame]:
    """Load parquet files from disk or generate fresh synthetic data."""
    if data_dir is not None:
        p = Path(data_dir)
        logger.info("Loading data from %s", p)
        tables = {}
        for name in [
            "users",
            "sellers",
            "listings",
            "transactions",
            "reviews",
            "devices",
            "events",
        ]:
            fpath = p / f"{name}.parquet"
            if not fpath.exists():
                raise FileNotFoundError(f"Expected {fpath} but it does not exist.")
            tables[name] = pd.read_parquet(fpath)
            logger.info("  Loaded %s: %d rows", name, len(tables[name]))
        return tables

    logger.info("Generating fresh synthetic data (seed=%d)", config.data.random_seed)
    generator = MarketplaceDataGenerator(config.data)
    ds = generator.generate()

    result = validate_dataset(
        ds.users,
        ds.sellers,
        ds.listings,
        ds.transactions,
        ds.reviews,
        ds.devices,
        ds.events,
    )
    result.raise_if_errors()

    return {
        "users": ds.users,
        "sellers": ds.sellers,
        "listings": ds.listings,
        "transactions": ds.transactions,
        "reviews": ds.reviews,
        "devices": ds.devices,
        "events": ds.events,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    tables = load_or_generate_data(config, args.data_dir)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    logger.info("Building feature matrix...")
    fm = build_feature_matrix(
        sellers=tables["sellers"],
        listings=tables["listings"],
        transactions=tables["transactions"],
        reviews=tables["reviews"],
        users=tables["users"],
        events=tables["events"],
    )
    logger.info(
        "Feature matrix: %d sellers × %d features | fraud rate: %.2f%%",
        len(fm.X),
        fm.X.shape[1],
        fm.y.mean() * 100,
    )

    # ------------------------------------------------------------------
    # 3. Temporal split
    # ------------------------------------------------------------------
    split = temporal_split(
        X=fm.X,
        y=fm.y,
        seller_ids=fm.seller_ids,
        transactions=tables["transactions"],
        config=config.model.temporal_split,
    )

    # ------------------------------------------------------------------
    # 4. MLflow setup
    # ------------------------------------------------------------------
    use_mlflow = not args.no_mlflow
    run_id = None
    if use_mlflow:
        try:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
            mlflow.set_experiment(config.mlflow.experiment_name)
            mlflow.start_run()
            run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
            logger.info("MLflow run: %s", run_id)

            # Log dataset config
            mlflow.log_params(
                {
                    "n_sellers": config.data.n_sellers,
                    "n_transactions": config.data.n_transactions,
                    "random_seed": config.data.random_seed,
                    "noise_level": config.data.noise_level,
                    "n_features": fm.X.shape[1],
                    "train_size": len(split.X_train),
                    "val_size": len(split.X_val),
                    "test_size": len(split.X_test),
                    "train_fraud_rate": round(split.train_fraud_rate, 4),
                }
            )
            # Log XGBoost hyperparameters
            xgb_cfg = config.model.xgboost
            mlflow.log_params(
                {
                    "xgb_n_estimators": xgb_cfg.n_estimators,
                    "xgb_max_depth": xgb_cfg.max_depth,
                    "xgb_learning_rate": xgb_cfg.learning_rate,
                    "xgb_scale_pos_weight": xgb_cfg.scale_pos_weight,
                }
            )
        except Exception as e:
            logger.warning("MLflow unavailable (%s). Continuing without tracking.", e)
            use_mlflow = False

    # ------------------------------------------------------------------
    # 5. Train models
    # ------------------------------------------------------------------
    logger.info("Training baseline (LogisticRegression)...")
    baseline = BaselineModel(config.model.logistic_regression)
    baseline.fit(split.X_train, split.y_train)

    logger.info("Training XGBoost classifier...")
    classifier = FraudClassifier(config.model.xgboost)
    classifier.fit(split.X_train, split.y_train, split.X_val, split.y_val)

    logger.info("Training IsolationForest anomaly detector...")
    anomaly = AnomalyDetector(config.model.isolation_forest)
    anomaly.fit(split.X_train)

    # ------------------------------------------------------------------
    # 6. Quick validation metrics (logged but full eval is in evaluate.py)
    # ------------------------------------------------------------------
    from sklearn.metrics import average_precision_score, roc_auc_score

    for name, model, X_eval, y_eval in [
        ("baseline_val", baseline, split.X_val, split.y_val),
        ("classifier_val", classifier, split.X_val, split.y_val),
        ("classifier_test", classifier, split.X_test, split.y_test),
    ]:
        proba = model.predict_proba(X_eval)
        pr_auc = average_precision_score(y_eval, proba)
        roc_auc = roc_auc_score(y_eval, proba)
        logger.info("  %-20s | PR-AUC: %.4f | ROC-AUC: %.4f", name, pr_auc, roc_auc)
        if use_mlflow:
            mlflow.log_metrics(
                {
                    f"{name}_pr_auc": round(pr_auc, 4),
                    f"{name}_roc_auc": round(roc_auc, 4),
                }
            )

    # ------------------------------------------------------------------
    # 7. Assemble ensemble and save
    # ------------------------------------------------------------------
    ensemble = EnsembleRiskScorer(
        classifier=classifier,
        anomaly_detector=anomaly,
        baseline=baseline,
    )

    baseline.save(output_dir / "baseline.pkl")
    classifier.save(output_dir / "classifier.pkl")
    anomaly.save(output_dir / "anomaly.pkl")
    ensemble.save(output_dir / "ensemble.pkl")

    # Save feature names so inference can reconstruct the correct column order
    feature_meta = {
        "feature_names": fm.feature_names,
        "n_features": fm.X.shape[1],
        "run_id": run_id,
    }
    with open(output_dir / "feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    logger.info("All models saved to %s/", output_dir)

    if use_mlflow:
        mlflow.log_artifact(str(output_dir / "feature_meta.json"))
        mlflow.end_run()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
