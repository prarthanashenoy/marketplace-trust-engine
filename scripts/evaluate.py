"""
Evaluation script for the Marketplace Trust Engine.

Loads trained models and test data, runs the full evaluation suite,
prints a report to stdout, and saves artefacts to disk.

Usage:
    poetry run python scripts/evaluate.py
    poetry run python scripts/evaluate.py --models-dir models/ --data-dir data/raw/
    poetry run python scripts/evaluate.py --no-shap   # skip SHAP (faster)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from marketplace_trust.config import load_config
from marketplace_trust.data.generator import MarketplaceDataGenerator
from marketplace_trust.evaluation.explainability import SHAPExplainer
from marketplace_trust.evaluation.metrics import evaluate_model
from marketplace_trust.evaluation.reports import (
    print_comparison_table,
    print_evaluation_report,
    save_evaluation_json,
    save_shap_importance,
)
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
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--data-dir", default=None, help="Load data from disk instead of regenerating"
    )
    parser.add_argument(
        "--output-dir", default="data/processed", help="Where to save evaluation JSON artefacts"
    )
    parser.add_argument(
        "--no-shap", action="store_true", help="Skip SHAP computation (faster, for CI)"
    )
    return parser.parse_args()


def load_data(config, data_dir: str | None) -> dict[str, pd.DataFrame]:
    if data_dir is not None:
        p = Path(data_dir)
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
            tables[name] = pd.read_parquet(p / f"{name}.parquet")
        return tables
    ds = MarketplaceDataGenerator(config.data).generate()
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
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data and build features
    # ------------------------------------------------------------------
    logger.info("Loading data...")
    tables = load_data(config, args.data_dir)

    logger.info("Building feature matrix...")
    fm = build_feature_matrix(
        sellers=tables["sellers"],
        listings=tables["listings"],
        transactions=tables["transactions"],
        reviews=tables["reviews"],
        users=tables["users"],
        events=tables["events"],
    )

    split = temporal_split(
        X=fm.X,
        y=fm.y,
        seller_ids=fm.seller_ids,
        transactions=tables["transactions"],
        config=config.model.temporal_split,
    )

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    logger.info("Loading models from %s/", models_dir)
    baseline = BaselineModel.load(models_dir / "baseline.pkl")
    classifier = FraudClassifier.load(models_dir / "classifier.pkl")
    AnomalyDetector.load(models_dir / "anomaly.pkl")  # loaded to validate artefact exists
    ensemble = EnsembleRiskScorer.load(models_dir / "ensemble.pkl")

    thresholds = config.evaluation.thresholds
    top_k = config.evaluation.top_k_values

    # ------------------------------------------------------------------
    # Evaluate each model
    # ------------------------------------------------------------------
    evaluations = []

    for split_name, X_eval, y_eval in [
        ("val", split.X_val, split.y_val),
        ("test", split.X_test, split.y_test),
    ]:
        for name, model in [("baseline", baseline), ("classifier", classifier)]:
            proba = model.predict_proba(X_eval)
            ev = evaluate_model(
                model_name=name,
                split_name=split_name,
                y_true=y_eval,
                y_prob=proba,
                thresholds=thresholds,
                top_k_values=[k for k in top_k if k <= len(y_eval)],
            )
            evaluations.append(ev)
            print_evaluation_report(ev)
            save_evaluation_json(ev, output_dir / f"eval_{name}_{split_name}.json")

        # Ensemble risk score evaluation (treat risk_score/100 as probability)
        scores_df = ensemble.score(X_eval, None)
        ensemble_prob = scores_df["risk_score"].values / 100.0
        ev_ens = evaluate_model(
            model_name="ensemble",
            split_name=split_name,
            y_true=y_eval,
            y_prob=ensemble_prob,
            thresholds=thresholds,
            top_k_values=[k for k in top_k if k <= len(y_eval)],
        )
        evaluations.append(ev_ens)
        print_evaluation_report(ev_ens)
        save_evaluation_json(ev_ens, output_dir / f"eval_ensemble_{split_name}.json")

    # Summary comparison
    print_comparison_table(evaluations)

    # ------------------------------------------------------------------
    # SHAP global importance
    # ------------------------------------------------------------------
    if not args.no_shap:
        logger.info("Computing SHAP values on test set...")
        explainer = SHAPExplainer(classifier)
        explainer.fit(split.X_train)
        importance = explainer.global_importance(split.X_test)
        save_shap_importance(importance, output_dir / "shap_importance.csv")
        logger.info("Top 10 features by mean |SHAP|:")
        for feat, val in importance.head(10).items():
            logger.info("  %-45s %.4f", feat, val)

        # Save full SHAP matrix for dashboard use
        shap_df = explainer.summary_df(split.X_test)
        shap_df.to_parquet(output_dir / "shap_values_test.parquet", index=False)
        logger.info("SHAP values saved → %s", output_dir / "shap_values_test.parquet")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
