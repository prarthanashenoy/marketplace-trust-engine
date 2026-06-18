"""
Central configuration loader for the Marketplace Trust Engine.

Reads from configs/config.yaml and environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    random_seed: int = 42
    n_users: int = 10000
    n_sellers: int = 1000
    n_listings: int = 10000
    n_transactions: int = 50000
    n_reviews: int = 30000
    n_devices: int = 5000
    n_events: int = 100000
    fraud_prevalence: float = 0.15
    scenario_weights: dict[str, float] = {}
    noise_level: float = 0.3
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"


class XGBoostConfig(BaseModel):
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 5.0
    random_state: int = 42


class IsolationForestConfig(BaseModel):
    n_estimators: int = 200
    contamination: float = 0.15
    random_state: int = 42


class LogisticRegressionConfig(BaseModel):
    max_iter: int = 5000
    random_state: int = 42
    C: float = 0.1


class TemporalSplitConfig(BaseModel):
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15


class ModelConfig(BaseModel):
    temporal_split: TemporalSplitConfig = TemporalSplitConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
    isolation_forest: IsolationForestConfig = IsolationForestConfig()
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()


class EvaluationConfig(BaseModel):
    thresholds: list[float] = [0.30, 0.50, 0.70, 0.90]
    top_k_values: list[int] = [100, 500, 1000]


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "models/trust_engine.pkl"


class DashboardConfig(BaseModel):
    port: int = 8501


class MLflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "marketplace-trust-engine"


class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    api: ApiConfig = ApiConfig()
    dashboard: DashboardConfig = DashboardConfig()
    mlflow: MLflowConfig = MLflowConfig()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load application configuration from a YAML file.

    Searches for config.yaml in the following order:
    1. Explicitly provided path
    2. CONFIG_PATH environment variable
    3. configs/config.yaml relative to the project root
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")

    config_path = Path(config_path)

    if not config_path.exists():
        # Try relative to this file's location
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        return AppConfig.model_validate(raw)

    return AppConfig()


# Module-level default config
config = load_config()
