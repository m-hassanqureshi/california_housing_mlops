"""Model training stage for California Housing MLOps Pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
from lightgbm import LGBMRegressor
import pandas as pd
import yaml


def run_training(config_path: str = "configs/params.yaml") -> None:
    """Trains a LightGBM regression model on the processed training set."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = root_dir / "data" / "processed"
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    X_train = train_df.drop(columns=["MedHouseVal"])
    y_train = train_df["MedHouseVal"]

    train_params = config.get("train", {})
    regressor = LGBMRegressor(
        n_estimators=train_params.get("n_estimators", 250),
        learning_rate=train_params.get("learning_rate", 0.05),
        max_depth=train_params.get("max_depth", 6),
        num_leaves=train_params.get("num_leaves", 31),
        subsample=train_params.get("subsample", 0.8),
        colsample_bytree=train_params.get("colsample_bytree", 0.8),
        random_state=config["base"]["seed"],
        n_jobs=-1,
        verbose=-1,
    )

    regressor.fit(X_train, y_train)
    model_output_path = models_dir / "model.pkl"
    joblib.dump(regressor, model_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--config", default="configs/params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    run_training(args.config)
