# -*- coding: utf-8 -*-
"""Download the raw California Housing dataset into data/raw as a CSV.

Deliverable for Hassan's Data Owner work (PLAN.md Phase 4).
Target column: MedHouseVal.
"""
import logging
from pathlib import Path

from sklearn.datasets import fetch_california_housing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_DIR / "data" / "raw" / "california_housing.csv"


def main(output_path: Path = RAW_PATH) -> Path:
    """Fetch California Housing and persist it as a CSV at output_path."""
    logger.info("Fetching California Housing dataset")
    frame = fetch_california_housing(as_frame=True).frame
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    logger.info("Wrote %d rows x %d cols to %s", frame.shape[0], frame.shape[1], output_path)
    return output_path


if __name__ == "__main__":
    main()
"""Prepare data stage for California Housing MLOps Pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import yaml

from src.features import construct_derived_features


def run_preparation(config_path: str = "configs/params.yaml") -> None:
    """Prepares raw data and creates train and test parquet partitions."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = root_dir / "data" / "raw"
    processed_dir = root_dir / "data" / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_dir / "california_housing.csv"
    if not raw_file.exists():
        bunch = fetch_california_housing(as_frame=True)
        df = bunch.frame
        df.to_csv(raw_file, index=False)
    else:
        df = pd.read_csv(raw_file)

    # Optional feature engineering via configs
    if config.get("features", {}).get("engineer_ratios", False):
        df = construct_derived_features(df)

    # Enforce absolute isolation of preprocessing
    train_df, test_df = train_test_split(
        df,
        test_size=config["split"]["test_size"],
        random_state=config["base"]["seed"],
    )

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset splits")
    parser.add_argument("--config", default="configs/params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    run_preparation(args.config)
