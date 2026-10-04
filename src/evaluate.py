"""Evaluation stage for California Housing MLOps Pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml


def get_git_commit_hash() -> str:
    """Retrieves current Git commit SHA, or UNKNOWN_COMMIT if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "UNKNOWN_COMMIT"


def run_evaluation(config_path: str = "configs/params.yaml") -> None:
    """Evaluates the trained model on test data and logs metrics to JSON."""
    output_metric_path = "metrics/eval.json"
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        output_metric_path = config.get("evaluate", {}).get("output_metric_file", output_metric_path)

    processed_dir = root_dir / "data" / "processed"
    models_dir = root_dir / "models"
    metrics_file = root_dir / output_metric_path
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_parquet(processed_dir / "test.parquet")
    X_test = test_df.drop(columns=["MedHouseVal"])
    y_test = test_df["MedHouseVal"]

    model = joblib.load(models_dir / "model.pkl")
    predictions = model.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "git_commit_sha": get_git_commit_hash(),
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", default="configs/params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    run_evaluation(args.config)
