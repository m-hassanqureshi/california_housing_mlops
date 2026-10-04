"""Produce an outlier-removed version of the California Housing dataset.

Demonstrates DVC data version switching (PLAN.md checklist item 14).
Rows whose target (MedHouseVal) falls outside the 1.5 x IQR fence are dropped.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_DIR / "data" / "raw" / "california_housing.csv"
TARGET = "MedHouseVal"


def remove_outliers(path: Path = RAW_PATH, factor: float = 1.5) -> Path:
    """Drop target outliers using the IQR rule and overwrite path."""
    frame = pd.read_csv(path)
    before = len(frame)

    q1 = frame[TARGET].quantile(0.25)
    q3 = frame[TARGET].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - factor * iqr, q3 + factor * iqr

    frame = frame[(frame[TARGET] >= low) & (frame[TARGET] <= high)].reset_index(
        drop=True
    )
    frame.to_csv(path, index=False)

    logger.info(
        "Removed %d outlier rows (%d -> %d); target fence [%.3f, %.3f]",
        before - len(frame),
        before,
        len(frame),
        low,
        high,
    )
    return path


if __name__ == "__main__":
    remove_outliers()
