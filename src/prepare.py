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
