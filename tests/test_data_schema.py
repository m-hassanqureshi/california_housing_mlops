from pathlib import Path
import pandas as pd
import pytest

REQUIRED_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
]


def test_processed_data_schema():
    train_path = Path("data/processed/train.parquet")
    if not train_path.exists():
        pytest.skip("Processed dataset not pulled in light CI mode")

    df = pd.read_parquet(train_path)

    # Check schema completeness
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"

    # Assert no unexpected null allocations
    assert df.isnull().sum().sum() == 0, (
        "Null values detected in processed split"
    )

    # Value range assertions
    assert (df["MedHouseVal"] >= 0).all(), (
        "Target value contains negative numbers"
    )
    assert (df["HouseAge"] >= 0).all(), "House age contains negative values"
