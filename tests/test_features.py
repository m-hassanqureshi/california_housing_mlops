import numpy as np
import pandas as pd
from src.features import construct_derived_features


def test_construct_derived_features():
    raw_payload = pd.DataFrame({
        "AveRooms": [10.0, 5.0],
        "AveOccup": [2.0, 1.0],
        "AveBedrms": [2.0, 1.0],
    })
    result_df = construct_derived_features(raw_payload)
    assert "rooms_per_household" in result_df.columns
    assert "bedrooms_per_room" in result_df.columns
    assert np.isclose(result_df["rooms_per_household"].iloc[0], 5.0)
    assert np.isclose(result_df["bedrooms_per_room"].iloc[0], 0.2)
