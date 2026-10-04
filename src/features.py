"""Feature extraction and engineering routines for California Housing dataset.

Part of Phase 5 (Notebook Governance & Feature Extraction) for Model Owner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def construct_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates household demographic ratios and spatial clusters.

    Adds:
    - rooms_per_household: AveRooms / AveOccup
    - bedrooms_per_room: AveBedrms / AveRooms
    """
    df_out = df.copy()
    # Prevent divide-by-zero errors via epsilon clipping
    df_out["rooms_per_household"] = df_out["AveRooms"] / (
        df_out["AveOccup"] + 1e-5
    )
    df_out["bedrooms_per_room"] = df_out["AveBedrms"] / (
        df_out["AveRooms"] + 1e-5
    )
    return df_out
