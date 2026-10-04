# %% [markdown]
# # 01: Exploratory Data Analysis (California Housing)
#
# **Assignment 01: Git-Based Collaboration for an ML Project**
# * **Author:** Ahmad (Model Owner)
# * **Purpose:** Understand feature distributions, spatial clustering, and evaluate feature transformations.

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.features import construct_derived_features

# %% [markdown]
# ## 1. Ingest Data

# %%
data_path = Path("../data/raw/california_housing.csv")
if not data_path.exists():
    from sklearn.datasets import fetch_california_housing
    df = fetch_california_housing(as_frame=True).frame
else:
    df = pd.read_csv(data_path)

df.head()

# %% [markdown]
# ## 2. Summary Statistics & Missing Values

# %%
print(f"Dataset shape: {df.shape}")
print(f"Missing values count:\n{df.isnull().sum()}")
df.describe().T

# %% [markdown]
# ## 3. Feature Extraction Verification
# Test extracted features from `src.features`:

# %%
df_enriched = construct_derived_features(df)
df_enriched[["AveRooms", "AveOccup", "rooms_per_household", "AveBedrms", "bedrooms_per_room"]].head()

# %% [markdown]
# ## 4. Correlation Analysis

# %%
correlations = df_enriched.corr(numeric_only=True)["MedHouseVal"].sort_values(ascending=False)
print("Correlation with MedHouseVal:")
print(correlations)
