# California Housing - Data Dictionary

**Source:** `sklearn.datasets.fetch_california_housing`
**Raw file:** `data/raw/california_housing.csv` (DVC-tracked)
**Grain:** one row per census block group
**Rows:** 20,640 (original) / 19,569 (outlier-removed version)
**Target:** `MedHouseVal`

All numeric columns are stored as `float64`. Values are derived from the 1990 U.S. census.

| # | Column | Type | Units | Description | Notes |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | `MedInc` | float | 10k USD | Median income in the block group | Strongest predictor of house value |
| 2 | `HouseAge` | float | years | Median house age in the block group | |
| 3 | `AveRooms` | float | rooms | Average rooms per household | Influenced by a few very large values |
| 4 | `AveBedrms` | float | bedrooms | Average bedrooms per household | Highly correlated with `AveRooms` |
| 5 | `Population` | float | persons | Block group population | |
| 6 | `AveOccup` | float | persons | Average household members | Contains extreme outliers |
| 7 | `Latitude` | float | degrees | Block group latitude | 32.54 - 41.95 |
| 8 | `Longitude` | float | degrees | Block group longitude | -124.35 - -114.31 |
| 9 | `MedHouseVal` | float | 100k USD | Median house value (target) | Capped at 5.00001 |

## Notes for modeling

* `MedHouseVal` is **right-censored at 5.00001** - rows at the cap represent values of 500k USD or more.
* `AveRooms`, `AveBedrms`, `AveOccup` and `Population` contain heavy outliers; the `data/remove-outliers` branch demonstrated IQR filtering (1,071 rows removed by the 1.5 x IQR fence on the target).
* `Latitude` / `Longitude` often need feature engineering (e.g. distance to coast / city centroids) or geographic clustering.
* No missing values in the source dataset.

## Reference

Pace, R. Kelley and Ronald Barry, *Sparse Spatial Autoregressions*, Statistics & Probability Letters, 33 (1997) 291-297.
