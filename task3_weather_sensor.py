import pandas as pd
import numpy as np
from scipy import stats

# Weather sensor data
data = {
    "Hour": range(1, 16),
    "Temperature": [
        22, 23, np.nan, np.nan, np.nan,
        25, 26, 58, 24, 23,
        22, 21, 23, 24, 25
    ]
}

df = pd.DataFrame(data)

# --------------------------------------------------
# 1. Statistics BEFORE cleaning
# --------------------------------------------------

print("Stats BEFORE cleaning:")

print("Mean:", df["Temperature"].mean())
print("Std:", df["Temperature"].std())

temp_before = df["Temperature"].dropna()

print("Skewness:", stats.skew(temp_before))
print("Kurtosis:", stats.kurtosis(temp_before))

# --------------------------------------------------
# 2. Fill missing values using linear interpolation
# --------------------------------------------------

df["Temperature_Clean"] = df["Temperature"].interpolate(
    method="linear"
)

print("\nAfter interpolation, missing values:",
      df["Temperature_Clean"].isnull().sum())

# --------------------------------------------------
# 3. Detect outlier using Z-score > 2
# --------------------------------------------------

z_scores = stats.zscore(df["Temperature_Clean"])

df["Zscore"] = z_scores

outlier_mask = df["Zscore"].abs() > 2

print("\nZ-score(>2) outlier detected:")

print(
    df.loc[
        outlier_mask,
        ["Hour", "Temperature_Clean", "Zscore"]
    ]
)

# --------------------------------------------------
# 4. Replace outlier with median
# --------------------------------------------------

median_temperature = df["Temperature_Clean"].median()

df.loc[outlier_mask, "Temperature_Clean"] = median_temperature

print("\nReplaced outlier with column median:",
      median_temperature)

# --------------------------------------------------
# 5. Standardize final cleaned temperature
# --------------------------------------------------

df["Temperature_Zscore"] = stats.zscore(
    df["Temperature_Clean"]
)

# --------------------------------------------------
# 6. Before / After comparison
# --------------------------------------------------

print("\nBefore/After comparison:")

print(
    df[
        [
            "Hour",
            "Temperature",
            "Temperature_Clean",
            "Temperature_Zscore"
        ]
    ]
)

# --------------------------------------------------
# 7. Statistics AFTER cleaning
# --------------------------------------------------

print("\nStats AFTER cleaning:")

print("Mean:",
      df["Temperature_Clean"].mean())

print("Std:",
      df["Temperature_Clean"].std())

print(
    "\nComparison:"
)

print(
    "Mean before:",
    df["Temperature"].mean()
)

print(
    "Mean after:",
    df["Temperature_Clean"].mean()
)

print(
    "Std before:",
    df["Temperature"].std()
)

print(
    "Std after:",
    df["Temperature_Clean"].std()
)