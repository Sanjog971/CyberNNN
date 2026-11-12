from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path("data") / "raw" / "MachineLearningCVE" / "Tuesday-WorkingHours.pcap_ISCX.csv"


def read_csv_safely(path):
    """Read a CSV; fall back to latin-1 if the file is not valid UTF-8."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        print("UTF-8 decoding failed; retrying with latin-1")
        return pd.read_csv(path, encoding="latin-1")


df = read_csv_safely(DATA_FILE)

print("Shape (rows, columns):", df.shape)
print("Memory (MB):", round(df.memory_usage(deep=True).sum() / 1e6, 1))

print("\nColumn names (repr shows hidden spaces) and types:")
for i, name in enumerate(df.columns):
    print(f"{i:3d}  {name!r:45s} {df[name].dtype}")

print("\nNon-numeric columns and their most common values:")
for name in df.select_dtypes(exclude="number").columns:
    print(f"--- {name!r}")
    print(df[name].value_counts().head(20))

numeric = df.select_dtypes(include="number")
print("\nMissing values (total):", int(df.isna().sum().sum()))
print("Columns with missing values:")
missing = df.isna().sum()
print(missing[missing > 0])
print("\nInfinite values (total):", int(np.isinf(numeric).sum().sum()))
inf_counts = np.isinf(numeric).sum()
print("Columns with infinite values:")
print(inf_counts[inf_counts > 0])
print("\nDuplicate rows:", int(df.duplicated().sum()))