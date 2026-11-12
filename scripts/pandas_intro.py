from pathlib import Path

import pandas as pd

DATA_FILE = Path("data") / "raw" / "sample_connections.csv"

df = pd.read_csv(DATA_FILE)

print("Shape (rows, columns):", df.shape)
print()
print("Column names:", list(df.columns))
print()
print("Data types:")
print(df.dtypes)
print()
print("First rows:")
print(df.head())
print()
print("Label distribution:")
print(df["label"].value_counts())
print()
print("Missing values per column:")
print(df.isna().sum())
print()
print("Average bytes_sent by label:")
print(df.groupby("label")["bytes_sent"].mean())
print(df[df["dst_port"] == 22])