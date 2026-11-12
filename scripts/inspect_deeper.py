from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path("data") / "raw" / "MachineLearningCVE" / "Tuesday-WorkingHours.pcap_ISCX.csv"

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()          # remove hidden leading/trailing spaces

print("Columns after stripping names:", df.shape[1])
print("Label column present:", "Label" in df.columns)

print("\n1. Class distribution")
counts = df["Label"].value_counts()
print(counts)
print((counts / len(df) * 100).round(2).astype(str) + " %")

print("\n2. Most common destination ports within each class")
for label, group in df.groupby("Label"):
    print(f"--- {label}")
    print(group["Destination Port"].value_counts().head(5))

print("\n3. Constant columns (only one distinct value)")
distinct = df.drop(columns="Label").nunique()
print(distinct[distinct == 1].index.tolist())

print("\n4. Is 'Fwd Header Length.1' an exact copy of 'Fwd Header Length'?")
if "Fwd Header Length.1" in df.columns:
    print(df["Fwd Header Length"].equals(df["Fwd Header Length.1"]))
else:
    print("Column 'Fwd Header Length.1' not found")

print("\n5. Duplicate rows by class")
duplicated = df.duplicated()
print(df.loc[duplicated, "Label"].value_counts())

print("\n6. Rows with missing or infinite values in the rate columns, by class")
rate_cols = ["Flow Bytes/s", "Flow Packets/s"]
bad = ~np.isfinite(df[rate_cols]).all(axis=1)
print("Rows affected:", int(bad.sum()))
print(df.loc[bad, "Label"].value_counts())