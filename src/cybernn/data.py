"""Loading, stateless cleaning and splitting of CIC-IDS2017 flow CSV files.

"Stateless" means the step does not learn anything from the data (it only
strips names, removes exact copies, and so on), so it is safe to run before
the train/validation/test split. Steps that learn from data (scaling,
imputing, choosing constant columns) belong after the split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"


def load_flow_csv(path):
    """Read a flow CSV and strip hidden spaces from the column names."""
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df.columns = df.columns.str.strip()
    return df


def clean_flows(df):
    """Stateless cleaning. Returns (cleaned DataFrame, report dictionary)."""
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Expected a '{LABEL_COLUMN}' column, found: {list(df.columns)[:5]}...")

    report = {"rows_in": len(df)}
    df = df.copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    # 1. Drop columns that pandas renamed 'X.1' AND that are exact copies of 'X'.
    copies = [c for c in df.columns
              if c.endswith(".1") and c[:-2] in df.columns and df[c].equals(df[c[:-2]])]
    df = df.drop(columns=copies)
    report["duplicate_columns_dropped"] = copies

    # 2. Remove exact duplicate rows (before splitting, so copies cannot leak).
    before = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_removed"] = before - len(df)

    # 3. Report (but do not remove) identical features with different labels.
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
    report["rows_sharing_features_with_another_label"] = int(
        df.duplicated(subset=feature_cols, keep=False).sum())

    # 4. Turn infinite values into missing values (imputed later, after the split).
    numeric_cols = df.select_dtypes(include="number").columns
    report["infinite_values_replaced"] = int(np.isinf(df[numeric_cols]).to_numpy().sum())
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    report["rows_with_missing_values"] = int(df[numeric_cols].isna().any(axis=1).sum())

    # 5. Binary target: 0 = BENIGN, 1 = any attack.
    df["is_attack"] = (df[LABEL_COLUMN] != BENIGN_LABEL).astype(int)

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def split_data(df, seed=42):
    """Stratified 70/15/15 split (stratified on the original multi-class label)."""
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df[LABEL_COLUMN], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, stratify=train_val[LABEL_COLUMN], random_state=seed)
    return train, val, test