"""Validation of uploaded flow files. Uploaded files are untrusted input."""

import io

import numpy as np
import pandas as pd

MAX_ROWS = 500_000


class UploadError(ValueError):
    """An error whose message is safe and useful to show to the person who uploaded the file."""


def read_upload(file_bytes, max_rows=MAX_ROWS):
    """Parse uploaded bytes as a CSV, strip column-name spaces, and apply basic limits."""
    if not file_bytes:
        raise UploadError("The file is empty.")
    try:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise UploadError("The file contains no data.")
    except pd.errors.ParserError as error:
        raise UploadError(f"The file could not be parsed as a CSV: {error}")

    if len(df) == 0:
        raise UploadError("The file has a header row but no data rows.")
    if len(df) > max_rows:
        raise UploadError(f"The file has {len(df):,} rows; the limit is {max_rows:,}. Upload a smaller file.")
    df.columns = df.columns.str.strip()
    return df


def check_schema(df, feature_names):
    """Raise UploadError if any column the model needs is missing."""
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise UploadError(
            f"The file is missing {len(missing)} of the {len(feature_names)} columns the model needs. "
            f"First missing: {missing[:10]}. Found columns: {list(df.columns)[:10]}...")


def repair_report(df, feature_names):
    """Count values the preprocessor will have to repair (replace with training medians)."""
    raw = df[feature_names]
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    non_numeric = (numeric.isna() & raw.notna()).to_numpy()
    missing = raw.isna().to_numpy()
    infinite = np.isinf(numeric.to_numpy(dtype=float))
    return {
        "non_numeric_cells": int(non_numeric.sum()),
        "missing_cells": int(missing.sum()),
        "infinite_cells": int(infinite.sum()),
        "rows_affected": int((non_numeric | missing | infinite).any(axis=1).sum()),
    }


def ground_truth(df):
    """Return a 0/1 Series (1 = attack) if the file has labels, otherwise None."""
    if "Label" in df.columns:
        return (df["Label"].astype(str).str.strip() != "BENIGN").astype(int)
    if "is_attack" in df.columns:
        return df["is_attack"].astype(int)
    return None