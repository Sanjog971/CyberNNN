"""Fitted preprocessing for CyberNN: fill gaps, compress, standardize.

Everything the preprocessor learns (feature list, medians, means, scales) comes
from the TRAINING split only. It is saved as plain JSON, not as a pickle, so
loading it can never run code hidden inside a file.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

NON_FEATURE_COLUMNS = ("Label", "is_attack")
SCHEMA_VERSION = 1


def signed_log1p(values):
    """log(1 + |x|) with the original sign; tames huge values, keeps negatives valid."""
    return np.sign(values) * np.log1p(np.abs(values))


class Preprocessor:
    def __init__(self):
        self.feature_names = []
        self.medians = None
        self.means = None
        self.scales = None
        self.excluded_columns = []
        self.constant_columns_dropped = []
        self.n_train_rows = 0

    def fit(self, train_df, exclude=()):
        """Learn everything from the training DataFrame only."""
        numeric = train_df.select_dtypes(include="number").columns
        candidates = [c for c in numeric if c not in NON_FEATURE_COLUMNS and c not in exclude]
        raw = train_df[candidates].replace([np.inf, -np.inf], np.nan)

        self.constant_columns_dropped = [c for c in candidates if raw[c].nunique(dropna=True) <= 1]
        self.feature_names = [c for c in candidates if c not in self.constant_columns_dropped]
        raw = raw[self.feature_names]

        medians = raw.median()
        self.medians = medians.to_numpy(dtype=float)
        logged = signed_log1p(raw.fillna(medians).to_numpy(dtype=float))
        self.means = logged.mean(axis=0)
        stds = logged.std(axis=0)
        self.scales = np.where(stds == 0, 1.0, stds)

        self.excluded_columns = list(exclude)
        self.n_train_rows = len(train_df)
        return self

    def transform(self, df):
        """Turn a DataFrame of flows into a float32 matrix, one row per flow."""
        if not self.feature_names:
            raise RuntimeError("Preprocessor is not fitted.")
        df = df.rename(columns=lambda c: str(c).strip())
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} required column(s): {missing[:10]}")

        x = df[self.feature_names].apply(pd.to_numeric, errors="coerce")
        values = x.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        values = np.where(np.isnan(values), self.medians, values)
        logged = signed_log1p(values)
        return ((logged - self.means) / self.scales).astype(np.float32)

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "n_train_rows": self.n_train_rows,
            "excluded_columns": self.excluded_columns,
            "constant_columns_dropped": self.constant_columns_dropped,
            "feature_names": self.feature_names,
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {data.get('schema_version')}")
        names = data["feature_names"]
        arrays = {k: np.asarray(data[k], dtype=float) for k in ("medians", "means", "scales")}
        for key, arr in arrays.items():
            if arr.shape != (len(names),) or not np.all(np.isfinite(arr)):
                raise ValueError(f"Invalid values or length for '{key}'")
        if np.any(arrays["scales"] <= 0):
            raise ValueError("Scales must be positive")
        pre = cls()
        pre.feature_names = list(names)
        pre.medians, pre.means, pre.scales = arrays["medians"], arrays["means"], arrays["scales"]
        pre.excluded_columns = list(data.get("excluded_columns", []))
        pre.constant_columns_dropped = list(data.get("constant_columns_dropped", []))
        pre.n_train_rows = int(data.get("n_train_rows", 0))
        return pre

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))