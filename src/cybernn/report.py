"""Investigation exports: flagged-flow CSV, safe against spreadsheet formulas, and a Markdown summary."""

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
MAX_EXPORT_ROWS = 50_000


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sanitize_for_csv(df):
    """Prefix text cells and column names that start with = + - @ so spreadsheets do not run them as formulas."""
    out = df.copy()
    out.columns = [("'" + c if c.startswith(FORMULA_PREFIXES) else c) for c in map(str, out.columns)]
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "'" + v if isinstance(v, str) and v.startswith(FORMULA_PREFIXES) else v)
    return out


def flagged_export(df, scores, threshold, max_rows=MAX_EXPORT_ROWS):
    """All flagged flows (highest score first) with their line numbers. Returns (DataFrame, truncated)."""
    table = df.copy()
    table.insert(0, "score", scores)
    table.insert(0, "csv_line", table.index + 2)          # line number in the uploaded file
    table = table[scores >= threshold].sort_values("score", ascending=False)
    truncated = len(table) > max_rows
    return sanitize_for_csv(table.head(max_rows)), truncated


def build_report(file_name, file_sha256, df, scores, threshold, default_threshold, repair,
                 metrics, model_info):
    """A short Markdown summary an analyst can attach to a ticket."""
    flagged = scores >= threshold
    n, n_flagged = len(df), int(flagged.sum())
    safe_name = str(file_name).replace("`", "'")
    lines = [
        "# CyberNN investigation summary", "",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", "",
        "## Input",
        f"- File: `{safe_name}`",
        f"- File SHA-256: `{file_sha256}`",
        f"- Flows scored: {n:,}",
        f"- Values repaired before scoring: {repair['rows_affected']:,} rows "
        f"({repair['missing_cells']:,} missing, {repair['infinite_cells']:,} infinite, "
        f"{repair['non_numeric_cells']:,} non-numeric cells; replaced with training-set medians)", "",
        "## Model",
        f"- Model file SHA-256: `{model_info['model_sha256']}`",
        f"- Preprocessor SHA-256: `{model_info['preprocessor_sha256']}`",
        f"- Architecture: MLP, input features {model_info['n_features']}, hidden layers {model_info['hidden']}",
        f"- Training data: CIC-IDS2017 Tuesday file only ({model_info['n_train_rows']:,} flows); "
        f"seed {model_info['seed']}, best epoch {model_info['best_epoch']}", "",
        "## Result",
        f"- Alert threshold used: {threshold:.2f} (default chosen on validation data: {default_threshold:.2f})",
        f"- Flagged as suspicious: {n_flagged:,} of {n:,} ({n_flagged / n * 100:.2f}%)",
    ]
    if n_flagged:
        flagged_scores = scores[flagged]
        lines.append(f"- Scores of flagged flows: min {flagged_scores.min():.4f}, "
                     f"median {np.median(flagged_scores):.4f}, max {flagged_scores.max():.4f}")
        if "Destination Port" in df.columns:
            ports = pd.to_numeric(df.loc[flagged, "Destination Port"], errors="coerce").dropna().astype(int)
            if len(ports):
                top = ", ".join(f"{p} ({c:,})" for p, c in ports.value_counts().head(5).items())
                lines.append(f"- Most common destination ports among flagged flows: {top}")
    lines.append("")
    if metrics is not None:
        lines += [
            "## Evaluation against the file's own labels",
            f"- Precision {metrics['precision']:.3f}, recall {metrics['recall']:.3f}, F1 {metrics['f1']:.3f}, "
            f"false-positive rate {metrics['fpr']:.4f}",
            f"- Confusion counts: TN {metrics['TN']:,}, FP {metrics['FP']:,}, FN {metrics['FN']:,}, TP {metrics['TP']:,}",
            "- These are computed from the file's own Label column. If the file was part of the training or "
            "validation data, they are NOT an honest estimate of real-world performance.", ""]
    lines += [
        "## Limitations",
        "- Scores are ranking values, not probabilities; they are investigation leads, not proof of malicious activity.",
        "- The model saw only two attack types (FTP-Patator, SSH-Patator) from one day of a research dataset with "
        "machine-generated benign traffic, and failed to detect an attack type it had not seen in testing.",
        "- The dataset has known labeling and feature errors reported in published research.",
        "- Research prototype, not a production IDS. It never blocks traffic or takes action.", "",
        "## Analyst notes",
        "- Reviewed by:", "- Decision:", "- Follow-up actions:", ""]
    return "\n".join(lines)