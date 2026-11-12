import numpy as np
import pandas as pd

from cybernn.report import build_report, flagged_export, sanitize_for_csv, sha256_bytes, sha256_file

REPAIR = {"rows_affected": 0, "missing_cells": 0, "infinite_cells": 0, "non_numeric_cells": 0}
MODEL_INFO = {"model_sha256": "m" * 64, "preprocessor_sha256": "p" * 64, "n_features": 66,
              "hidden": [64, 32], "n_train_rows": 295290, "seed": 42, "best_epoch": 24}
METRICS = {"precision": 0.9, "recall": 0.8, "f1": 0.85, "fpr": 0.01, "TN": 10, "FP": 1, "FN": 2, "TP": 8}


def report_for(scores, df=None, metrics=None, file_name="flows.csv", threshold=0.5):
    scores = np.asarray(scores, dtype=float)
    if df is None:
        df = pd.DataFrame({"Destination Port": [21] * len(scores)})
    return build_report(file_name, "a" * 64, df, scores, threshold, 0.86, REPAIR, metrics, MODEL_INFO)


def test_sha256_matches_the_known_test_vector():
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_of_a_file_matches_the_hash_of_its_bytes(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"hello")
    assert sha256_file(path) == sha256_bytes(b"hello")


def test_formula_text_cells_are_neutralized_but_numbers_are_not():
    df = pd.DataFrame({"Label": ["=1+1", "@cmd", "+x", "-y", "BENIGN"], "n": [-3, 1, 2, 3, 4]})
    out = sanitize_for_csv(df)
    assert out["Label"].tolist() == ["'=1+1", "'@cmd", "'+x", "'-y", "BENIGN"]
    assert out["n"].tolist() == [-3, 1, 2, 3, 4]


def test_formula_column_names_are_neutralized():
    out = sanitize_for_csv(pd.DataFrame({"=SUM(A1)": [1], "ok": [2]}))
    assert list(out.columns) == ["'=SUM(A1)", "ok"]


def test_normal_data_is_left_untouched():
    df = pd.DataFrame({"Label": ["BENIGN", "FTP-Patator"], "n": [1.5, -2.0]})
    pd.testing.assert_frame_equal(sanitize_for_csv(df), df)


def test_flagged_export_filters_sorts_and_numbers_the_lines():
    df = pd.DataFrame({"x": [10, 11, 12, 13, 14]})
    exported, truncated = flagged_export(df, np.array([0.2, 0.95, 0.9, 0.99, 0.5]), threshold=0.9)
    assert exported["csv_line"].tolist() == [5, 3, 4]       # row index + 2, highest score first
    assert exported["x"].tolist() == [13, 11, 12]
    assert list(exported.columns[:2]) == ["csv_line", "score"]
    assert truncated is False


def test_flagged_export_reports_truncation():
    df = pd.DataFrame({"x": range(5)})
    exported, truncated = flagged_export(df, np.array([0.9] * 5), threshold=0.5, max_rows=2)
    assert len(exported) == 2 and truncated is True


def test_flagged_export_neutralizes_formulas_in_labels():
    df = pd.DataFrame({"Label": ["=1+1", "BENIGN"]})
    exported, _ = flagged_export(df, np.array([0.99, 0.99]), threshold=0.5)
    assert "'=1+1" in exported["Label"].tolist()


def test_report_contains_hashes_threshold_and_limitations():
    text = report_for([0.9, 0.1, 0.7])
    assert "a" * 64 in text and "m" * 64 in text and "p" * 64 in text
    assert "Alert threshold used: 0.50" in text
    assert "Flagged as suspicious: 2 of 3" in text
    assert "## Limitations" in text and "not probabilities" in text
    assert "never blocks traffic" in text


def test_report_escapes_backticks_in_the_file_name():
    assert "bad'name.csv" in report_for([0.9], file_name="bad`name.csv")


def test_report_shows_evaluation_only_when_metrics_exist():
    assert "## Evaluation" not in report_for([0.9, 0.1])
    with_metrics = report_for([0.9, 0.1], metrics=METRICS)
    assert "## Evaluation" in with_metrics and "NOT an honest estimate" in with_metrics


def test_report_lists_the_most_common_ports_among_flagged_flows():
    df = pd.DataFrame({"Destination Port": [21, 21, 22, 80]})
    assert "21 (2), 22 (1)" in report_for([0.9, 0.9, 0.9, 0.1], df=df)


def test_report_handles_the_case_where_nothing_is_flagged():
    text = report_for([0.1, 0.2])
    assert "Flagged as suspicious: 0 of 2" in text
    assert "Most common destination ports" not in text
