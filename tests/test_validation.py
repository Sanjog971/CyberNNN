import numpy as np
import pandas as pd
import pytest

from cybernn.validation import UploadError, check_schema, ground_truth, read_upload, repair_report


def test_empty_file_is_rejected():
    with pytest.raises(UploadError, match="empty"):
        read_upload(b"")


def test_header_only_file_is_rejected():
    with pytest.raises(UploadError, match="no data rows"):
        read_upload(b"a,b,c\n")


def test_unparseable_file_is_rejected():
    with pytest.raises(UploadError, match="could not be parsed"):
        read_upload(b"hello\nthis is not a csv\n1,2,3,4,5\n")


def test_row_limit_is_enforced():
    with pytest.raises(UploadError, match="limit"):
        read_upload(b"a,b\n1,2\n3,4\n5,6\n", max_rows=2)


def test_column_names_are_stripped():
    df = read_upload(b" Flow Duration , Label\n5,BENIGN\n")
    assert list(df.columns) == ["Flow Duration", "Label"]


def test_latin1_file_is_accepted():
    df = read_upload("Label,note\nBENIGN,caf\xe9\n".encode("latin-1"))
    assert len(df) == 1


def test_check_schema_accepts_complete_files_and_lists_missing_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    check_schema(df, ["a", "b"])                       # no error expected
    with pytest.raises(UploadError, match="missing 2 of the 3"):
        check_schema(df, ["a", "x", "y"])


def test_repair_report_counts_each_kind_of_bad_cell():
    df = pd.DataFrame({"x": [1.0, np.nan, np.inf, 4.0], "y": ["1", "2", "3", "oops"]})
    report = repair_report(df, ["x", "y"])
    assert report["missing_cells"] == 1
    assert report["infinite_cells"] == 1
    assert report["non_numeric_cells"] == 1
    assert report["rows_affected"] == 3


def test_repair_report_is_zero_for_clean_data():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3, 4]})
    report = repair_report(df, ["x", "y"])
    assert report == {"non_numeric_cells": 0, "missing_cells": 0, "infinite_cells": 0, "rows_affected": 0}


def test_ground_truth_from_label_column_ignores_spaces():
    df = pd.DataFrame({"Label": ["BENIGN", " FTP-Patator", "DDoS", " BENIGN"]})
    assert ground_truth(df).tolist() == [0, 1, 1, 0]


def test_ground_truth_from_is_attack_column():
    assert ground_truth(pd.DataFrame({"is_attack": [0, 1]})).tolist() == [0, 1]


def test_ground_truth_is_none_without_labels():
    assert ground_truth(pd.DataFrame({"x": [1]})) is None
