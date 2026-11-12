
import numpy as np
import pandas as pd
import pytest

from cybernn.data import clean_flows, load_flow_csv, split_data


def make_flows(n_benign=60, n_ftp=20, n_ssh=20, seed=0):
    rng = np.random.default_rng(seed)
    n = n_benign + n_ftp + n_ssh
    return pd.DataFrame({
        "Flow Duration": rng.integers(1, 10_000, n),
        "Total Fwd Packets": rng.integers(1, 50, n),
        "Flow Bytes/s": rng.random(n) * 1000,
        "Label": ["BENIGN"] * n_benign + ["FTP-Patator"] * n_ftp + ["SSH-Patator"] * n_ssh,
    })


def test_load_flow_csv_strips_header_spaces(tmp_path):
    path = tmp_path / "flows.csv"
    path.write_text(" Flow Duration, Label\n5,BENIGN\n")
    assert list(load_flow_csv(path).columns) == ["Flow Duration", "Label"]


def test_clean_adds_binary_target_and_strips_label_spaces():
    df = make_flows(2, 1, 1)
    df["Label"] = [" BENIGN", "BENIGN ", "FTP-Patator", "SSH-Patator "]
    clean, _ = clean_flows(df)
    assert clean["Label"].tolist() == ["BENIGN", "BENIGN", "FTP-Patator", "SSH-Patator"]
    assert clean["is_attack"].tolist() == [0, 0, 1, 1]


def test_clean_removes_exact_duplicates_and_reports_the_count():
    df = make_flows()
    doubled = pd.concat([df, df.iloc[:5]], ignore_index=True)
    clean, report = clean_flows(doubled)
    assert len(clean) == len(df)
    assert report["duplicate_rows_removed"] == 5
    assert report["rows_in"] == len(df) + 5 and report["rows_out"] == len(df)


def test_clean_drops_a_copy_column_only_when_it_is_identical():
    df = make_flows()
    df["Fwd Header Length"] = df["Flow Duration"]
    df["Fwd Header Length.1"] = df["Fwd Header Length"]
    clean, report = clean_flows(df)
    assert "Fwd Header Length.1" not in clean.columns
    assert report["duplicate_columns_dropped"] == ["Fwd Header Length.1"]

    different = make_flows()
    different["Fwd Header Length"] = different["Flow Duration"]
    different["Fwd Header Length.1"] = different["Fwd Header Length"] + 1
    clean, report = clean_flows(different)
    assert "Fwd Header Length.1" in clean.columns
    assert report["duplicate_columns_dropped"] == []


def test_clean_turns_infinity_into_missing_and_reports_it():
    df = make_flows()
    df.loc[0, "Flow Bytes/s"] = np.inf
    clean, report = clean_flows(df)
    assert not np.isinf(clean["Flow Bytes/s"]).any()
    assert clean["Flow Bytes/s"].isna().sum() == 1
    assert report["infinite_values_replaced"] == 1
    assert report["rows_with_missing_values"] == 1


def test_clean_requires_a_label_column():
    with pytest.raises(ValueError, match="Label"):
        clean_flows(pd.DataFrame({"Flow Duration": [1, 2]}))


def test_split_partitions_the_data_without_overlap():
    clean, _ = clean_flows(make_flows(600, 100, 100))
    train, val, test = split_data(clean)
    parts = [set(train.index), set(val.index), set(test.index)]
    assert parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2]) and parts[1].isdisjoint(parts[2])
    assert parts[0] | parts[1] | parts[2] == set(clean.index)
    assert abs(len(test) - 0.15 * len(clean)) <= 1
    assert abs(len(val) - 0.15 * len(clean)) <= 1


def test_split_keeps_class_shares_in_every_part():
    clean, _ = clean_flows(make_flows(600, 100, 100))
    overall = clean["Label"].value_counts(normalize=True)
    for part in split_data(clean):
        shares = part["Label"].value_counts(normalize=True)
        for label in overall.index:
            assert abs(shares[label] - overall[label]) < 0.02


def test_split_is_repeatable_for_the_same_seed_and_changes_with_another():
    clean, _ = clean_flows(make_flows(600, 100, 100))
    a = split_data(clean, seed=1)
    b = split_data(clean, seed=1)
    c = split_data(clean, seed=2)
    assert list(a[2].index) == list(b[2].index)
    assert list(a[2].index) != list(c[2].index)
