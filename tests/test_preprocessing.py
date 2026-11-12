import json

import numpy as np
import pandas as pd
import pytest

from cybernn.preprocessing import Preprocessor, signed_log1p


def make_train(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Destination Port": rng.choice([21, 22, 80, 443], n),
        "Flow Duration": rng.integers(0, 5_000_000, n).astype(float),
        "Flow Bytes/s": rng.random(n) * 10_000,
        "Constant Flag": np.zeros(n),
        "Label": ["BENIGN"] * (n - 20) + ["FTP-Patator"] * 20,
        "is_attack": [0] * (n - 20) + [1] * 20,
    })


def test_signed_log_keeps_the_sign():
    out = signed_log1p(np.array([-3.0, 0.0, 3.0]))
    assert np.allclose(out, [-np.log(4), 0.0, np.log(4)])


def test_fit_picks_numeric_features_and_drops_constant_and_excluded_columns():
    pre = Preprocessor().fit(make_train(), exclude=["Destination Port"])
    assert pre.feature_names == ["Flow Duration", "Flow Bytes/s"]
    assert pre.constant_columns_dropped == ["Constant Flag"]
    assert pre.excluded_columns == ["Destination Port"]
    assert "Label" not in pre.feature_names and "is_attack" not in pre.feature_names


def test_training_data_comes_out_standardized():
    train = make_train()
    x = Preprocessor().fit(train).transform(train)
    assert np.abs(x.mean(axis=0)).max() < 1e-4
    assert np.abs(x.std(axis=0) - 1).max() < 1e-3


def test_transform_returns_float32_with_one_column_per_feature():
    train = make_train()
    pre = Preprocessor().fit(train, exclude=["Destination Port"])
    x = pre.transform(train)
    assert x.shape == (len(train), 2) and x.dtype == np.float32


def test_infinite_and_missing_values_are_filled_with_the_training_median():
    train = make_train()
    pre = Preprocessor().fit(train)
    broken = train.head(3).copy()
    broken["Flow Bytes/s"] = [np.inf, np.nan, -np.inf]
    filled = train.head(3).copy()
    filled["Flow Bytes/s"] = train["Flow Bytes/s"].median()
    assert np.isfinite(pre.transform(broken)).all()
    assert np.allclose(pre.transform(broken), pre.transform(filled))


def test_non_numeric_text_is_treated_as_missing():
    train = make_train()
    pre = Preprocessor().fit(train)
    texty = train.head(2).copy()
    texty["Flow Duration"] = texty["Flow Duration"].astype(object)
    texty.loc[texty.index[0], "Flow Duration"] = "oops"
    filled = train.head(2).copy()
    filled.loc[filled.index[0], "Flow Duration"] = train["Flow Duration"].median()
    assert np.allclose(pre.transform(texty), pre.transform(filled))


def test_missing_required_column_gives_a_clear_error():
    train = make_train()
    pre = Preprocessor().fit(train)
    with pytest.raises(ValueError, match="Missing 1 required column"):
        pre.transform(train.drop(columns=["Flow Duration"]))


def test_column_names_with_extra_spaces_are_accepted():
    train = make_train()
    pre = Preprocessor().fit(train)
    spaced = train.rename(columns=lambda c: " " + c)
    assert np.array_equal(pre.transform(train), pre.transform(spaced))


def test_a_flows_result_does_not_depend_on_the_other_rows():
    train = make_train()
    pre = Preprocessor().fit(train)
    whole = pre.transform(train)
    single = pre.transform(train.iloc[[7]])
    assert np.allclose(whole[7], single[0])


def test_statistics_come_from_training_data_only():
    train = make_train()
    pre = Preprocessor().fit(train)
    shifted = train.copy()
    shifted["Flow Duration"] = shifted["Flow Duration"] * 1000 + 1_000_000
    x = pre.transform(shifted)
    column = pre.feature_names.index("Flow Duration")
    assert x[:, column].mean() > 2          # would be about 0 if transform re-fitted on its own input


def test_unfitted_preprocessor_refuses_to_transform():
    with pytest.raises(RuntimeError, match="not fitted"):
        Preprocessor().transform(make_train())


def test_save_and_load_give_identical_output(tmp_path):
    train = make_train()
    pre = Preprocessor().fit(train, exclude=["Destination Port"])
    path = tmp_path / "pre.json"
    pre.save(path)
    loaded = Preprocessor.load(path)
    assert np.array_equal(pre.transform(train), loaded.transform(train))
    assert json.loads(path.read_text())["feature_names"] == pre.feature_names


def test_loading_rejects_a_wrong_schema_version():
    data = Preprocessor().fit(make_train()).to_dict()
    data["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        Preprocessor.from_dict(data)


def test_loading_rejects_non_positive_scales():
    data = Preprocessor().fit(make_train()).to_dict()
    data["scales"][0] = 0
    with pytest.raises(ValueError, match="positive"):
        Preprocessor.from_dict(data)


def test_loading_rejects_wrong_lengths_and_non_finite_numbers():
    good = Preprocessor().fit(make_train()).to_dict()
    short = dict(good, means=good["means"][:-1])
    with pytest.raises(ValueError, match="Invalid values or length"):
        Preprocessor.from_dict(short)
    nan = dict(good, medians=[float("nan")] + good["medians"][1:])
    with pytest.raises(ValueError, match="Invalid values or length"):
        Preprocessor.from_dict(nan)
