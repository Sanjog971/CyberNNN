import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cybernn.model import build_mlp, load_model, predict_scores, save_model
from cybernn.preprocessing import Preprocessor
from cybernn.training import train_mlp

ROOT = Path(__file__).resolve().parent.parent
CALLED = []


def _marker():
    CALLED.append("ran")
    return 1


class Marker:
    """A harmless stand-in for a malicious file: loading it unsafely calls _marker()."""

    def __reduce__(self):
        return (_marker, ())


def make_flows(n_benign, n_attack, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_benign + n_attack, 4)).astype(np.float32)
    y = np.array([0] * n_benign + [1] * n_attack, dtype=np.float32)
    X[y == 1] += 3.0                       # attacks sit far away from benign flows
    return X, y


def test_mlp_has_the_requested_shape_and_parameter_count():
    model = build_mlp(66, (64, 32))
    assert model(torch.zeros(5, 66)).shape == (5, 1)
    assert sum(p.numel() for p in model.parameters()) == 66 * 64 + 64 + 64 * 32 + 32 + 32 + 1   # 6,401


def test_scores_are_probabilities_in_the_unit_interval():
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32) * 10
    scores = predict_scores(build_mlp(4, (8,)), X)
    assert scores.shape == (50,) and scores.dtype == np.float32
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_chunked_scoring_matches_one_big_batch():
    torch.manual_seed(0)
    model = build_mlp(4, (8,))
    X = np.random.default_rng(1).normal(size=(50, 4)).astype(np.float32)
    assert np.allclose(predict_scores(model, X, batch_size=7), predict_scores(model, X, batch_size=10_000))


def test_scoring_does_not_change_the_model_weights():
    torch.manual_seed(0)
    model = build_mlp(4, (8,))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    predict_scores(model, np.zeros((10, 4), dtype=np.float32))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_save_and_load_give_identical_scores(tmp_path):
    torch.manual_seed(0)
    model = build_mlp(4, (8,))
    path = tmp_path / "m.pt"
    save_model(model, path, {"n_features": 4, "hidden": [8]})
    assert path.with_suffix(".json").exists()
    loaded, config = load_model(path)
    X = np.random.default_rng(2).normal(size=(20, 4)).astype(np.float32)
    assert np.allclose(predict_scores(model, X), predict_scores(loaded, X))
    assert config["hidden"] == [8]


def test_loading_refuses_files_that_are_not_plain_weights(tmp_path):
    CALLED.clear()
    path = tmp_path / "m.pt"
    torch.save(Marker(), path)
    path.with_suffix(".json").write_text(json.dumps({"n_features": 4, "hidden": [8]}))
    with pytest.raises(Exception):
        load_model(path)
    assert CALLED == []                    # the code inside the file never ran


def test_unsafe_loading_would_have_run_the_code(tmp_path):
    CALLED.clear()
    path = tmp_path / "m.pt"
    torch.save(Marker(), path)
    torch.load(path, weights_only=False)   # the unsafe way, shown only as a control
    assert CALLED == ["ran"]


def test_training_lowers_the_loss_and_learns_an_easy_pattern():
    X_train, y_train = make_flows(200, 25, seed=0)
    X_val, y_val = make_flows(50, 10, seed=1)
    model, history, best_epoch = train_mlp(X_train, y_train, X_val, y_val, hidden=(8,),
                                           batch_size=32, max_epochs=15, patience=5, verbose=False)
    assert 1 <= best_epoch <= len(history["train"]) <= 15
    assert len(history["val"]) == len(history["train"])
    assert history["train"][-1] < history["train"][0]
    scores = predict_scores(model, X_val)
    assert scores[y_val == 1].mean() > scores[y_val == 0].mean() + 0.3


def test_training_is_repeatable_for_the_same_seed():
    X_train, y_train = make_flows(100, 15, seed=0)
    X_val, y_val = make_flows(30, 8, seed=1)
    runs = []
    for _ in range(2):
        model, _, _ = train_mlp(X_train, y_train, X_val, y_val, hidden=(8,), batch_size=32,
                                max_epochs=5, patience=5, seed=3, verbose=False)
        runs.append(predict_scores(model, X_val))
    assert np.allclose(runs[0], runs[1], atol=1e-5)


def test_saved_model_and_preprocessor_agree_on_the_number_of_features():
    model_path = ROOT / "models" / "cybernn_mlp.pt"
    pre_path = ROOT / "models" / "preprocessor.json"
    if not (model_path.exists() and pre_path.exists()):
        pytest.skip("trained artifacts are not present")
    pre = Preprocessor.load(pre_path)
    model, config = load_model(model_path)
    assert config["n_features"] == len(pre.feature_names)
    scores = predict_scores(model, np.zeros((3, config["n_features"]), dtype=np.float32))
    assert scores.shape == (3,) and 0.0 <= scores.min() and scores.max() <= 1.0
    assert 0.0 < config["threshold_selected_on_validation"] < 1.0
