import json

import numpy as np

from cybernn.evaluation import evaluate_binary


def test_confusion_counts_and_metrics():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1]
    m = evaluate_binary(y_true, scores, threshold=0.5)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (2, 1, 1, 4)
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["f1"] - 2 / 3) < 1e-9
    assert abs(m["fpr"] - 1 / 5) < 1e-9
    assert 0.0 <= m["average_precision"] <= 1.0


def test_threshold_is_inclusive():
    m = evaluate_binary([1], [0.5], threshold=0.5)
    assert m["TP"] == 1 and m["FN"] == 0


def test_no_predicted_attacks_gives_zero_precision_without_error():
    m = evaluate_binary([1, 0], [0.1, 0.1], threshold=0.5)
    assert m["TP"] == 0 and m["FP"] == 0
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_perfect_ranking_has_average_precision_one():
    m = evaluate_binary([1, 0, 0], [0.9, 0.2, 0.1], threshold=0.5)
    assert m["average_precision"] == 1.0


def test_lowering_the_threshold_never_reduces_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    low = evaluate_binary(y, scores, threshold=0.3)
    high = evaluate_binary(y, scores, threshold=0.7)
    assert low["recall"] >= high["recall"]
    assert low["fpr"] >= high["fpr"]


def test_result_can_be_saved_as_json():
    m = evaluate_binary([1, 0, 1, 0], [0.9, 0.1, 0.4, 0.6], threshold=0.5)
    json.dumps(m)          # raises TypeError if numpy numbers leak into the dictionary
