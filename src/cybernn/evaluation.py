"""Metrics for a binary intrusion detector (1 = attack, 0 = benign)."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def evaluate_binary(y_true, scores, threshold=0.5):
    """Return confusion-matrix counts and the main IDS metrics at one threshold.

    scores are attack probabilities between 0 and 1. A flow is flagged as an
    attack when its score is at or above the threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": float(threshold),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "average_precision": float(average_precision_score(y_true, scores)),
    }