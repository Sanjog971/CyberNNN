import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from cybernn.evaluation import evaluate_binary
from cybernn.model import load_model, predict_scores
from cybernn.preprocessing import Preprocessor

pre = Preprocessor.load(Path("models") / "preprocessor.json")
processed = Path("data") / "processed"
train = pd.read_csv(processed / "train.csv")
val = pd.read_csv(processed / "val.csv")
test = pd.read_csv(processed / "test.csv")        # opened for the FIRST time here

X_train, y_train = pre.transform(train), train["is_attack"].to_numpy()
X_val, y_val = pre.transform(val), val["is_attack"].to_numpy()
X_test, y_test = pre.transform(test), test["is_attack"].to_numpy()
attack_types = sorted(set(test["Label"]) - {"BENIGN"})
print(f"Test flows: {len(y_test)} | attacks: {int(y_test.sum())} | benign: {int((1 - y_test).sum())}")


def best_threshold(y, scores):
    """Threshold with the best F1 on VALIDATION data (never on test data)."""
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda t: evaluate_binary(y, scores, t)["f1"]))


# Baselines are retrained on the training split only; the MLP is loaded from disk.
lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                            n_jobs=-1, random_state=42).fit(X_train, y_train)
mlp, config = load_model(Path("models") / "cybernn_mlp.pt")

scorers = {
    "LogisticRegression": lambda X: lr.predict_proba(X)[:, 1],
    "RandomForest": lambda X: rf.predict_proba(X)[:, 1],
    "CyberNN MLP": lambda X: predict_scores(mlp, X),
}

results = {}
for name, score_fn in scorers.items():
    tuned = best_threshold(y_val, score_fn(X_val))       # chosen on validation
    test_scores = score_fn(X_test)
    results[name] = {}
    print(f"\n=== {name} (validation-selected threshold: {tuned:.2f}) ===")
    for label, t in [("0.5", 0.5), (f"{tuned:.2f} (tuned on val)", tuned)]:
        m = evaluate_binary(y_test, test_scores, t)
        flagged = test_scores >= t
        per_attack = {a: float(flagged[(test["Label"] == a).to_numpy()].mean()) for a in attack_types}
        results[name][label] = {**m, "recall_by_attack": per_attack}
        print(f"threshold {label}: TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")
        print(f"   precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f} "
              f"FPR={m['fpr']:.4f} avg_precision={m['average_precision']:.3f}")
        print("   recall by attack: " + ", ".join(f"{a} {r:.3f}" for a, r in per_attack.items()))

print("\nMLP threshold stored in its config file:", config["threshold_selected_on_validation"])

# End-to-end latency for the MLP: preprocessing + scoring, from a DataFrame.
t0 = time.perf_counter()
_ = predict_scores(mlp, pre.transform(test))
print(f"MLP end-to-end (preprocess + score): "
      f"{(time.perf_counter() - t0) / len(test) * 1000 * 1000:.1f} ms per 1000 flows")

# Confusion-matrix picture for the MLP at its validation-selected threshold.
mlp_tuned = [k for k in results["CyberNN MLP"] if "tuned" in k][0]
m = results["CyberNN MLP"][mlp_tuned]
matrix = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
Path("reports").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(np.log10(matrix + 1), cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=12)
ax.set_xticks([0, 1], ["Predicted\nBENIGN", "Predicted\nATTACK"])
ax.set_yticks([0, 1], ["Actual\nBENIGN", "Actual\nATTACK"])
ax.set_title(f"CyberNN MLP on test (threshold {mlp_tuned.split()[0]})")
plt.tight_layout()
plt.savefig(Path("reports") / "confusion_matrix_test.png", dpi=120)

with open(Path("reports") / "final_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved reports\\confusion_matrix_test.png and reports\\final_test_results.json")