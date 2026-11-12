import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from cybernn.evaluation import evaluate_binary
from cybernn.preprocessing import Preprocessor

parser = argparse.ArgumentParser()
parser.add_argument("--keep-port", action="store_true",
                    help="use the preprocessor that keeps Destination Port")
args = parser.parse_args()

pre_path = Path("models") / ("preprocessor_with_port.json" if args.keep_port else "preprocessor.json")
if not pre_path.exists():
    sys.exit(f"Missing {pre_path}. Run scripts\\build_preprocessor.py"
             + (" --keep-port" if args.keep_port else "") + " first.")

pre = Preprocessor.load(pre_path)
print("Preprocessor:", pre_path, "| features:", len(pre.feature_names))

processed = Path("data") / "processed"
train = pd.read_csv(processed / "train.csv")
val = pd.read_csv(processed / "val.csv")          # the test split is NOT loaded here

X_train, y_train = pre.transform(train), train["is_attack"].to_numpy()
X_val, y_val = pre.transform(val), val["is_attack"].to_numpy()
attack_types = sorted(set(val["Label"]) - {"BENIGN"})

models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
}

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"\n=== {name} (trained in {time.perf_counter() - start:.1f} s) ===")

    for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        t0 = time.perf_counter()
        scores = model.predict_proba(X)[:, 1]
        elapsed = time.perf_counter() - t0
        m = evaluate_binary(y, scores, threshold=0.5)
        print(f"{split:5s} TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}  "
              f"precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"F1={m['f1']:.3f} FPR={m['fpr']:.4f} avg_precision={m['average_precision']:.3f}")
        if split == "val":
            print(f"      inference: {elapsed / len(y) * 1000 * 1000:.1f} ms per 1000 flows")
            flagged = scores >= 0.5
            for label in attack_types:
                mask = (val["Label"] == label).to_numpy()
                print(f"      recall on {label}: {flagged[mask].mean():.3f} ({int(mask.sum())} flows)")