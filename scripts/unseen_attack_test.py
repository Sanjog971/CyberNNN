import argparse
import sys
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
                    help="keep Destination Port as a feature")
args = parser.parse_args()
exclude = [] if args.keep_port else ["Destination Port"]

processed = Path("data") / "processed"
train = pd.read_csv(processed / "train.csv")
val = pd.read_csv(processed / "val.csv")          # the test split is NOT loaded here
attack_types = sorted(set(train["Label"]) - {"BENIGN"})

for held_out in attack_types:
    seen = [a for a in attack_types if a != held_out]
    print(f"\n##### Train WITHOUT {held_out}; it appears only at validation time #####")

    # 1. Everything learned (including the preprocessor) comes from the reduced training set.
    train_sub = train[train["Label"] != held_out]
    pre = Preprocessor().fit(train_sub, exclude=exclude)
    X_train, y_train = pre.transform(train_sub), train_sub["is_attack"].to_numpy()

    # 2. Validation = benign flows + the held-out attack only.
    val_sub = val[val["Label"].isin(["BENIGN", held_out])]
    X_val, y_val = pre.transform(val_sub), val_sub["is_attack"].to_numpy()
    print(f"Train attack types: {seen} | val flows: {len(val_sub)} "
          f"({int(y_val.sum())} unseen-attack flows)")

    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        m = evaluate_binary(y_val, scores, threshold=0.5)
        print(f"{name:18s} caught {m['TP']}/{m['TP'] + m['FN']} unseen attacks "
              f"(recall={m['recall']:.3f})  false alarms={m['FP']} (FPR={m['fpr']:.4f})  "
              f"precision={m['precision']:.3f} avg_precision={m['average_precision']:.3f}")