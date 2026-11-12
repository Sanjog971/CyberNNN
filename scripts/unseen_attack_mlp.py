import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

from cybernn.evaluation import evaluate_binary
from cybernn.model import predict_scores
from cybernn.preprocessing import Preprocessor
from cybernn.training import train_mlp

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
    print(f"\n##### CyberNN MLP trained WITHOUT {held_out} #####")

    train_sub = train[train["Label"] != held_out]
    pre = Preprocessor().fit(train_sub, exclude=exclude)
    X_train, y_train = pre.transform(train_sub), train_sub["is_attack"].to_numpy()

    # Early stopping must NOT see the held-out attack, so it uses validation rows without it.
    val_stop = val[val["Label"] != held_out]
    X_stop, y_stop = pre.transform(val_stop), val_stop["is_attack"].to_numpy()

    model, history, best_epoch = train_mlp(X_train, y_train, X_stop, y_stop, verbose=False)
    print(f"Trained {len(history['train'])} epochs, best epoch {best_epoch}")

    # Evaluation = benign validation flows + the unseen attack only.
    val_eval = val[val["Label"].isin(["BENIGN", held_out])]
    X_eval, y_eval = pre.transform(val_eval), val_eval["is_attack"].to_numpy()
    scores = predict_scores(model, X_eval)
    m = evaluate_binary(y_eval, scores, threshold=0.5)
    print(f"CyberNN MLP        caught {m['TP']}/{m['TP'] + m['FN']} unseen attacks "
          f"(recall={m['recall']:.3f})  false alarms={m['FP']} (FPR={m['fpr']:.4f})  "
          f"precision={m['precision']:.3f} avg_precision={m['average_precision']:.3f}")