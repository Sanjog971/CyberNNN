import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from cybernn.model import load_model, predict_scores
from cybernn.preprocessing import Preprocessor

THRESHOLD = 0.86                      # the threshold chosen on validation data
SHOW = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Flow Bytes/s", "Flow Packets/s", "Init_Win_bytes_forward", "Label"]
COMPARE = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow Bytes/s",
           "Flow Packets/s", "Init_Win_bytes_forward", "Fwd Packet Length Max"]

pre = Preprocessor.load(Path("models") / "preprocessor.json")
model, config = load_model(Path("models") / "cybernn_mlp.pt")

# VALIDATION data on purpose: the test set has been used for the final numbers, so any
# change we make because of what we see here must not be judged on it.
val = pd.read_csv(Path("data") / "processed" / "val.csv")
val["score"] = predict_scores(model, pre.transform(val))
flagged = (val["score"] >= THRESHOLD).to_numpy()
attack = (val["is_attack"] == 1).to_numpy()
show = [c for c in SHOW if c in val.columns]
compare = [c for c in COMPARE if c in val.columns]

print(f"Validation flows: {len(val)} | threshold {THRESHOLD}")
print(f"True positives: {int((flagged & attack).sum())} | false positives: {int((flagged & ~attack).sum())} "
      f"| missed attacks: {int((~flagged & attack).sum())}")

print("\n=== False positives (benign flows that were flagged) ===")
fp = val[flagged & ~attack].sort_values("score", ascending=False)
print(fp[["score"] + show].to_string() if len(fp) else "none")

print("\n=== Missed attacks by type ===")
for label in sorted(set(val.loc[attack, "Label"])):
    of_type = attack & (val["Label"] == label).to_numpy()
    print(f"{label}: missed {int((of_type & ~flagged).sum())} of {int(of_type.sum())}")

fn = val[~flagged & attack].sort_values("score")
print("\n=== Score summary of missed attacks ===")
print(fn["score"].describe().round(4).to_string() if len(fn) else "none")

print("\n=== Lowest-scoring missed attacks (first 10) ===")
print(fn[["score"] + show].head(10).to_string() if len(fn) else "none")

print("\n=== Median feature values: caught vs missed attacks of the same type ===")
attacks = val[attack].copy()
attacks["outcome"] = np.where(flagged[attack], "caught", "missed")
print(attacks.groupby(["Label", "outcome"])[compare].median().round(1).to_string())
print(attacks.groupby(["Label", "outcome"]).size().to_string())