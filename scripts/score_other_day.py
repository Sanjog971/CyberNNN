import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

from cybernn.data import clean_flows, load_flow_csv
from cybernn.model import load_model, predict_scores
from cybernn.preprocessing import Preprocessor

parser = argparse.ArgumentParser(description="Score a CIC-IDS2017 day file the model has never seen.")
parser.add_argument("csv_path", help="path to the day's CSV file")
args = parser.parse_args()

df, report = clean_flows(load_flow_csv(args.csv_path))
print("Cleaning report:")
for key, value in report.items():
    print(f"  {key}: {value}")

pre = Preprocessor.load(Path("models") / "preprocessor.json")
model, config = load_model(Path("models") / "cybernn_mlp.pt")
df["score"] = predict_scores(model, pre.transform(df))

print(f"\nFlows scored: {len(df)}")
rows = []
for label, group in df.groupby("Label"):
    row = {"Label": label, "flows": len(group), "median_score": round(float(group["score"].median()), 4)}
    for t in (0.86, 0.5):
        n = int((group["score"] >= t).sum())
        row[f"flagged@{t}"] = n
        row[f"rate@{t}"] = round(n / len(group), 4)
    rows.append(row)
print("\nShare of flows flagged, by label (for attack labels this is recall; for BENIGN it is the false-alarm rate):")
print(pd.DataFrame(rows).to_string(index=False))

benign = df[df["Label"] == "BENIGN"]
if len(benign):
    false_alarms = benign[benign["score"] >= 0.86]
    print(f"\nFalse alarms among benign flows at 0.86: {len(false_alarms)} of {len(benign)} "
          f"({len(false_alarms) / len(benign) * 1000:.2f} per 1,000)")
    if len(false_alarms):
        print("Most common destination ports among those false alarms:")
        print(false_alarms["Destination Port"].value_counts().head(8).to_string())