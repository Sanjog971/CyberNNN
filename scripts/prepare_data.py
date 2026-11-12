import json
import sys
from pathlib import Path

sys.path.insert(0, "src")                      # lets Python find src\cybernn

from cybernn.data import clean_flows, load_flow_csv, split_data

RAW_FILE = Path("data") / "raw" / "MachineLearningCVE" / "Tuesday-WorkingHours.pcap_ISCX.csv"
OUT_DIR = Path("data") / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)

df = load_flow_csv(RAW_FILE)
clean, report = clean_flows(df)
train, val, test = split_data(clean, seed=42)

print("Cleaning report:")
for key, value in report.items():
    print(f"  {key}: {value}")

for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"\n{name}: {len(part)} rows, attack share {part['is_attack'].mean():.4f}")
    print(part["Label"].value_counts().to_string())

with open(Path("reports") / "data_prep_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nSaved: data\\processed\\train.csv, val.csv, test.csv and reports\\data_prep_report.json")