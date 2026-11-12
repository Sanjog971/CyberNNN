import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from cybernn.preprocessing import Preprocessor

parser = argparse.ArgumentParser()
parser.add_argument("--keep-port", action="store_true",
                    help="keep Destination Port as a feature (for the shortcut experiment)")
args = parser.parse_args()

exclude = [] if args.keep_port else ["Destination Port"]
out_name = "preprocessor_with_port.json" if args.keep_port else "preprocessor.json"
out_path = Path("models") / out_name

processed = Path("data") / "processed"
train = pd.read_csv(processed / "train.csv")
val = pd.read_csv(processed / "val.csv")
test = pd.read_csv(processed / "test.csv")

# What do the raw training features look like? (this informs the scaling choice)
numeric = train.select_dtypes(include="number").drop(columns=["is_attack"])
has_negative = [c for c in numeric.columns if (numeric[c] < 0).any()]
huge = [c for c in numeric.columns if numeric[c].max() > 1_000_000]
print("Numeric columns in train:", numeric.shape[1])
print("Columns containing negative values:", len(has_negative), has_negative[:8])
print("Columns with a maximum above 1,000,000:", len(huge))

pre = Preprocessor().fit(train, exclude=exclude)
print("\nExcluded on purpose:", pre.excluded_columns)
print("Constant columns dropped:", pre.constant_columns_dropped)
print("Features kept:", len(pre.feature_names))

for name, part in [("train", train), ("val", val), ("test", test)]:
    x = pre.transform(part)
    print(f"\n{name}: matrix {x.shape}, dtype {x.dtype}, NaN count {int(np.isnan(x).sum())}")
    print(f"  largest |column mean| = {np.abs(x.mean(axis=0)).max():.4f}")
    print(f"  column std range      = {x.std(axis=0).min():.3f} to {x.std(axis=0).max():.3f}")

pre.save(out_path)
reloaded = Preprocessor.load(out_path)
same = np.array_equal(pre.transform(test), reloaded.transform(test))
print(f"\nSaved: {out_path}")
print("Reloaded preprocessor gives identical output on the test split:", same)                                                                         