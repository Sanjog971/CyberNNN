import copy
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "src")

from cybernn.evaluation import evaluate_binary
from cybernn.model import build_mlp, load_model, predict_scores, save_model
from cybernn.preprocessing import Preprocessor

SEED = 42
HIDDEN = (64, 32)
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5

torch.manual_seed(SEED)
np.random.seed(SEED)

pre = Preprocessor.load(Path("models") / "preprocessor.json")
processed = Path("data") / "processed"
train = pd.read_csv(processed / "train.csv")
val = pd.read_csv(processed / "val.csv")          # the test split is NOT loaded here

X_train, y_train = pre.transform(train), train["is_attack"].to_numpy(dtype=np.float32)
X_val, y_val = pre.transform(val), val["is_attack"].to_numpy(dtype=np.float32)
attack_types = sorted(set(val["Label"]) - {"BENIGN"})

n_attack = float(y_train.sum())
n_benign = len(y_train) - n_attack
pos_weight = torch.tensor([n_benign / n_attack], dtype=torch.float32)
print(f"Train rows: {len(y_train)} | features: {X_train.shape[1]} | pos_weight: {pos_weight.item():.1f}")

model = build_mlp(X_train.shape[1], HIDDEN)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)

history = {"train": [], "val": []}
best_val, best_epoch, best_state, wait = float("inf"), 0, None, 0
start = time.perf_counter()

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running, seen = 0.0, 0
    for xb, yb in loader:
        loss = loss_fn(model(xb).squeeze(1), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * len(yb)
        seen += len(yb)
    train_loss = running / seen

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(Xv).squeeze(1), yv).item()
    val_ap = average_precision_score(y_val, predict_scores(model, X_val))
    history["train"].append(train_loss)
    history["val"].append(val_loss)
    print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_AP={val_ap:.4f}")

    if val_loss < best_val - 1e-4:
        best_val, best_epoch, wait = val_loss, epoch, 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"Training time: {time.perf_counter() - start:.1f} s | best epoch: {best_epoch}")
model.load_state_dict(best_state)

# ---- Evaluate on validation (threshold 0.5, then the threshold that maximizes validation F1) ----
t0 = time.perf_counter()
scores = predict_scores(model, X_val)
latency = (time.perf_counter() - t0) / len(y_val) * 1000 * 1000
thresholds = np.linspace(0.05, 0.95, 91)
best_threshold = float(max(thresholds, key=lambda t: evaluate_binary(y_val, scores, t)["f1"]))

for label, t in [("threshold 0.5", 0.5), (f"best-F1 threshold {best_threshold:.2f}", best_threshold)]:
    m = evaluate_binary(y_val, scores, t)
    print(f"\nVal at {label}: TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")
    print(f"  precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f} "
          f"FPR={m['fpr']:.4f} avg_precision={m['average_precision']:.3f}")
    flagged = scores >= t
    for attack in attack_types:
        mask = (val["Label"] == attack).to_numpy()
        print(f"  recall on {attack}: {flagged[mask].mean():.3f} ({int(mask.sum())} flows)")
print(f"\nInference: {latency:.1f} ms per 1000 flows (batch scoring)")

# ---- Save the model, reload it, and check the outputs are identical ----
config = {
    "n_features": int(X_train.shape[1]),
    "hidden": list(HIDDEN),
    "preprocessor": "models/preprocessor.json",
    "seed": SEED,
    "best_epoch": best_epoch,
    "pos_weight": float(pos_weight.item()),
    "threshold_selected_on_validation": best_threshold,
}
model_path = Path("models") / "cybernn_mlp.pt"
save_model(model, model_path, config)
reloaded, _ = load_model(model_path)
same = np.allclose(scores, predict_scores(reloaded, X_val), atol=1e-6)
print(f"Saved {model_path} and its .json config | reload gives the same scores: {same}")

# ---- Loss curves ----
Path("reports").mkdir(exist_ok=True)
epochs = range(1, len(history["train"]) + 1)
plt.figure(figsize=(7, 4))
plt.plot(epochs, history["train"], label="train loss")
plt.plot(epochs, history["val"], label="validation loss")
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
plt.xlabel("Epoch")
plt.ylabel("Weighted loss")
plt.title("CyberNN training and validation loss")
plt.legend()
plt.tight_layout()
plt.savefig(Path("reports") / "cybernn_loss_curves.png", dpi=120)
print("Saved reports\\cybernn_loss_curves.png")