import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to a file, no pop-up window
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(seed, n_per_cluster):
    rng = np.random.default_rng(seed)
    centers = [(1.5, 1.5, 0), (-1.5, -1.5, 0), (1.5, -1.5, 1), (-1.5, 1.5, 1)]
    X, y = [], []
    for cx, cy, label in centers:
        X.append(rng.normal([cx, cy], 0.6, size=(n_per_cluster, 2)))
        y.append(np.full(n_per_cluster, label))
    X = np.vstack(X)
    y = np.concatenate(y)
    order = rng.permutation(len(y))
    return (torch.tensor(X[order], dtype=torch.float32),
            torch.tensor(y[order], dtype=torch.float32))


def build_model():
    return nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))


def accuracy(model, X, y):
    with torch.no_grad():
        pred = (torch.sigmoid(model(X).squeeze(1)) >= 0.5).float()
    return (pred == y).float().mean().item()


torch.manual_seed(42)
X, y = make_data(seed=42, n_per_cluster=150)
n_train, n_val = int(0.70 * len(y)), int(0.15 * len(y))
X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
model = build_model()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

history = {"train": [], "val": []}
best_val, best_epoch, best_state = float("inf"), 0, None
patience, wait = 10, 0

for epoch in range(1, 101):
    model.train()
    for xb, yb in loader:
        loss = loss_fn(model(xb).squeeze(1), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = loss_fn(model(X_train).squeeze(1), y_train).item()
        val_loss = loss_fn(model(X_val).squeeze(1), y_val).item()
    history["train"].append(train_loss)
    history["val"].append(val_loss)

    if val_loss < best_val - 1e-4:                     # a real improvement
        best_val, best_epoch, wait = val_loss, epoch, 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        wait += 1

    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch {epoch:3d}  train_loss={train_loss:.3f}  val_loss={val_loss:.3f}")
    if wait >= patience:
        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
        break

model.load_state_dict(best_state)                      # go back to the best weights
print(f"Best epoch: {best_epoch}  best val_loss: {best_val:.3f}")
print("Validation accuracy:", round(accuracy(model, X_val, y_val), 3))
print("Test accuracy      :", round(accuracy(model, X_test, y_test), 3))

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
torch.save(model.state_dict(), Path("models") / "mlp_xor_best.pt")

epochs = range(1, len(history["train"]) + 1)
plt.figure(figsize=(7, 4))
plt.plot(epochs, history["train"], label="train loss")
plt.plot(epochs, history["val"], label="validation loss")
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.tight_layout()
plt.savefig(Path("reports") / "loss_curves.png", dpi=120)
print("Saved plot to reports\\loss_curves.png")