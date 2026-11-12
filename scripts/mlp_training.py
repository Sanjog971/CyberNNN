from pathlib import Path

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
X, y = make_data(seed=42, n_per_cluster=150)          # 600 rows, already shuffled

n_train = int(0.70 * len(y))                           # 420
n_val = int(0.15 * len(y))                             # 90
X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
print("Rows  train/val/test:", len(y_train), len(y_val), len(y_test))

loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

model = build_model()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

for epoch in range(1, 61):
    model.train()
    for xb, yb in loader:                              # one mini-batch at a time
        loss = loss_fn(model(xb).squeeze(1), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = loss_fn(model(X_train).squeeze(1), y_train).item()
        val_loss = loss_fn(model(X_val).squeeze(1), y_val).item()
    if epoch in (1, 5, 10, 20, 40, 60):
        print(f"epoch {epoch:2d}  train_loss={train_loss:.3f}  val_loss={val_loss:.3f}")

print("Validation accuracy:", round(accuracy(model, X_val, y_val), 3))
print("Test accuracy      :", round(accuracy(model, X_test, y_test), 3))

# Save the learned weights, then load them into a fresh model
model_path = Path("models") / "mlp_xor.pt"
model_path.parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), model_path)

reloaded = build_model()
reloaded.load_state_dict(torch.load(model_path, weights_only=True))
reloaded.eval()

with torch.no_grad():
    same = torch.allclose(model(X_test), reloaded(X_test))
print("Saved to:", model_path)
print("Reloaded model gives identical outputs:", same)