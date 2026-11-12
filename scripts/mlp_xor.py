import numpy as np
import torch
from torch import nn


def make_data(seed, n_per_cluster):
    """Four clusters: matched (benign) and mismatched (attack) feature pairs."""
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


def train(model, X, y, epochs=500, lr=0.05):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = loss_fn(model(X).squeeze(1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def accuracy(model, X, y):
    with torch.no_grad():
        pred = (torch.sigmoid(model(X).squeeze(1)) >= 0.5).float()
    return (pred == y).float().mean().item()


torch.manual_seed(42)
X_train, y_train = make_data(seed=42, n_per_cluster=100)
X_test, y_test = make_data(seed=7, n_per_cluster=50)

single = nn.Linear(2, 1)
mlp = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

for name, model in [("Single neuron", single), ("MLP (8 hidden)", mlp)]:
    final_loss = train(model, X_train, y_train)
    print(f"{name:15s} loss={final_loss:.3f}  "
          f"train acc={accuracy(model, X_train, y_train):.3f}  "
          f"test acc={accuracy(model, X_test, y_test):.3f}")