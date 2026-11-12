"""Reusable training loop for the CyberNN MLP (same logic as scripts/train_cybernn.py)."""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cybernn.model import build_mlp


def train_mlp(X_train, y_train, X_val, y_val, hidden=(64, 32), batch_size=512,
              learning_rate=1e-3, max_epochs=30, patience=5, seed=42, verbose=True):
    """Train with a class-weighted loss and early stopping on validation loss.

    Returns (model with the best weights restored, history dict, best epoch).
    """
    torch.manual_seed(seed)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    n_attack = float(y_train.sum())
    n_benign = len(y_train) - n_attack
    pos_weight = torch.tensor([n_benign / n_attack], dtype=torch.float32)

    model = build_mlp(X_train.shape[1], hidden)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)

    history = {"train": [], "val": []}
    best_val, best_epoch, best_state, wait = float("inf"), 0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            loss = loss_fn(model(xb).squeeze(1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
            seen += len(yb)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv).squeeze(1), yv).item()
        history["train"].append(running / seen)
        history["val"].append(val_loss)
        if verbose:
            print(f"  epoch {epoch:2d}  train_loss={running / seen:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch