"""The CyberNN multilayer perceptron and helpers to save and load it safely."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def build_mlp(n_features, hidden=(64, 32)):
    """Input -> hidden layers with ReLU -> one output (a raw score, not yet a probability)."""
    layers = []
    previous = n_features
    for width in hidden:
        layers += [nn.Linear(previous, width), nn.ReLU()]
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def predict_scores(model, X, batch_size=8192):
    """Attack probabilities (0 to 1) for a float32 matrix X, scored in chunks."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size])
            chunks.append(torch.sigmoid(model(batch).squeeze(1)).numpy())
    return np.concatenate(chunks)


def save_model(model, path, config):
    """Save weights (.pt) and a human-readable config (.json) side by side."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_model(path):
    """Rebuild the architecture from the JSON config, then load the weights safely."""
    path = Path(path)
    with open(path.with_suffix(".json"), encoding="utf-8") as f:
        config = json.load(f)
    n_features = int(config["n_features"])
    hidden = tuple(int(w) for w in config["hidden"])
    model = build_mlp(n_features, hidden)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model, config