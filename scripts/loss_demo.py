import numpy as np


def bce(y, p):
    """Binary cross-entropy loss for one flow."""
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


print("True label = 1 (real attack)")
for p in [0.99, 0.9, 0.5, 0.1, 0.01]:
    print(f"  score={p:<5} loss={bce(1, p):.3f}")

print("True label = 0 (benign)")
for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
    print(f"  score={p:<5} loss={bce(0, p):.3f}")

scores = np.array([0.9, 0.8, 0.1, 0.2])
labels = np.array([1, 1, 0, 0])
print("Average loss, good model:", round(float(bce(labels, scores).mean()), 3))
scores_bad = np.array([0.4, 0.3, 0.6, 0.7])
print("Average loss, bad model :", round(float(bce(labels, scores_bad).mean()), 3))