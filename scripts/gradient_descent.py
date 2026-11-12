import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Tiny artificial dataset: [bytes_scaled, packets_scaled]
X = np.array([
    [0.9, 0.2], [0.8, 0.3], [1.0, 0.1], [0.7, 0.4],   # benign
    [0.1, 0.9], [0.2, 0.8], [0.3, 1.0], [0.2, 0.7],   # attack
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

w = np.zeros(2)          # start with all weights at zero
b = 0.0
learning_rate = 0.5

for epoch in range(1, 501):
    p = sigmoid(X @ w + b)                              # forward pass
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    grad_w = X.T @ (p - y) / len(y)                     # slope of loss for each weight
    grad_b = np.mean(p - y)                             # slope of loss for the bias

    w = w - learning_rate * grad_w                      # step downhill
    b = b - learning_rate * grad_b

    if epoch in (1, 10, 50, 100, 250, 500):
        print(f"epoch {epoch:3d}  loss={loss:.4f}  w={np.round(w, 3)}  b={b:.3f}")

final = sigmoid(X @ w + b)
print("Final scores:", np.round(final, 2))
print("Predicted   :", (final >= 0.5).astype(int))
print("True labels :", y)