import numpy as np


def sigmoid(z):
    """Squash any number into the range 0 to 1."""
    return 1 / (1 + np.exp(-z))


# Two features for one network flow (already scaled to small numbers)
x = np.array([0.5, 2.0])      # [bytes_scaled, packets_scaled]

# Parameters the neuron "learned" (we choose them by hand here)
w = np.array([-1.0, 1.5])     # weights
b = -0.5                      # bias

z = np.dot(w, x) + b          # weighted sum plus bias
score = sigmoid(z)

print("Weighted sum z:", z)
print("Attack score  :", round(float(score), 3))