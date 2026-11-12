import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    """Keep positive values, turn negatives into 0."""
    return np.maximum(0, z)


x = np.array([0.5, 2.0])               # one flow: [bytes_scaled, packets_scaled]

# Hidden layer: 3 neurons, each with 2 weights (one per input)
W1 = np.array([[ 1.0,  0.5],
               [-1.0,  1.0],
               [ 2.0, -1.0]])
b1 = np.array([0.0, -1.0, 0.5])

# Output layer: 1 neuron with 3 weights (one per hidden neuron)
w2 = np.array([1.0, -2.0, 0.5])
b2 = -0.2

z1 = W1 @ x + b1                        # three weighted sums
h = relu(z1)                            # hidden activations
z2 = np.dot(w2, h) + b2                 # output weighted sum
score = sigmoid(z2)

print("Hidden sums z1  :", z1)
print("Hidden outputs h:", h)
print("Output sum z2   :", z2)
print("Attack score    :", round(float(score), 3))