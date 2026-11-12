import torch
from torch import nn

X = torch.tensor([
    [0.9, 0.2], [0.8, 0.3], [1.0, 0.1], [0.7, 0.4],   # benign
    [0.1, 0.9], [0.2, 0.8], [0.3, 1.0], [0.2, 0.7],   # attack
], dtype=torch.float32)
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)

model = nn.Linear(2, 1)                 # 2 inputs -> 1 output (one neuron)
with torch.no_grad():                   # start at zero, like the NumPy version
    model.weight.zero_()
    model.bias.zero_()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(1, 501):
    logits = model(X).squeeze(1)        # forward pass: shape (8, 1) -> (8,)
    loss = loss_fn(logits, y)           # average loss over the 8 flows

    optimizer.zero_grad()               # clear old gradients
    loss.backward()                     # compute new gradients
    optimizer.step()                    # update weights and bias

    if epoch in (1, 10, 50, 100, 250, 500):
        w = model.weight.detach().numpy()[0].round(3)
        b = model.bias.item()
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}  w={w}  b={b:.3f}")

with torch.no_grad():
    scores = torch.sigmoid(model(X).squeeze(1))
print("Final scores:", scores.numpy().round(2))
print("Predicted   :", (scores >= 0.5).int().numpy())
print("True labels :", y.int().numpy())