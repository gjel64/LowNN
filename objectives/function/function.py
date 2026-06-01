import torch
import torch.nn as nn
import numpy
import math
import random
import matplotlib.pyplot as plt

def f(x):
  return math.cos(3*x + 5*(x**2))

DATA_X = []
DATA_Y = []

for i in (range(-100, 100, 1)):
  ii = i / 100
  DATA_X.append(ii)
  DATA_Y.append(f(ii))

plt.plot(DATA_X, DATA_Y, label="f(x)")
plt.plot()

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("F function")
plt.legend()
plt.show()

X = torch.tensor(DATA_X, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(DATA_Y, dtype=torch.float32).unsqueeze(1)

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(1, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 1)
    )
  
  def forward(self, x):
    return self.layers(x)

model = MLP()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

learning_intervals = 10

for epoch in range(20000):
    model.train()
    optimizer.zero_grad()

    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(epoch, loss.item())

FINAL_Y = model(X)
FINAL_Y = FINAL_Y.detach().numpy()

plt.plot(DATA_X, DATA_Y, label="f(x)")
plt.plot(DATA_X, FINAL_Y, label="f(x) model")
plt.plot()

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("F function")
plt.legend()
plt.show()