import numpy as np

import mytorch
from mytorch import Tensor, nn
import mytorch.nn.functional as F


np.random.seed(0)
x = np.random.rand(100, 3)
y = 5 + np.sum(np.sin(x), axis=1, keepdims=True) + np.random.rand(100, 1)

x, y = Tensor(x), Tensor(y)

print(x.shape, y.shape)

class FFN(nn.Module):

    def __init__(self):
        self.ln1 = nn.Linear(3, 4)
        self.act = nn.Sigmoid()
        self.ln2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.ln2(self.act(self.ln1(x)))

def MSE(pred, y):
    diff = pred - y
    return F.sum(diff ** 2) / len(y)

iters = 1000
lr = 1e-2
model = FFN()

for i in range(iters):
    pred = model(x)
    loss = MSE(pred, y)
    
    model.clear_grad()
    loss.backward()

    print(loss)

