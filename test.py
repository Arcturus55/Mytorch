import numpy as np

import mytorch
from mytorch import Tensor, nn, optim
import mytorch.nn.functional as F


np.random.seed(0)
x = np.random.rand(100, 3)
y = 5 + np.sum(np.sin(x), axis=1, keepdims=True) + np.random.rand(100, 1)

x, y = Tensor(x), Tensor(y)

class FFN(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.Linear(3, 4)
        self.act = nn.Sigmoid()
        self.ln2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.ln2(self.act(self.ln1(x)))


    
dataloader = [(Tensor(x.data[i:i+4]), Tensor(y.data[i:i+4])) for i in range(0, 96, 4)]

epochs = 200
lr = 1e-1
model = FFN()
loss_fn = nn.MSELoss()
optimizer = optim.MomentumSGD(model.parameters(), lr=lr)

for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        pred = model(x)
        loss = loss_fn(pred, y)

        total_loss += loss.data
        
        model.clear_grads()
        loss.backward()

        optimizer.step()

    if epoch%10 == 0:
        print(f"Epoch: {epoch}, loss: {total_loss/len(dataloader)}.")

