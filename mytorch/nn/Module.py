import numpy as np

import mytorch
import mytorch.nn.functional as F
from ..tensor import Tensor
from .parameter import Parameter

class Module:

    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Module)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *xs):
        return self.forward(*xs)

    def forward(self, *xs):
        pass

    def parameters(self):
        for name in self._params:
            value = self.__dict__[name]
            if isinstance(value, Module):
                yield from value.parameters()
            else:
                yield value

    def clear_grads(self):
        for parameter in self.parameters():
            parameter.clear_grad()

class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(mytorch.randn(size=(in_features, out_features)))
        self.weight.data *= np.sqrt(1/in_features)
        if bias:
            self.bias = Parameter(mytorch.zeros(size=(out_features, )))
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is None:
            return x @ self.weight
        return x @ self.weight + self.bias
    
class Sigmoid(Module):

    def forward(self, x):
        return 1 / (1 + F.exp(x))

class MSELoss(Module):
    def forward(self, pred, y):
        diff = pred - y
        loss = F.sum(diff ** 2) / len(y)
        return loss