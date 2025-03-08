import numpy as np

import mytorch
import mytorch.nn.functional as F
from ..tensor import Tensor

class Module:

    def __call__(self, *xs):
        return self.forward(*xs)
    
    def forward(self, *xs):
        pass

    def clear_grad(self):
        for layer in dir(self):
            if isinstance(layer, Linear):
                layer.weight.cleargrad()
                layer.bias.cleargrad()

class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        self.weight = mytorch.randn(size=(in_features, out_features))
        if bias:
            self.bias = mytorch.randn(size=(out_features, ))
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is None:
            return x @ self.weight
        return x @ self.weight + self.bias
    
class Sigmoid(Module):

    def forward(self, x):
        return 1 / (1 + F.exp(x))
