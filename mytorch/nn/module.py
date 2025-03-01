import numpy as np

from ..tensor import Tensor

class Module:

    def __call__(self, input: Tensor):
        x = input.data
        y = self.forward(x)
        return Tensor(y)

    def forward(self, x):
        pass


class Square(Module):

    def forward(self, x):

        return x**2