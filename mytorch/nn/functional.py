import numpy as np
from ..tensor import Operation, as_tensor

class Sin(Operation):

    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        x = self.inputs[0]
        return cos(x) * gy
    
class Cos(Operation):

    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        x = self.inputs[0]
        return -sin(x) * gy
    
class Square(Operation):

    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0]
        return 2 * x * gy
    
class Exp(Operation):

    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0]
        return np.exp(x) * gy
    
class Tanh(Operation):

    def forward(self, x):
        return (exp(x) - exp(-x))/(exp(x) + exp(-x))
    
    def backward(self, gy):
        y = self.outputs[0]
        return gy * (1 - y * y)
    
class Reshape(Operation):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.old_shape = x.shape
        return np.reshape(x, self.shape)
    
    def backward(self, gy):
        return reshape(gy, self.old_shape)
    
class Transpose(Operation):

    def __init__(self, axes):
        self.axes = axes

    def forward(self, x):
        return np.transpose(x, self.axes)
    
    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        # print(inv_axes)
        return transpose(gy, inv_axes)
    
def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def reshape(x, shape):
    if x.shape == shape:
        return as_tensor(x)
    return Reshape(shape)(x)

def transpose(x, axes=None):
    return Transpose(axes)(x)