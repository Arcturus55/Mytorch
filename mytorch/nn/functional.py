import numpy as np

import mytorch
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
    
class Sum(Operation):

    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.old_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = mytorch.utils.reshape_sum_backward(gy, self.old_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.old_shape)
        return gx

class BroadcastTo(Operation):
    
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.old_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.old_shape)
        return gx
    
class SumTo(Operation):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.old_shape = x.shape
        y = mytorch.utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.old_shape)
        return gx
    
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

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def broadcast_to(x, shape):
    return BroadcastTo(shape)(x)

def sum_to(x, shape):
    return SumTo(shape)(x)