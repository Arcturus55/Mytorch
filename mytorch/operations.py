import numpy as np

from .tensor import Tensor

class Operation:

    def __call__(self, *inputs):
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)

        requires_grad = False
        for input in inputs:
            if input.requires_grad:
                requires_grad = True
                break
        
        if not isinstance(ys, tuple):
            ys = (ys, )
        
        outputs = [Tensor(y, requires_grad=requires_grad) for y in ys]

        self.generation = max([input.generation for input in inputs])

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, *x):
        pass

    def backward(self, *gy):
        pass

    def __lt__(self, other):
        '''
        For priority queue, operations with high generation has high priority.
        '''

        return self.generation > other.generation

class Square(Operation):

    def forward(self, x):

        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data

        return 2 * x * gy
    
class Exp(Operation):

    def forward(self, x):

        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data

        return np.exp(x) * gy
        
    
class Add(Operation):

    def forward(self, x0, x1):

        return x0 + x1
    
    def backward(self, gy):

        return gy, gy
    
def square(x):
    
    return Square()(x)

def exp(x):
    
    return Exp()(x)

def add(x0, x1):

    return Add()(x0, x1)