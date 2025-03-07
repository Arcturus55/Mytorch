import numpy as np
from weakref import ref
from queue import PriorityQueue

from .config import Config

class Tensor:

    def __init__(self, data: np.ndarray, requires_grad=False, name=None):

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self.data = data
        self.grad = Tensor(None) if requires_grad else None
        self.name = name
        self.creator = None
        self.generation = 0

        self.requires_grad = requires_grad

    __array_priority__ = 200

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        assert self.requires_grad, "Cannot apply cleargrad to a tensor which doesn't require grad."
        
        self.grad = Tensor(None)

    def backward(self, retain_grad=False):
        assert self.requires_grad, "Cannot apply backward to a tensor which doesn't require grad."

        if self.grad.data == None:
            self.grad = Tensor(np.ones_like(self.data))

        funcs = PriorityQueue()
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.put(f)
                seen_set.add(f)

        add_func(self.creator)

        while not funcs.empty():
            f = funcs.get()
            
            inputs, outputs = f.inputs, f.outputs

            gys = [output().grad.data for output in outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for input, gx in zip(inputs, gxs):

                if not input.requires_grad:
                    continue

                if input.grad.data == None:
                    input.grad.data = gx
                else:
                    input.grad.data = input.grad.data + gx

                if input.creator is not None:
                    add_func(input.creator)

            if not retain_grad:
                for output in f.outputs:
                    output().grad.data = None

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "tensor(None)"
        p = str(self.data).replace('\n', '\n' + ' '*7)
        return 'tensor(' + p + ', dtype=' + str(self.dtype) + (', requires_grad=True' if self.requires_grad else '') + ')'

    def __eq__(self, other):
        return self.data == other.data
    
    def __neg__(self):
        return neg(self)
    
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(self, other)
    
    def __sub__(self, other):
        return sub(self, other)
    
    def __rsub__(self, other):
        return sub(other, self)
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __rmul__(self, other):
        return mul(self, other)
    
    def __truediv__(self, other):
        return div(self, other)
    
    def __rtruediv__(self, other):
        return div(other, self)
    
    def __pow__(self, exponent):
        return pow(self, exponent)
    
class Operation:

    def __call__(self, *inputs):
        inputs = [as_tensor(input) for input in inputs]
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

        if Config.enable_backprop:
            self.generation = max([input.generation for input in inputs])

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [ref(output) for output in outputs]

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

class Add(Operation):

    def forward(self, x0, x1):
        return x0 + x1
    
    def backward(self, gy):
        return gy, gy
    
class Sub(Operation):

    def forward(self, x0, x1):
        return x0 - x1
    
    def backward(self, gy):
        return gy, -gy
    
class Mul(Operation):

    def forward(self, x0, x1):
        return x0 * x1
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1 * gy, x0 * gy
    
class Div(Operation):

    def forward(self, x0, x1):
        return x0 / x1
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy / x1, - (gy * x0) / (x1 ** 2)
    
class Pow(Operation):

    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        return gy * c * x ** (c-1)

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
    
class Neg(Operation):

    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
        
def add(x0, x1):
    return Add()(x0, x1)

def sub(x0, x1):
    return Sub()(x0, x1)

def mul(x0, x1):
    return Mul()(x0, x1)   

def div(x0, x1):
    return Div()(x0, x1)

def pow(x, c):
    return Pow(c)(x) 
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def neg(x):
    return Neg()(x)

def as_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x)

def rand(size, requires_grad=False, name=None):
    return Tensor(np.random.rand(*size), requires_grad=requires_grad, name=name)

def randn(size, requires_grad=False, name=None):
    return Tensor(np.random.randn(*size), requires_grad=requires_grad, name=name)

def randint(low, high=None, size=..., dtype=np.int64, requires_grad=False, name=None):
    return Tensor(np.random.randint(low, high, size, dtype), requires_grad=requires_grad, name=name)