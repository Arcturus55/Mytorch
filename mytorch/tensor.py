import numpy as np
from weakref import ref
from queue import PriorityQueue

import mytorch
from .config import Config, using_config
# from .nn.functional import reshape

class Tensor:

    def __init__(self, data: np.ndarray, name=None):

        if isinstance(data, Tensor):
            data = data.data
        elif not isinstance(data, np.ndarray):
            data = np.array(data)

        self.data = data
        self.grad = None
        self.name = name
        self.creator = None
        self.generation = 0

    __array_priority__ = 200

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):

        if self.grad == None:
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

            gys = [output().grad for output in outputs]
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for input, gx in zip(inputs, gxs):
                if input.grad is None:
                    input.grad = gx
                else:
                    input.grad = input.grad + gx

                if input.creator is not None:
                    add_func(input.creator)

            if not retain_grad:
                for output in f.outputs:
                    output().grad = None

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
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return mytorch.nn.functional.reshape(self, shape)
    
    def transpose(self, axes=None):
        return mytorch.nn.functional.transpose(self, axes)
    
    @property
    def T(self):
        return mytorch.nn.functional.transpose(self)
    
    def sum(self, axis=None, keepdims=False):
        return mytorch.nn.functional.sum(self, axis, keepdims)
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "tensor(None)"
        p = str(self.data).replace('\n', '\n' + ' '*7)
        return 'tensor(' + p + ', dtype=' + str(self.dtype) + ')'

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
    
    def __matmul__(self, other):
        return matmul(self, other)
    
class Operation:

    def __call__(self, *inputs):
        inputs = [as_tensor(input) for input in inputs]
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )
        
        outputs = [Tensor(y) for y in ys]

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
        self.shape0, self.shape1 = x0.shape, x1.shape
        return x0 + x1
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.shape0 != self.shape1:
            gx0 = mytorch.nn.functional.sum_to(gx0, self.shape0)
            gx1 = mytorch.nn.functional.sum_to(gx1, self.shape1)
        return gx0, gx1
    
class Sub(Operation):

    def forward(self, x0, x1):
        self.shape0, self.shape1 = x0.shape, x1.shape
        return x0 - x1
    
    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.shape0 != self.shape1:
            gx0 = mytorch.nn.functional.sum_to(gx0, self.shape0)
            gx1 = mytorch.nn.functional.sum_to(gx1, self.shape1)
        return gx0, gx1
    
class Mul(Operation):

    def forward(self, x0, x1):
        self.shape0, self.shape1 = x0.shape, x1.shape
        return x0 * x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = x1 * gy
        gx1 = x0 * gy
        if self.shape0 != self.shape1:
            gx0 = mytorch.nn.functional.sum_to(gx0, self.shape0)
            gx1 = mytorch.nn.functional.sum_to(gx1, self.shape1)
        return gx0, gx1
    
class Div(Operation):

    def forward(self, x0, x1):
        self.shape0, self.shape1 = x0.shape, x1.shape
        return x0 / x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = -(gy * x0) / (x1 ** 2)
        if self.shape0 != self.shape1:
            gx0 = mytorch.nn.functional.sum_to(gx0, self.shape0)
            gx1 = mytorch.nn.functional.sum_to(gx1, self.shape1)
        return gx0, gx1
    
class Pow(Operation):

    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        x = self.inputs[0]
        c = self.c

        return gy * c * x ** (c-1)
    
class Neg(Operation):

    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
class MatMul(Operation):

    def forward(self, x, W):
        return x.dot(W)
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)

        return gx, gW

        
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

def neg(x):
    return Neg()(x)

def matmul(x, W):
    return MatMul()(x, W)

def as_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x)

def zeros(size, dtype=None, name=None):
    return Tensor(np.zeros(size, dtype), name=name)

def rand(size, name=None):
    return Tensor(np.random.rand(*size), name=name)

def randn(size, name=None):
    return Tensor(np.random.randn(*size), name=name)

def randint(low, high=None, size=..., dtype=np.int64, name=None):
    return Tensor(np.random.randint(low, high, size, dtype), name=name)