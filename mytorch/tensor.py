import numpy as np
from queue import PriorityQueue

class Tensor:

    def __init__(self, data: np.ndarray, requires_grad=False):

        if data is not np.ndarray:
            data = np.array(data)

        self.data = data
        self.grad = Tensor(None) if requires_grad else None
        self.creator = None
        self.generation = 0

        self.requires_grad = requires_grad

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        assert self.requires_grad, "Cannot apply cleargrad to a tensor which doesn't require grad."
        
        self.grad = Tensor(None)

    def backward(self):

        assert self.requires_grad, "Cannot apply backward to a tensor which doesn't require grad."

        if self.grad.data == None:
            self.grad = Tensor(np.ones_like(self.data))

        # funcs = [self.creator]
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

            gys = [output.grad.data for output in outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for input, gx in zip(inputs, gxs):
                if input.grad.data == None:
                    input.grad.data = gx
                else:
                    input.grad.data = input.grad.data + gx

                if input.creator is not None:
                    add_func(input.creator)

    def __str__(self):
        return str(self.data)

    def __eq__(self, other):
        return self.data == other.data