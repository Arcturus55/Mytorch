import numpy as np

class Optimizer:

    def __init__(self, params):
        self.params = list(params)
        self.hooks = []

    def step(self):
        for f in self.hooks:
            f(self.params)

        for param in self.params:
            self.update_one(param)

    def update_one(self, param):
        pass

    def add_hook(self, hook):
        self.hooks.append(hook)

class SGD(Optimizer):

    def __init__(self, params, lr):
        super().__init__(params)
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

class MomentumSGD(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param)

        v = self.vs[v_key]
        v = self.momentum * v - self.lr * param.grad.data
        param.data += v