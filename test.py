import numpy as np

import mytorch

x = mytorch.rand(size=(3, 2), requires_grad=True)

y = x ** 3

print(y)

y.backward()

print(x.grad)

