import numpy as np

import mytorch
from mytorch import Tensor
import mytorch.nn.functional as F

x0 = mytorch.randn(size=(2, 3))
x1 = mytorch.randn(size=(1, 3))

y = x0 + x1
y.backward()

print(x0.grad)
print(x1.grad)