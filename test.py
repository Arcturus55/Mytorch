import numpy as np

import mytorch
from mytorch import Tensor
import mytorch.nn.functional as F

x = mytorch.randn(size=(3, 2, 5, 4))

y = x.transpose()

print(y.shape)

y.backward()

print(x.grad.shape)

