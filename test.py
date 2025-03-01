import numpy as np
from mytorch import Tensor
from mytorch import nn

x = Tensor(np.random.randn(3, 5))

layer = nn.Square()

print(layer(x).data)