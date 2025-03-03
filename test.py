import numpy as np
import mytorch
from mytorch import *

x = Tensor(np.array(2.0), requires_grad=True)
a = square(x)
y = add(square(a), square(a))
y.backward()
print(y.data)
print(x.grad)

# import unittest

# def numerical_diff(f, x, eps=1e-4):
#     x0 = Tensor(x.data - eps)
#     x1 = Tensor(x.data + eps)

#     y0 = f(x0)
#     y1 = f(x1)

#     return (y1.data - y0.data) / (2 * eps)

# class SquareTest(unittest.TestCase):

#     def test_forward(self):
#         x = Tensor(2.0)

#         y = F.square(x)

#         expected = Tensor(4.0)

#         self.assertEqual(y, expected)

#     def test_grad(self):

#         x = Tensor(np.random.rand())

#         y = F.square(x)

#         y.backward()

#         dx = numerical_diff(F.square, x)

#         flag = np.allclose(x.grad.data, dx)

# class AddTest(unittest.TestCase):

#     def test_backward(self):

#         x0 = Tensor(0.5)
#         x1 = Tensor(0.7)

#         y = F.add(F.square(x0), F.exp(x1))

#         y.backward()

#         self.assertEqual(x0.grad, Tensor(1))
#         self.assertEqual(x1.grad, F.exp(x1))