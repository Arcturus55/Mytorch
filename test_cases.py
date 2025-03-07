import unittest
import numpy as np

import mytorch
from mytorch import Tensor

def numerical_diff(f, x, eps=1e-4):
    x0 = Tensor(x.data - eps)
    x1 = Tensor(x.data + eps)

    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):

    def test_forward(self):
        x = Tensor(2.0)

        y = mytorch.square(x)

        expected = Tensor(4.0)

        self.assertEqual(y, expected)

    def test_grad(self):

        x = Tensor(np.random.rand())

        y = mytorch.square(x)

        y.backward()

        dx = numerical_diff(mytorch.square, x)

        flag = np.allclose(x.grad.data, dx)

        assert flag

class AddTest(unittest.TestCase):

    def test_backward(self):

        x0 = Tensor(0.5)
        x1 = Tensor(0.7)

        y = mytorch.square(x0) + mytorch.exp(x1)

        y.backward()

        self.assertEqual(x0.grad, Tensor(1))
        self.assertEqual(x1.grad, mytorch.exp(x1))