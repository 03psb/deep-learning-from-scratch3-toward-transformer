import unittest
import numpy as np
import dezero.layers as L
import dezero.functions as F
from dezero.utils import check_backward
import chainer.functions as CF


class TestConv2d_simple(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (1, 1), (1, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = F.conv2d_simple(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (3, 1), (2, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = F.conv2d_simple(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_forward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = F.conv2d_simple(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_forward4(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = np.random.randn(o).astype('f')
        y = F.conv2d_simple(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda x: F.conv2d_simple(x, W, b, s, p)
        self.assertTrue(check_backward(f, x))

    def test_backward2(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda b: F.conv2d_simple(x, W, b, s, p)
        self.assertTrue(check_backward(f, b))

    def test_backward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda W: F.conv2d_simple(x, W, b, s, p)
        self.assertTrue(check_backward(f, W))


class TestConv2d(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (1, 1), (1, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = F.conv2d(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (3, 1), (2, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = F.conv2d(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_forward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = F.conv2d(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_forward4(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = np.random.randn(o).astype('f')
        y = F.conv2d(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda x: F.conv2d(x, W, b, s, p)
        self.assertTrue(check_backward(f, x))

    def test_backward2(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda b: F.conv2d(x, W, b, s, p)
        self.assertTrue(check_backward(f, b))

    def test_backward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda W: F.conv2d(x, W, b, s, p)
        self.assertTrue(check_backward(f, W))