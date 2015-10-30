import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from roi_pooling_2d import roi_pooling_2d
functions.roi_pooling_2d = roi_pooling_2d


class TestROIPooling2D(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.randn(4, 3, 12, 8).astype(numpy.float32)
        self.rois = numpy.array([
            [0, 1, 1, 6, 6],
            [2, 6, 2, 7, 11],
            [1, 3, 1, 5, 10],
            [0, 3, 3, 3, 3]
        ], dtype=numpy.float32)
        self.gy = numpy.random.uniform(-1, 1,
                                       (4, 3, 7, 7)).astype(numpy.float32)
        #    (4, 3, 7, 7)).astype(numpy.float32)

    def check_forward(self, x_data, roi_data):
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        y = functions.roi_pooling_2d(x, rois)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois))

    def check_backward(self, x_data, roi_data, y_grad):
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        y = functions.roi_pooling_2d(x, rois)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data, rois.data))
        gx, gr = gradient_check.numerical_grad(f, (x.data, rois.data),
                                               (y.grad,))

        gradient_check.assert_allclose(cuda.to_cpu(gx), cuda.to_cpu(x.grad))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
