import unittest

import numpy
try:
    import scipy.linalg

    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import cuda
from cupy import testing
from cupy.testing import condition

import cupy_decomp_lu


@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@unittest.skipUnless(scipy_available, 'requires scipy')
@testing.gpu
@testing.fix_random()
class TestLUFactor(unittest.TestCase):

    @testing.for_float_dtypes(no_float16=True)
    def check_x(self, array, dtype):
        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        result_cpu = scipy.linalg.lu_factor(a_cpu)
        result_gpu = cupy_decomp_lu.lu_factor(a_gpu)
        self.assertEqual(result_cpu[0].dtype, result_gpu[0].dtype)
        self.assertEqual(result_cpu[1].dtype, result_gpu[1].dtype)
        cupy.testing.assert_allclose(result_cpu[0], result_gpu[0], atol=1e-4)
        cupy.testing.assert_array_equal(result_cpu[1], result_gpu[1])

    def test_lu_factor(self):
        self.check_x(numpy.random.randn(1, 1))
        self.check_x(numpy.random.randn(2, 2))
        self.check_x(numpy.random.randn(3, 3))
        self.check_x(numpy.random.randn(5, 5))


@testing.parameterize(
    {'trans': 0},
    {'trans': 1},
)
class TestLUSolve(unittest.TestCase):

    @testing.for_float_dtypes(no_float16=True)
    def check_x(self, a_shape, b_shape, dtype, trans):
        a_cpu = numpy.random.randn(*a_shape).astype(dtype)
        b_cpu = numpy.random.randn(*b_shape).astype(dtype)
        a_gpu = cupy.asarray(a_cpu)
        b_gpu = cupy.asarray(b_cpu)
        lu_cpu = scipy.linalg.lu_factor(a_cpu)
        lu_gpu = cupy_decomp_lu.lu_factor(a_gpu)
        result_cpu = scipy.linalg.lu_solve(lu_cpu, b_cpu, trans=trans)
        result_gpu = cupy_decomp_lu.lu_solve(lu_gpu, b_gpu, trans=trans)
        self.assertEqual(result_cpu.dtype, result_gpu.dtype)
        cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-4)

    def test_solve(self):
        self.check_x((4, 4), (4,), trans=self.trans)
        self.check_x((5, 5), (5, 2), trans=self.trans)
