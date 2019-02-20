import numpy
from numpy import linalg

import cupy
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import util

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver


def lu_factor(a, overwrite_a=False, check_finite=True):
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    util._assert_cupy_array(a)
    util._assert_rank2(a)
    util._assert_nd_squareness(a)

    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

    # to prevent `a` to be overwritten
    a = a.astype(dtype, order='F', copy=True)

    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.intc)

    ipiv = cupy.empty((a.shape[0], 1), dtype=numpy.intc)

    if dtype == 'f':
        getrf = cusolver.sgetrf
        getrf_bufferSize = cusolver.sgetrf_bufferSize
    else:  # dtype == 'd'
        getrf = cusolver.dgetrf
        getrf_bufferSize = cusolver.dgetrf_bufferSize

    m = a.shape[0]

    buffersize = getrf_bufferSize(cusolver_handle, m, m, a.data.ptr, m)
    workspace = cupy.empty(buffersize, dtype=dtype)

    # LU factorization
    getrf(cusolver_handle, m, m, a.data.ptr, m, workspace.data.ptr,
          ipiv.data.ptr, dev_info.data.ptr)

    return (a, ipiv - 1)


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    (lu, ipiv) = lu_and_piv

    util._assert_cupy_array(lu)
    util._assert_rank2(lu)
    util._assert_nd_squareness(lu)

    m = lu.shape[0]
    if m != b.shape[0]:
        raise ValueError("incompatible dimensions.")

    if lu.dtype.char == 'f' or lu.dtype.char == 'd':
        dtype = lu.dtype.char
    else:
        dtype = numpy.find_common_type((lu.dtype.char, 'f'), ()).char

    lu = lu.astype(dtype, order='F', copy=False)
    ipiv = ipiv.astype(ipiv.dtype, order='F', copy=False)
    ipiv += 1

    if b.ndim == 1:
        b = cupy.expand_dims(b, 1)
    b = b.astype(dtype, order='F', copy=True)

    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.intc)

    if dtype == 'f':
        getrs = cusolver.sgetrs
    else:  # dtype == 'd'
        getrs = cusolver.dgetrs

    # solve for the inverse
    getrs(cusolver_handle,
          cublas.CUBLAS_OP_T if trans else cublas.CUBLAS_OP_N,
          m, b.shape[1], lu.data.ptr, m, ipiv.data.ptr, b.data.ptr,
          m, dev_info.data.ptr)

    return b


if __name__ == '__main__':
    xp = cupy
    
    A = xp.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]], dtype=numpy.float64)
    b = xp.array([1, 1, 1, 1], dtype=numpy.float64)
    lu, piv = lu_factor(A)
    print(lu)
    print(piv)
    
    L, U = xp.tril(lu, k=-1) + xp.eye(4), xp.triu(lu)
    print(L @ U)
    
    x = lu_solve((lu, piv), b.copy())
    print(x)
    print(A @ x)
    print(b)
    print(xp.allclose(A @ x - b, xp.zeros((4,))))
