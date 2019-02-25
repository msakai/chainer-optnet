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
    """LU decomposition.

    Decompose a given two-dimensional square matrix into ``P * L * U``,
    where ``P`` is a permutation matrix,  ``L`` lower-triangular with
    unit diagonal elements, and ``U`` upper-triangular matrix.
    Note that in the current implementation ``a`` must be
    a real matrix, and only float32 and float64 are supported.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(N, N)``
        overwrite_a (bool): Allow overwriting data in ``a`` (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        tuple:
            ``(lu, piv)`` where ``lu`` is a :class:`cupy.ndarray`
            storing ``U`` in its upper triangle, and ``L`` without
            unit diagonal elements in its lower triangle, and `piv` is
            a :class:`cupy.ndarray` storing pivot indices representing
            permutation matrix ``P``.

    .. seealso:: :func:`scipy.linalg.lu_factor`
    """

    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    util._assert_cupy_array(a)
    util._assert_rank2(a)
    util._assert_nd_squareness(a)

    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

    a = a.astype(dtype, order='F', copy=(not overwrite_a))

    if check_finite:
        if a.dtype.kind == 'f' and not cupy.isfinite(a).all():
            raise ValueError(
                "array must not contain infs or NaNs")

    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.intc)

    ipiv = cupy.empty((a.shape[0],), dtype=numpy.intc)

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

    # cuSolver uses 1-origin while SciPy uses 0-origin
    ipiv -= 1

    return (a, ipiv)


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """Solve an equation system, ``a * x = b``, given the LU factorization of ``a``

    Args:
        lu_and_piv (tuple): LU factorization of matrix ``a`` (``(M, N)``)
            together with pivot indices.
        b (cupy.ndarray): The matrix with dimension ``(M,)`` or
            ``(M, N)``.
        trans ({0, 1, 2}): Type of system to solve:
            ========  =========
            trans     system
            ========  =========
            0         a x  = b
            1         a^T x = b
            2         a^H x = b
            ========  =========
        overwrite_b (bool): Allow overwriting data in b (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M,)`` or ``(M, N)``.

    .. seealso:: :func:`scipy.linalg.lu_solve`
    """

    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

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

    if trans == 0:
        trans = cublas.CUBLAS_OP_N
    elif trans == 1:
        trans = cublas.CUBLAS_OP_T
    elif trans == 2:
        trans = cublas.CUBLAS_OP_H
    else:
        raise ValueError("unknown trans")

    lu = lu.astype(dtype, order='F', copy=False)
    ipiv = ipiv.astype(ipiv.dtype, order='F', copy=True)
    # cuSolver uses 1-origin while SciPy uses 0-origin
    ipiv += 1
    b = b.astype(dtype, order='F', copy=(not overwrite_b))

    if check_finite:
        if lu.dtype.kind == 'f' and not cupy.isfinite(lu).all():
            raise ValueError(
                "array must not contain infs or NaNs")
        if b.dtype.kind == 'f' and not cupy.isfinite(b).all():
            raise ValueError(
                "array must not contain infs or NaNs")

    n = 1 if b.ndim == 1 else b.shape[1]
    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.intc)

    if dtype == 'f':
        getrs = cusolver.sgetrs
    else:  # dtype == 'd'
        getrs = cusolver.dgetrs

    # solve for the inverse
    getrs(cusolver_handle,
          trans,
          m, n, lu.data.ptr, m, ipiv.data.ptr, b.data.ptr,
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
