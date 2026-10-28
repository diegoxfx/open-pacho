"""Native cuBLAS vector products for the contiguous CUDA array contract."""

from math import prod

import numpy as np


def vector_matmul(backend, a, b, out, dtype):
    """Submit a nonempty DOT or batched vector product under ``_activate``.

    The caller owns the handle lock/context and has checked the 32-bit limits.
    GEMV reads row-major storage through cuBLAS transpose flags; no transpose or
    scalar result travels through the host. The handle normally uses HOST scalar
    pointers, so DOT changes its mode only for the duration of that submission.
    """
    cublas, handle = backend._cublas, backend._handle
    single = np.dtype(dtype) == np.dtype("float32")
    if a.ndim == b.ndim == 1:
        function = cublas.sdot if single else cublas.ddot
        backend._cublas_pointer_mode_dirty = True
        try:
            cublas.set_pointer_mode(handle, cublas.PointerMode.DEVICE)
            function(handle, a.shape[0], a._data.ptr, 1, b._data.ptr, 1, out.ptr)
            # Retain before restoring the mode: even a failed restoration must
            # not release storage already referenced by asynchronous work.
            backend._retain((a._data, b._data, out))
        finally:
            cublas.set_pointer_mode(handle, cublas.PointerMode.HOST)
            backend._cublas_pointer_mode_dirty = False
        return

    function = cublas.sgemv if single else cublas.dgemv
    alpha, beta = np.array(1, dtype=dtype), np.array(0, dtype=dtype)
    if b.ndim == 1:
        matrix, vector = a, b
        rows, columns = a.shape[-1], a.shape[-2]
        operation, output_size = cublas.Operation.T, columns
    else:
        matrix, vector = b, a
        rows, columns = b.shape[-1], b.shape[-2]
        operation, output_size = cublas.Operation.N, rows
    itemsize = np.dtype(dtype).itemsize
    # One shared rank-one vector, with contiguous, independent matrix slices.
    # GEMV strided-batched forbids zero strides, so preserve the existing loop
    # instead of relying on unsupported shared-vector stride zero.
    for index in range(prod(matrix.shape[:-2])):
        function(
            handle,
            operation,
            rows,
            columns,
            alpha.ctypes.data,
            matrix._data.ptr + index * rows * columns * itemsize,
            rows,
            vector._data.ptr,
            1,
            beta.ctypes.data,
            out.ptr + index * output_size * itemsize,
            1,
        )
        backend._retain((a._data, b._data, out))
