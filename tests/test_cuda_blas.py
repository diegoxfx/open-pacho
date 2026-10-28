"""CPU cuBLAS ABI checks: layout, scalar location, lifetime and failure cleanup."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
from math import prod
from threading import RLock, local
from types import SimpleNamespace

import numpy as np
import pytest

from openpacho._cuda import CUDABackend


class FakeBLAS:
    Operation = SimpleNamespace(N=0, T=1)
    PointerMode = SimpleNamespace(HOST=0, DEVICE=1)

    def __init__(self, backend):
        self.backend = backend
        self.pointer_mode = self.PointerMode.HOST
        self.events = []
        self.fail = None

    def set_pointer_mode(self, handle, mode):
        assert handle == self.backend._handle
        assert self.backend._activation_depth > 0
        self.events.append(("mode", mode))
        if self.fail == "restore" and mode == self.PointerMode.HOST:
            raise RuntimeError("restore failed")
        self.pointer_mode = mode
        if self.fail == "device" and mode == self.PointerMode.DEVICE:
            raise RuntimeError("device failed")

    def _buffer(self, ptr, count, dtype):
        for allocation in self.backend.allocations:
            array = allocation.host
            if array.ctypes.data <= ptr < array.ctypes.data + array.nbytes:
                offset, remainder = divmod(ptr - array.ctypes.data, array.itemsize)
                assert not remainder
                assert array.dtype == np.dtype(dtype)
                result = array.reshape(-1)[offset : offset + count]
                assert result.size == count
                return result
        raise AssertionError(f"unknown device pointer {ptr}")

    def _dot(self, dtype, handle, size, x, incx, y, incy, out):
        assert self.pointer_mode == self.PointerMode.DEVICE
        assert handle == self.backend._handle
        assert self.backend._activation_depth > 0
        assert incx == incy == 1
        self.events.append(("dot", np.dtype(dtype).name, size))
        if self.fail == "dot":
            raise RuntimeError("dot failed")
        self._buffer(out, 1, dtype)[:] = self._buffer(x, size, dtype) @ self._buffer(y, size, dtype)

    def sdot(self, *args):
        self._dot(np.float32, *args)

    def ddot(self, *args):
        self._dot(np.float64, *args)

    def _gemv(self, dtype, handle, trans, m, n, alpha, matrix, lda, x, incx, beta, y, incy):
        assert self.pointer_mode == self.PointerMode.HOST
        assert handle == self.backend._handle
        assert self.backend._activation_depth > 0
        assert incx == incy == 1
        assert lda == m
        scalar = ctypes.c_float if dtype == np.float32 else ctypes.c_double
        assert scalar.from_address(alpha).value == 1
        assert scalar.from_address(beta).value == 0
        self.events.append(("gemv", np.dtype(dtype).name, trans, m, n))
        if self.fail == "gemv":
            raise RuntimeError("gemv failed")
        view = self._buffer(matrix, m * n, dtype).reshape((m, n), order="F")
        if trans == self.Operation.T:
            view = view.T
        self._buffer(y, view.shape[0], dtype)[:] = view @ self._buffer(x, view.shape[1], dtype)

    def sgemv(self, *args):
        self._gemv(np.float32, *args)

    def dgemv(self, *args):
        self._gemv(np.float64, *args)

    def _gemm(self, dtype, handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc):
        assert self.pointer_mode == self.PointerMode.HOST
        assert handle == self.backend._handle
        assert self.backend._activation_depth > 0
        assert transa == transb == self.Operation.N
        assert lda == ldc == m and ldb == k
        scalar = ctypes.c_float if dtype == np.float32 else ctypes.c_double
        assert scalar.from_address(alpha).value == 1
        assert scalar.from_address(beta).value == 0
        self.events.append(("gemm", np.dtype(dtype).name, m, n, k))
        if self.fail == "gemm":
            raise RuntimeError("gemm failed")
        left = self._buffer(a, m * k, dtype).reshape((m, k), order="F")
        right = self._buffer(b, k * n, dtype).reshape((k, n), order="F")
        self._buffer(c, m * n, dtype).reshape((m, n), order="F")[:] = left @ right

    def sgemm(self, *args):
        self._gemm(np.float32, *args)

    def dgemm(self, *args):
        self._gemm(np.float64, *args)


def fake_backend():
    backend = object.__new__(CUDABackend)
    backend.blas, backend._handle = "cublas", 71
    backend._lock, backend._closed = RLock(), False
    backend._context = 47
    backend._pending, backend.allocations = [], []
    state = local()

    def stack():
        if not hasattr(state, "stack"):
            state.stack = [99]
        return state.stack

    def push(context):
        stack().append(context)
        return (0,)

    def pop():
        return (0, stack().pop())

    backend.driver = SimpleNamespace(
        cuCtxPushCurrent=push,
        cuCtxPopCurrent=pop,
        cuCtxGetCurrent=lambda: (0, stack()[-1]),
    )

    def allocate(shape, dtype):
        host = np.full(shape, np.nan, dtype=dtype)
        data = SimpleNamespace(host=host, ptr=host.ctypes.data)
        backend.allocations.append(data)
        return data

    def full(shape, value, dtype):
        data = allocate(shape, dtype)
        data.host[...] = value
        return data

    def retain(args):
        backend._pending.extend(args)

    backend.empty, backend.full, backend._retain = allocate, full, retain
    backend._cublas = FakeBLAS(backend)
    return backend, stack


@pytest.fixture
def mocked_cuda():
    backend, stack = fake_backend()
    try:
        yield backend, stack
    finally:
        backend._closed = True


def upload(backend, host):
    data = backend.empty(host.shape, host.dtype)
    data.host[...] = host
    return SimpleNamespace(
        shape=host.shape, ndim=host.ndim, size=host.size, dtype=host.dtype, _data=data
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shapes",
    [
        ((17,), (17,)),
        ((13, 17), (17,)),
        ((17,), (17, 13)),
        ((2, 1, 13, 17), (17,)),
        ((17,), (1, 3, 17, 13)),
        ((2, 3, 13, 17), (17,)),
        ((17,), (2, 3, 17, 13)),
        ((1, 13, 17), (17,)),
        ((17,), (1, 17, 13)),
        ((1, 1, 13, 17), (17,)),
        ((17,), (1, 1, 17, 13)),
        ((1, 17), (17,)),
        ((17,), (17, 1)),
    ],
)
def test_vector_routing_matches_row_major_numpy_and_retains_allocations(mocked_cuda, dtype, shapes):
    backend, stack = mocked_cuda
    rng = np.random.default_rng(2031)
    hosts = [rng.uniform(-0.5, 0.5, shape).astype(dtype) for shape in shapes]
    a, b = [upload(backend, host) for host in hosts]
    expected = hosts[0] @ hosts[1]
    out = backend.matmul(a, b, expected.shape, np.dtype(dtype))
    np.testing.assert_allclose(out.host, expected, rtol=2e-6, atol=2e-6)
    assert stack() == [99]
    assert backend._cublas.pointer_mode == FakeBLAS.PointerMode.HOST
    assert backend._cublas_pointer_mode_dirty is False
    if a.ndim == b.ndim == 1:
        assert [event[0] for event in backend._cublas.events] == ["mode", "dot", "mode"]
        calls = 1
    else:
        matrix = a if b.ndim == 1 else b
        calls = prod(matrix.shape[:-2])
        primitive = "gemv" if matrix.ndim > 2 else "gemm"
        assert [event[0] for event in backend._cublas.events] == [primitive] * calls
    assert len(backend._pending) == calls * 3
    assert backend._pending[0] is a._data
    assert backend._pending[1] is b._data
    assert backend._pending[2] is out


@pytest.mark.parametrize("recovery_shape", [(3, 5), (2, 3, 5)])
@pytest.mark.parametrize("failure", ["device", "dot", "restore", "gemv", "gemm"])
def test_vector_failures_restore_mode_context_and_retain_submitted_work(
    mocked_cuda, failure, recovery_shape
):
    backend, stack = mocked_cuda
    backend._cublas.fail = failure
    host = np.ones({"gemv": (2, 3, 5), "gemm": (3, 5)}.get(failure, (5,)), np.float32)
    a, b = upload(backend, host), upload(backend, np.ones(5, np.float32))
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        backend.matmul(a, b, host.shape[:-1], np.dtype("float32"))
    expected_mode = (
        FakeBLAS.PointerMode.DEVICE if failure == "restore" else FakeBLAS.PointerMode.HOST
    )
    assert backend._cublas.pointer_mode == expected_mode
    assert backend._cublas_pointer_mode_dirty is (failure == "restore")
    assert stack() == [99]
    assert backend._activation_depth == 0
    assert len(backend._pending) == (3 if failure == "restore" else 0)
    backend._cublas.fail = None
    matrix = upload(backend, np.ones(recovery_shape, np.float32))
    actual = backend.matmul(matrix, b, recovery_shape[:-1], np.dtype("float32"))
    np.testing.assert_array_equal(actual.host, np.full(recovery_shape[:-1], 5, np.float32))
    assert backend._cublas.events[-1][0] == ("gemv" if len(recovery_shape) > 2 else "gemm")
    assert backend._cublas.pointer_mode == FakeBLAS.PointerMode.HOST
    assert backend._cublas_pointer_mode_dirty is False


@pytest.mark.parametrize("shape", [(3, 5), (1, 3, 5), (2, 3, 5)])
def test_failed_mode_recovery_blocks_later_vector_submission(mocked_cuda, shape):
    backend, stack = mocked_cuda
    a = upload(backend, np.ones(5, np.float32))
    backend._cublas.fail = "restore"
    with pytest.raises(RuntimeError, match="restore failed"):
        backend.matmul(a, a, (), np.dtype("float32"))
    matrix = upload(backend, np.ones(shape, np.float32))
    events = len(backend._cublas.events)
    with pytest.raises(RuntimeError, match="restore failed"):
        backend.matmul(matrix, a, shape[:-1], np.dtype("float32"))
    assert backend._cublas.events[events:] == [("mode", FakeBLAS.PointerMode.HOST)]
    assert backend._cublas_pointer_mode_dirty is True
    assert stack() == [99]


@pytest.mark.parametrize("shapes", [((0,), (0,)), ((3, 0), (0,)), ((3,), (3, 0)), ((0, 3), (3,))])
def test_empty_vectors_never_enter_blas(mocked_cuda, shapes):
    backend, _ = mocked_cuda
    hosts = [np.zeros(shape, np.float32) for shape in shapes]
    a, b = [upload(backend, host) for host in hosts]
    expected = hosts[0] @ hosts[1]
    actual = backend.matmul(a, b, expected.shape, np.dtype("float32"))
    np.testing.assert_array_equal(actual.host, expected)
    assert not backend._cublas.events


def test_vector_dimensions_preserve_32_bit_api_guard(mocked_cuda):
    backend, stack = mocked_cuda
    a = b = SimpleNamespace(shape=(2**31,), ndim=1)
    with pytest.raises(ValueError, match="32-bit API"):
        backend.matmul(a, b, (), np.dtype("float32"))
    assert not backend._cublas.events
    assert stack() == [99]


def test_handle_mode_changes_are_serialized_across_threads(mocked_cuda):
    backend, stack = mocked_cuda
    a = upload(backend, np.ones(17, np.float32))
    matrix = upload(backend, np.ones((3, 17), np.float32))
    batched = upload(backend, np.ones((2, 3, 17), np.float32))

    def work(index):
        for _ in range(10):
            left, shape = ((a, ()), (matrix, (3,)), (batched, (2, 3)))[index % 3]
            actual = backend.matmul(left, a, shape, np.dtype("float32"))
            np.testing.assert_array_equal(actual.host, np.full(shape, 17, np.float32))
            assert stack() == [99]

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(work, range(6)))
    assert backend._activation_depth == 0
    assert backend._cublas.pointer_mode == FakeBLAS.PointerMode.HOST
