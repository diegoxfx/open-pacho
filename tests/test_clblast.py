"""ABI and argument-mapping tests for the optional CLBlast adapter."""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

import openpacho as op
import openpacho._clblast as module
from examples import solvers
from openpacho.backends import get_backend
from openpacho.exceptions import BackendUnavailableError, UnsupportedDtypeError


class FakeFunction:
    def __init__(self, status=0):
        self.status = status
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.status


class FakeLibrary:
    def __init__(self, *, status=0, symbols=("CLBlastSgemm", "CLBlastDgemm")):
        for symbol in symbols:
            setattr(self, symbol, FakeFunction(status))


class FakeData:
    def __init__(self, pointer):
        self.int_ptr = pointer


class FakeArray:
    def __init__(self, shape, dtype, pointer):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._data = FakeData(pointer)


class FakeQueue:
    int_ptr = 0x100


def _value(value):
    return value.value if hasattr(value, "value") else int(value)


def _queue_value(pointer):
    return ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value


def _size_values(pointer, count):
    values = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_size_t))
    return [values[index] for index in range(count)]


@pytest.fixture
def fake_library(monkeypatch):
    library = FakeLibrary()
    monkeypatch.setattr(module, "_load_library", lambda path: library)
    return library


def test_explicit_path_configures_the_two_supported_gemms(fake_library):
    adapter = module.CLBlast("C:/opt/clblast.dll")

    assert adapter.path == "C:/opt/clblast.dll"
    assert fake_library.CLBlastSgemm.restype is ctypes.c_int
    assert fake_library.CLBlastDgemm.restype is ctypes.c_int
    assert fake_library.CLBlastSgemm.argtypes[-2:] == [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    assert fake_library.CLBlastSgemm.argtypes[6] is ctypes.c_float
    assert fake_library.CLBlastDgemm.argtypes[6] is ctypes.c_double


@pytest.mark.parametrize("ta,tb", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("dtype,prefix", [(np.float32, "S"), (np.float64, "D")])
def test_transposed_gemm_uses_storage_leading_dimensions(fake_library, ta, tb, dtype, prefix):
    adapter = module.CLBlast("mock")
    left = (7, 5) if ta else (5, 7)
    right = (9, 7) if tb else (7, 9)
    adapter.matmul(
        FakeQueue(),
        FakeArray(left, dtype, 0x201),
        FakeArray(right, dtype, 0x202),
        FakeData(0x203),
        (5, 9),
        dtype,
        transpose_a=ta,
        transpose_b=tb,
    )
    call = getattr(fake_library, f"CLBlast{prefix}gemm").calls[0]
    assert [_value(v) for v in call[:6]] == [101, 112 if ta else 111, 112 if tb else 111, 5, 9, 7]
    assert [_value(call[i]) for i in (9, 12, 16)] == [left[1], right[1], 9]
    assert [_value(call[i]) for i in (7, 10, 14)] == [0x201, 0x202, 0x203]


def test_transpose_flags_reject_batched_operands(fake_library):
    adapter = module.CLBlast("mock")
    with pytest.raises(ValueError, match="two-dimensional"):
        adapter.matmul(
            FakeQueue(),
            FakeArray((2, 7, 5), np.float32, 0x201),
            FakeArray((7, 9), np.float32, 0x202),
            FakeData(0x203),
            (2, 5, 9),
            np.float32,
            transpose_a=True,
        )


def test_optional_batched_symbols_configure_exact_local_header_abi(monkeypatch):
    library = FakeLibrary(
        symbols=(
            "CLBlastSgemm",
            "CLBlastDgemm",
            "CLBlastSgemmBatched",
            "CLBlastDgemmBatched",
            "CLBlastSgemmStridedBatched",
            "CLBlastDgemmStridedBatched",
        )
    )
    monkeypatch.setattr(module, "_load_library", lambda _: library)

    adapter = module.CLBlast("mock")

    assert set(adapter._gemm_batched) == {"CLBlastSgemmBatched", "CLBlastDgemmBatched"}
    assert set(adapter._gemm_strided_batched) == {
        "CLBlastSgemmStridedBatched",
        "CLBlastDgemmStridedBatched",
    }
    assert library.CLBlastSgemmBatched.argtypes[6] == ctypes.POINTER(ctypes.c_float)
    assert library.CLBlastSgemmBatched.argtypes[8] == ctypes.POINTER(ctypes.c_size_t)
    assert library.CLBlastSgemmBatched.argtypes[13] == ctypes.POINTER(ctypes.c_float)
    assert library.CLBlastSgemmBatched.argtypes[-2:] == [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    assert library.CLBlastSgemmStridedBatched.argtypes[6] is ctypes.c_float
    assert library.CLBlastSgemmStridedBatched.argtypes[10] is ctypes.c_size_t
    assert library.CLBlastSgemmStridedBatched.argtypes[20] is ctypes.c_size_t


@pytest.mark.parametrize(
    "left,right,routine,transpose",
    [
        ((13, 17), (17,), "gemv", 111),
        ((13,), (13, 17), "gemv", 112),
        ((17,), (17,), "dot", None),
    ],
)
def test_native_vector_routine_mapping(monkeypatch, left, right, routine, transpose):
    library = FakeLibrary(symbols=("CLBlastSgemm", "CLBlastDgemm", "CLBlastSgemv", "CLBlastSdot"))
    monkeypatch.setattr(module, "_load_library", lambda _: library)
    fills = []
    monkeypatch.setattr(
        module, "_zero_gemv_output", lambda queue, out, count, dtype: fills.append(count)
    )
    adapter = module.CLBlast("fake.dll")
    shape = np.matmul(np.zeros(left), np.zeros(right)).shape
    adapter.matmul(
        FakeQueue(),
        FakeArray(left, "float32", 1),
        FakeArray(right, "float32", 2),
        FakeData(3),
        shape,
        np.dtype("float32"),
    )
    call = getattr(library, "CLBlastS" + routine).calls[0]
    assert not library.CLBlastSgemm.calls
    assert _queue_value(call[-2]) == FakeQueue.int_ptr
    if routine == "gemv":
        assert fills == [shape[0]]
        assert _value(call[1]) == transpose
        assert _value(call[10]) == _value(call[14]) == 1
        assert _value(call[4]) == 1 and _value(call[11]) == 0
    else:
        assert not fills
        assert _value(call[0]) == 17
        assert _value(call[1]) == 3 and _value(call[3]) == 1 and _value(call[6]) == 2


def test_environment_path_is_used_when_no_explicit_path(monkeypatch):
    library = FakeLibrary()
    seen = []
    monkeypatch.setenv("OPENPACHO_CLBLAST_LIBRARY", "/tmp/libclblast.so")
    monkeypatch.setattr(module, "_load_library", lambda path: seen.append(path) or library)

    adapter = module.CLBlast()

    assert adapter.path == "/tmp/libclblast.so"
    assert seen == ["/tmp/libclblast.so"]


def test_missing_library_has_actionable_error(monkeypatch):
    monkeypatch.delenv("OPENPACHO_CLBLAST_LIBRARY", raising=False)
    monkeypatch.setattr(module, "find_library", lambda name: None)

    def fail(path):
        raise OSError("not installed")

    monkeypatch.setattr(module, "_load_library", fail)

    with pytest.raises(BackendUnavailableError, match="OPENPACHO_CLBLAST_LIBRARY"):
        module.CLBlast()


def test_explicit_library_load_failure_is_not_silently_fallback(monkeypatch):
    def fail(path):
        raise OSError("bad DLL")

    monkeypatch.setattr(module, "_load_library", fail)

    with pytest.raises(BackendUnavailableError, match="bad DLL"):
        module.CLBlast("missing-clblast.dll")


def test_missing_gemm_symbol_is_actionable(monkeypatch):
    monkeypatch.setattr(
        module,
        "_load_library",
        lambda path: FakeLibrary(symbols=("CLBlastSgemm",)),
    )

    with pytest.raises(BackendUnavailableError, match="CLBlastDgemm"):
        module.CLBlast("mock")


@pytest.mark.parametrize(
    "ashape,bshape,shape,expected_dims",
    [
        ((3,), (3,), (), (1, 1, 3, 3, 1, 1)),
        ((3,), (3, 2), (2,), (1, 2, 3, 3, 2, 2)),
        ((4, 3), (3,), (4,), (4, 1, 3, 3, 1, 1)),
        ((4, 3), (3, 2), (4, 2), (4, 2, 3, 3, 2, 2)),
    ],
)
def test_row_major_gemm_mapping_for_vectors_and_matrices(
    fake_library, ashape, bshape, shape, expected_dims
):
    adapter = module.CLBlast("mock")
    a = FakeArray(ashape, np.float32, 0x201)
    b = FakeArray(bshape, np.float32, 0x202)
    out = FakeData(0x203)

    assert adapter.matmul(FakeQueue(), a, b, out, shape, np.float32) is None

    call = fake_library.CLBlastSgemm.calls[-1]
    assert call[:3] == (101, 111, 111)
    assert tuple(_value(value) for value in call[3:6]) == expected_dims[:3]
    assert _value(call[7]) == 0x201
    assert _value(call[10]) == 0x202
    assert _value(call[14]) == 0x203
    assert tuple(_value(value) for value in (call[9], call[12], call[16])) == expected_dims[3:]
    assert _queue_value(call[17]) == 0x100
    assert call[18] is None


def test_pointer_batched_gemm_uses_offsets_for_irregular_multidimensional_broadcast(monkeypatch):
    library = FakeLibrary(
        symbols=(
            "CLBlastSgemm",
            "CLBlastDgemm",
            "CLBlastSgemmBatched",
            "CLBlastSgemmStridedBatched",
        )
    )
    monkeypatch.setattr(module, "_load_library", lambda _: library)
    adapter = module.CLBlast("mock")
    # Batch shapes (2, 1) and (1, 3) broadcast to (2, 3).  The inputs remain
    # in their original buffers; singleton dimensions repeat offsets.
    a = FakeArray((2, 1, 3, 4), np.float32, 0x201)
    b = FakeArray((1, 3, 4, 5), np.float32, 0x202)
    out = FakeData(0x203)

    adapter.matmul(FakeQueue(), a, b, out, (2, 3, 3, 5), np.float32)

    assert not library.CLBlastSgemm.calls
    assert not library.CLBlastSgemmStridedBatched.calls
    call = library.CLBlastSgemmBatched.calls[0]
    assert tuple(_value(value) for value in call[3:6]) == (3, 5, 4)
    assert _value(call[7]) == 0x201 and _value(call[10]) == 0x202
    assert _size_values(call[8], 6) == [0, 0, 0, 12, 12, 12]
    assert _size_values(call[11], 6) == [0, 20, 40, 0, 20, 40]
    assert _size_values(call[15], 6) == [0, 15, 30, 45, 60, 75]
    assert _value(call[17]) == 6
    assert _queue_value(call[18]) == FakeQueue.int_ptr
    assert call[19] is None


def test_strided_batched_gemm_is_preferred_when_both_optional_symbols_exist(monkeypatch):
    library = FakeLibrary(
        symbols=(
            "CLBlastSgemm",
            "CLBlastDgemm",
            "CLBlastSgemmBatched",
            "CLBlastSgemmStridedBatched",
        )
    )
    monkeypatch.setattr(module, "_load_library", lambda _: library)
    adapter = module.CLBlast("mock")
    a = FakeArray((2, 3, 4), np.float32, 0x201)
    b = FakeArray((2, 4, 5), np.float32, 0x202)
    adapter.matmul(FakeQueue(), a, b, FakeData(0x203), (2, 3, 5), np.float32)

    call = library.CLBlastSgemmStridedBatched.calls[0]
    assert not library.CLBlastSgemmBatched.calls
    assert tuple(_value(value) for value in call[3:6]) == (3, 5, 4)
    assert _value(call[8]) == 0 and _value(call[9]) == 4 and _value(call[10]) == 12
    assert _value(call[12]) == 0 and _value(call[13]) == 5 and _value(call[14]) == 20
    assert _value(call[17]) == 0 and _value(call[18]) == 5 and _value(call[19]) == 15
    assert _value(call[20]) == 2


def test_irregular_broadcast_falls_back_to_regular_gemm_when_pointer_batch_missing(
    fake_library,
):
    adapter = module.CLBlast("mock")
    a = FakeArray((2, 1, 3, 4), np.float32, 0x201)
    b = FakeArray((1, 3, 4, 5), np.float32, 0x202)

    adapter.matmul(FakeQueue(), a, b, FakeData(0x203), (2, 3, 3, 5), np.float32)

    assert len(fake_library.CLBlastSgemm.calls) == 6


def test_double_pointer_batch_selects_dgemm_and_propagates_status(monkeypatch):
    library = FakeLibrary(
        status=-36,
        symbols=("CLBlastSgemm", "CLBlastDgemm", "CLBlastDgemmBatched"),
    )
    monkeypatch.setattr(module, "_load_library", lambda _: library)
    adapter = module.CLBlast("mock")
    a = FakeArray((2, 2, 3), np.float64, 0x201)
    b = FakeArray((2, 3, 2), np.float64, 0x202)

    with pytest.raises(RuntimeError, match="CLBlastDgemmBatched.*-36"):
        adapter.matmul(FakeQueue(), a, b, FakeData(0x203), (2, 2, 2), np.float64)

    assert not library.CLBlastSgemm.calls
    assert len(library.CLBlastDgemmBatched.calls) == 1


def test_float64_uses_dgemm(fake_library):
    adapter = module.CLBlast("mock")
    a = FakeArray((2, 3), np.float64, 0x201)
    b = FakeArray((3, 2), np.float64, 0x202)
    out = FakeData(0x203)

    adapter.matmul(FakeQueue(), a, b, out, (2, 2), np.float64)

    assert len(fake_library.CLBlastSgemm.calls) == 0
    assert len(fake_library.CLBlastDgemm.calls) == 1
    assert _value(fake_library.CLBlastDgemm.calls[0][6]) == 1.0
    assert _value(fake_library.CLBlastDgemm.calls[0][13]) == 0.0


def test_zero_dimension_is_skipped_without_call(fake_library):
    adapter = module.CLBlast("mock")
    a = FakeArray((0, 3), np.float32, 0x201)
    b = FakeArray((3, 2), np.float32, 0x202)
    out = FakeData(0x203)

    assert adapter.matmul(FakeQueue(), a, b, out, (0, 2), np.float32) is None
    assert not fake_library.CLBlastSgemm.calls


def test_unsupported_dtype_and_nonzero_status_raise(fake_library):
    adapter = module.CLBlast("mock")
    a = FakeArray((2, 2), np.int32, 0x201)
    b = FakeArray((2, 2), np.int32, 0x202)
    out = FakeArray((2, 2), np.int32, 0x203)
    with pytest.raises(UnsupportedDtypeError, match="float32 and float64"):
        adapter.matmul(FakeQueue(), a, b, out, (2, 2), np.int32)

    failing = FakeLibrary(status=-36)
    adapter._library = failing
    adapter._gemm["CLBlastSgemm"] = failing.CLBlastSgemm
    with pytest.raises(RuntimeError, match="status code -36"):
        adapter.matmul(
            FakeQueue(),
            FakeArray((2, 2), np.float32, 0x201),
            FakeArray((2, 2), np.float32, 0x202),
            FakeData(0x203),
            (2, 2),
            np.float32,
        )


@pytest.fixture(scope="module")
def native_clblast_backend():
    """Use a caller-provided DLL for optional public OpenCL regression tests."""

    path = os.environ.get("OPENPACHO_CLBLAST_LIBRARY")
    if not path:
        pytest.skip("set OPENPACHO_CLBLAST_LIBRARY to run native CLBlast tests")
    try:
        return get_backend("opencl", blas="clblast", clblast_library=path)
    except BackendUnavailableError as exc:
        pytest.skip(f"native CLBlast/OpenCL unavailable: {exc}")


@pytest.mark.opencl
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("ta,tb", [(True, False), (False, True), (True, True)])
def test_native_compiled_transpose_flags(native_clblast_backend, monkeypatch, dtype, ta, tb):
    backend = native_clblast_backend
    if dtype == np.float64 and not backend.fp64:
        pytest.skip("device does not support float64")
    rng = np.random.default_rng(395)
    left = rng.uniform(-0.5, 0.5, (33, 35) if ta else (35, 33)).astype(dtype)
    right = rng.uniform(-0.5, 0.5, (39, 33) if tb else (33, 39)).astype(dtype)
    a, b = op.array(left, backend=backend), op.array(right, backend=backend)
    replacement = op.array(left * 2, backend=backend)
    compiled = op.compile(lambda a, b: (a.T if ta else a) @ (b.T if tb else b))
    prepared = compiled.prepare(a, b)
    assert [s["kind"] for s in prepared.explain()["steps"]] == ["layout_matmul"]

    def forbidden(*args, **kwargs):
        raise AssertionError("private transpose product must not gather or download")

    with monkeypatch.context() as context:
        context.setattr(backend, "gather", forbidden)
        context.setattr(backend, "download", forbidden)
        result = prepared(a, b)
        rebound = prepared(replacement, b)
        backend.synchronize()
    expected = (left.T if ta else left) @ (right.T if tb else right)
    tolerance = 5e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(op.asnumpy(result), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(op.asnumpy(rebound), expected * 2, rtol=tolerance, atol=tolerance)


@pytest.mark.opencl
@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("float32", "float64"))
@pytest.mark.parametrize(
    "left,right",
    [
        (
            np.array([1.25, -2.0, 0.5]),
            np.array([2.0, 3.0, -4.0]),
        ),
        (
            np.array([1.25, -2.0, 0.5]),
            np.array([[2.0, 3.0], [1.0, -4.0], [0.5, 2.0]]),
        ),
        (
            np.array([[1.25, -2.0, 0.5], [2.0, 3.0, -4.0]]),
            np.array([2.0, 3.0, -4.0]),
        ),
        (
            np.arange(6).reshape(2, 3),
            np.arange(12).reshape(3, 4),
        ),
    ],
    ids=("vector-vector", "vector-matrix", "matrix-vector", "matrix-matrix"),
)
def test_native_clblast_public_matmul(native_clblast_backend, dtype, left, right):
    if np.dtype(dtype) == np.dtype(np.float64) and not native_clblast_backend.fp64:
        pytest.skip("selected OpenCL device does not support float64")

    left = left.astype(dtype)
    right = right.astype(dtype)
    actual = op.matmul(
        op.array(left, backend=native_clblast_backend),
        op.array(right, backend=native_clblast_backend),
    )

    np.testing.assert_allclose(op.asnumpy(actual), left @ right, rtol=2e-5, atol=2e-6)


@pytest.mark.opencl
@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("float32", "float64"))
def test_native_batched_broadcast_gemm_handles_irregular_offsets_and_poisoned_output(
    native_clblast_backend, dtype
):
    """Batched GEMM must reuse broadcast input buffers and overwrite beta=0 output."""

    backend = native_clblast_backend
    if dtype == np.float64 and not backend.fp64:
        pytest.skip("selected OpenCL device does not support float64")
    left = np.arange(2 * 1 * 3 * 4, dtype=dtype).reshape(2, 1, 3, 4) / dtype(11)
    right = np.arange(1 * 3 * 4 * 5, dtype=dtype).reshape(1, 3, 4, 5) / dtype(13)
    expected = np.matmul(left, right)
    a = op.asarray(left, backend=backend)
    b = op.asarray(right, backend=backend)
    destination = op.full(expected.shape, np.nan, dtype=dtype, backend=backend)
    backend._blas.matmul(backend.queue, a, b, destination._data, expected.shape, np.dtype(dtype))
    np.testing.assert_allclose(op.asnumpy(destination), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.opencl
@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("float32", "float64"))
def test_native_clblast_public_zero_inner_dimension(native_clblast_backend, dtype):
    if np.dtype(dtype) == np.dtype(np.float64) and not native_clblast_backend.fp64:
        pytest.skip("selected OpenCL device does not support float64")

    left = np.empty((2, 0), dtype=dtype)
    right = np.empty((0, 3), dtype=dtype)
    actual = op.matmul(
        op.array(left, backend=native_clblast_backend),
        op.array(right, backend=native_clblast_backend),
    )

    np.testing.assert_array_equal(op.asnumpy(actual), np.zeros((2, 3), dtype=dtype))


@pytest.mark.opencl
@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("float32", "float64"))
@pytest.mark.parametrize("transpose", (False, True))
@pytest.mark.parametrize("poison", (np.nan, np.inf), ids=("nan", "inf"))
def test_native_gemv_ignores_nonfinite_destination(
    native_clblast_backend, dtype, transpose, poison
):
    """beta=0 must not consume stale destination values, including nonfinites."""
    backend = native_clblast_backend
    if dtype == np.float64 and not backend.fp64:
        pytest.skip("selected OpenCL device does not support float64")
    prefix = "S" if dtype == np.float32 else "D"
    if f"CLBlast{prefix}gemv" not in backend._blas._vectors:
        pytest.skip("native CLBlast GEMV is unavailable")
    size = 2048
    matrix = np.diag(np.full(size, 2.5, dtype=dtype))
    vector = np.linspace(-1, 1, size, dtype=dtype)
    a, b = [op.asarray(x, backend=backend) for x in (matrix, vector)]
    if transpose:
        a, b = b, a
    destination = op.full((size,), poison, dtype=dtype, backend=backend)
    backend._blas.matmul(backend.queue, a, b, destination._data, (size,), np.dtype(dtype))
    np.testing.assert_allclose(op.asnumpy(destination), vector * dtype(2.5), rtol=2e-5, atol=2e-6)


@pytest.mark.opencl
@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("float32", "float64"))
def test_native_gemv_cg_with_recycled_nonfinite_buffers(native_clblast_backend, dtype):
    backend = native_clblast_backend
    if dtype == np.float64 and not backend.fp64:
        pytest.skip("selected OpenCL device does not support float64")
    size = 2048
    matrix = np.diag(np.full(size, 2.5, dtype=dtype))
    matrix += np.diag(np.full(size - 1, -1, dtype=dtype), 1)
    matrix += np.diag(np.full(size - 1, -1, dtype=dtype), -1)
    exact = np.random.default_rng(47).uniform(-0.5, 0.5, size).astype(dtype)
    rhs = dtype(2.5) * exact
    rhs[:-1] -= exact[1:]
    rhs[1:] -= exact[:-1]
    a, b = [op.asarray(x, backend=backend) for x in (matrix, rhs)]
    tolerance = 1e-4 if dtype == np.float32 else 1e-7
    for _ in range(4):
        recycled = [op.full((size,), np.nan, dtype=dtype, backend=backend) for _ in range(40)]
        backend.synchronize()
        del recycled
        result = solvers.conjugate_gradient(a, b, rtol=tolerance, maxiter=100)
        assert result.converged, (result.reason, result.iterations, result.residual_history)
        x = op.asnumpy(result.x)
        residual = rhs - dtype(2.5) * x
        residual[:-1] += x[1:]
        residual[1:] += x[:-1]
        assert np.linalg.norm(residual) <= tolerance * np.linalg.norm(rhs) * 1.1
