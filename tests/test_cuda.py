"""Focused conformance tests for the initial native CUDA backend.

The CUDA cases are intentionally small enough to run as a smoke suite on a
real device.  They skip when CUDA is unavailable, unless the caller requests a
required CUDA run with ``PACHO_REQUIRE_CUDA``.  The CPU subprocess case remains
executable on every machine and guards the optional import boundary.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import textwrap
import weakref

import numpy as np
import pytest

import openpacho as pacho
from examples import solvers
from openpacho.exceptions import (
    BackendMismatchError,
    BackendUnavailableError,
    UnsupportedDtypeError,
)


def _cuda_required() -> bool:
    return os.environ.get("PACHO_REQUIRE_CUDA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cuda_device() -> str:
    return os.environ.get("PACHO_CUDA_DEVICE", "0") or "0"


def _array(value, backend, dtype=None):
    return pacho.asarray(value, dtype=dtype, backend=backend)


def _check_dtype(backend, dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float64") and not getattr(backend, "fp64", True):
        pytest.skip("selected CUDA device does not support float64")
    return backend.validate_dtype(dtype)


def _assert_result(actual, expected, *, rtol=2e-5, atol=2e-6):
    expected = np.asarray(expected)
    result = pacho.asnumpy(actual)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype, (result.dtype, expected.dtype)
    if expected.dtype.kind in "fc":
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol, equal_nan=True)
    else:
        np.testing.assert_array_equal(result, expected)


@pytest.fixture(scope="module")
def cuda_backend(request):
    """Return the default cuBLAS backend, honoring the required flag."""

    del request
    try:
        backend = pacho.get_backend("cuda", device=_cuda_device())
    except (BackendUnavailableError, ImportError) as exc:
        message = f"CUDA unavailable{f' ({exc})' if str(exc) else ''}"
        if _cuda_required():
            pytest.fail(message)
        pytest.skip(message)
    yield backend
    backend.synchronize()


@pytest.fixture(scope="module", params=("cublas", "custom"), ids=("cublas", "custom"))
def cuda_provider(request):
    """Return each BLAS provider for the product conformance cases."""

    try:
        backend = pacho.get_backend("cuda", device=_cuda_device(), blas=request.param)
    except (BackendUnavailableError, ImportError) as exc:
        message = f"CUDA unavailable{f' ({exc})' if str(exc) else ''}"
        if _cuda_required():
            pytest.fail(message)
        pytest.skip(message)
    yield backend
    backend.synchronize()


def test_cuda_configuration_rejects_foreign_blas_providers_without_initializing_cuda():
    with pytest.raises(ValueError):
        pacho.get_backend("cuda", blas="clblast")
    with pytest.raises(ValueError):
        pacho.get_backend("cuda", clblast_library="not-a-cuda-library")
    with pytest.raises(ValueError):
        pacho.get_backend("opencl", blas="cublas")


def test_cuda_backend_identity_and_provider_are_explicit(cuda_backend):
    assert type(cuda_backend).__name__ == "CUDABackend"
    assert cuda_backend.name == "cuda"
    assert cuda_backend.blas == "cublas"
    selected = pacho.get_backend("cuda", device=_cuda_device())
    assert selected is cuda_backend


@pytest.mark.parametrize(
    "dtype", (np.bool_, np.float32, np.float64), ids=("bool", "float32", "float64")
)
def test_cuda_supported_dtype_roundtrip_has_native_resident_storage(cuda_backend, dtype):
    _check_dtype(cuda_backend, dtype)
    if np.dtype(dtype) == np.dtype("bool"):
        host = np.asarray([[True, False, True], [False, True, False]], dtype=dtype)
    else:
        host = np.asarray([[-2.5, 0.0, 3.25], [4.5, -1.25, 0.75]], dtype=dtype)
    value = _array(host, cuda_backend)

    assert value.backend == "cuda"
    assert value.device == cuda_backend.device
    assert value.shape == host.shape
    assert value.dtype == np.dtype(dtype)
    assert isinstance(value._data.ptr, int) and value._data.ptr > 0
    assert isinstance(value._data.capacity, int) and value._data.capacity >= value.nbytes
    _assert_result(value, host)


def test_cuda_integer_dtypes_are_rejected(cuda_backend):
    for dtype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ):
        with pytest.raises(UnsupportedDtypeError):
            cuda_backend.validate_dtype(dtype)
        with pytest.raises(UnsupportedDtypeError):
            _array([1, 2, 3], cuda_backend, dtype=dtype)


def test_cuda_integer_inference_is_rejected(cuda_backend):
    with pytest.raises(UnsupportedDtypeError):
        _array([1, 2, 3], cuda_backend)
    with pytest.raises(UnsupportedDtypeError):
        pacho.zeros((3,), backend=cuda_backend, dtype=np.int32)


def test_cuda_upload_snapshots_mutable_and_noncontiguous_host_inputs(cuda_backend):
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    view = source[:, ::2]
    expected = view.copy()
    value = _array(view, cuda_backend)
    source[...] = -99
    _assert_result(value, expected)


def test_cuda_scalar_and_empty_arrays_keep_shape_and_dtype(cuda_backend):
    scalar_host = np.asarray(3.5, dtype=np.float32)
    scalar = _array(scalar_host, cuda_backend)
    empty_host = np.empty((0, 3), dtype=np.float32)
    empty = _array(empty_host, cuda_backend)

    assert scalar.shape == () and scalar.ndim == 0 and scalar.size == 1
    assert empty.shape == (0, 3) and empty.size == 0 and empty.nbytes == 0
    _assert_result(scalar + 2.0, np.asarray(5.5, dtype=np.float32))
    _assert_result(empty + 1.0, empty_host)


def test_cuda_arithmetic_and_broadcasting_match_numpy(cuda_backend):
    left_host = np.asarray([[1.25, -2.0, 0.5], [2.5, 3.0, -4.0]], dtype=np.float32)
    right_host = np.asarray([0.5, -1.0, 2.0], dtype=np.float32)
    left, right = _array(left_host, cuda_backend), _array(right_host, cuda_backend)
    with np.errstate(all="ignore"):
        cases = (
            (left + right, left_host + right_host),
            (left - right, left_host - right_host),
            (left * right, left_host * right_host),
            (left / (right + 3.0), left_host / (right_host + 3.0)),
            (2.0 + left, 2.0 + left_host),
            (2.0 - left, 2.0 - left_host),
            (2.0 * left, 2.0 * left_host),
            (2.0 / left, 2.0 / left_host),
            (-left, -left_host),
            (pacho.sqrt(pacho.absolute(left)), np.sqrt(np.abs(left_host))),
        )
    for actual, expected in cases:
        _assert_result(actual, expected)


def test_cuda_boolean_comparisons_and_isfinite_stay_on_cuda(cuda_backend):
    host = np.asarray([[-2.0, 0.0, np.inf], [np.nan, 1.0, 3.0]], dtype=np.float32)
    value = _array(host, cuda_backend)
    for actual, expected in (
        (value > 0, host > 0),
        (value <= 0, host <= 0),
        (value == 0, host == 0),
        (pacho.isfinite(value), np.isfinite(host)),
    ):
        assert actual.backend == "cuda"
        _assert_result(actual, expected)


def test_cuda_cast_preserves_values_and_target_dtype(cuda_backend):
    for source_dtype, target_dtype in ((np.float32, np.float64), (np.float64, np.float32)):
        _check_dtype(cuda_backend, source_dtype)
        _check_dtype(cuda_backend, target_dtype)
        host = np.asarray([-3.5, 0.0, 2.25, np.inf, np.nan], dtype=source_dtype)
        source = _array(host, cuda_backend)
        cast = source.astype(target_dtype)
        _assert_result(cast, host.astype(target_dtype), rtol=2e-6, atol=2e-6)
        assert cast.backend == "cuda" and cast.dtype == np.dtype(target_dtype)
        with pytest.raises(UnsupportedDtypeError):
            source.astype(np.int32)


def test_cuda_gather_slicing_and_newaxis_are_copies(cuda_backend):
    host = np.arange(24, dtype=np.float32).reshape(4, 6)
    value = _array(host, cuda_backend)
    sliced = value[::-1, 1:6:2]
    expanded = value[..., ::-1][None, :, 0]

    _assert_result(sliced, host[::-1, 1:6:2])
    _assert_result(expanded, host[..., ::-1][None, :, 0])
    downloaded = pacho.asnumpy(sliced)
    downloaded[...] = -100
    _assert_result(sliced, host[::-1, 1:6:2])


def test_cuda_transpose_matches_numpy(cuda_backend):
    host = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    value = _array(host, cuda_backend)
    _assert_result(value.T, host.T)
    _assert_result(value.transpose(2, 0, 1), host.transpose(2, 0, 1))


def test_cuda_sum_and_mean_support_arbitrary_axes(cuda_backend):
    for dtype in (np.float32, np.float64):
        _check_dtype(cuda_backend, dtype)
        host = (np.arange(3 * 4 * 5, dtype=dtype).reshape(3, 4, 5) - dtype(20)) / dtype(7)
        value = _array(host, cuda_backend)
        for axis, keepdims in (
            (None, False),
            (0, False),
            (-1, True),
            ((0, 2), False),
            ((1,), True),
        ):
            sum_expected = np.sum(host, axis=axis, keepdims=keepdims)
            mean_expected = np.mean(host, axis=axis, keepdims=keepdims)
            _assert_result(pacho.sum(value, axis=axis, keepdims=keepdims), sum_expected, rtol=4e-5)
            _assert_result(
                pacho.mean(value, axis=axis, keepdims=keepdims), mean_expected, rtol=4e-5
            )


def test_cuda_bool_sum_default_int64_is_rejected(cuda_backend):
    value = _array([[True, False], [True, True]], cuda_backend)
    with pytest.raises(UnsupportedDtypeError):
        pacho.sum(value)


def test_cuda_bool_sum_with_supported_float_accumulator_works(cuda_backend):
    host = np.asarray([[True, False], [True, True]], dtype=np.bool_)
    value = _array(host, cuda_backend)
    _assert_result(
        pacho.sum(value, axis=0, dtype=np.float32), np.sum(host, axis=0, dtype=np.float32)
    )


_MATMUL_CASES = (
    (
        "vector",
        np.asarray([1.25, -2.0, 0.5, 3.0, -1.0, 2.5, 4.0]),
        np.asarray([-2.0, 3.0, 1.5, -0.5, 2.0, 0.25, -1.0]),
    ),
    (
        "gemv",
        np.arange(3 * 5, dtype=np.float64).reshape(3, 5) / 11,
        np.asarray([1.0, -2.0, 0.5, 3.0, -1.5]),
    ),
    (
        "vecmat",
        np.asarray([1.0, -2.0, 0.5, 3.0, -1.5]),
        np.arange(5 * 4, dtype=np.float64).reshape(5, 4) / 13,
    ),
    (
        "irregular_gemm",
        np.arange(5 * 7, dtype=np.float64).reshape(5, 7) / 17,
        np.arange(7 * 3, dtype=np.float64).reshape(7, 3) / 19,
    ),
    (
        "odd_tile_gemm",
        np.arange(33 * 17, dtype=np.float64).reshape(33, 17) / 101,
        np.arange(17 * 65, dtype=np.float64).reshape(17, 65) / 103,
    ),
    (
        "large_odd_gemm",
        np.arange(65 * 63, dtype=np.float64).reshape(65, 63) / 107,
        np.arange(63 * 97, dtype=np.float64).reshape(63, 97) / 109,
    ),
)


def test_cuda_vector_and_irregular_matrix_products(cuda_provider):
    for dtype in (np.float32, np.float64):
        _check_dtype(cuda_provider, dtype)
        for _, left_host, right_host in _MATMUL_CASES:
            left_host, right_host = left_host.astype(dtype), right_host.astype(dtype)
            left, right = _array(left_host, cuda_provider), _array(right_host, cuda_provider)
            actual = pacho.matmul(left, right)
            expected = np.matmul(left_host, right_host)
            tolerance = (3e-5, 3e-6) if dtype == np.dtype("float32") else (2e-11, 2e-12)
            _assert_result(actual, expected, rtol=tolerance[0], atol=tolerance[1])


_EMPTY_MATMUL_CASES = (
    ("zero_rows", (0, 5), (5, 3)),
    ("zero_inner", (2, 0), (0, 3)),
    ("row_k0", (1, 0), (0, 7)),
    ("zero_columns", (4, 3), (3, 0)),
    ("empty_vector", (0,), (0,)),
)


def test_cuda_empty_and_k0_products_are_defined(cuda_backend):
    for dtype in (np.float32, np.float64):
        _check_dtype(cuda_backend, dtype)
        for _, left_shape, right_shape in _EMPTY_MATMUL_CASES:
            left_host = np.zeros(left_shape, dtype=dtype)
            right_host = np.zeros(right_shape, dtype=dtype)
            actual = pacho.matmul(_array(left_host, cuda_backend), _array(right_host, cuda_backend))
            _assert_result(actual, np.matmul(left_host, right_host))


def _poisoned_matmul(backend, left_host, right_host, dtype, poison, monkeypatch):
    left_host, right_host = left_host.astype(dtype), right_host.astype(dtype)
    expected = np.matmul(left_host, right_host)
    left, right = _array(left_host, backend), _array(right_host, backend)
    destination = pacho.full(expected.shape, poison, dtype=dtype, backend=backend)
    original_empty = backend.empty

    def poisoned_empty(shape, requested_dtype):
        if tuple(shape) == expected.shape and np.dtype(requested_dtype) == np.dtype(dtype):
            return destination._data
        return original_empty(shape, requested_dtype)

    monkeypatch.setattr(backend, "empty", poisoned_empty)
    actual = pacho.matmul(left, right)
    assert actual._data is destination._data
    return actual, expected


def test_cuda_products_overwrite_nan_and_inf_destinations(cuda_provider, monkeypatch):
    poison_cases = _MATMUL_CASES[0:1] + _MATMUL_CASES[1:3] + _MATMUL_CASES[3:4]
    for dtype in (np.float32, np.float64):
        _check_dtype(cuda_provider, dtype)
        for poison in (np.nan, np.inf):
            for _, left_host, right_host in poison_cases:
                with monkeypatch.context() as patch:
                    actual, expected = _poisoned_matmul(
                        cuda_provider, left_host, right_host, dtype, poison, patch
                    )
                    _assert_result(
                        actual,
                        expected,
                        rtol=3e-5 if dtype == np.dtype("float32") else 2e-11,
                    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shapes",
    [
        ((3, 33, 17), (3, 17, 65)),
        ((2, 1, 33, 17), (1, 3, 17, 65)),
        ((2, 5, 7), (7, 3)),
        ((5, 7), (2, 7, 3)),
        ((2, 5, 7), (7,)),
        ((7,), (2, 7, 3)),
        ((2, 5, 0), (2, 0, 3)),
        ((0, 5, 7), (1, 7, 3)),
    ],
)
def test_cuda_batched_broadcast_products(cuda_provider, dtype, shapes):
    _check_dtype(cuda_provider, dtype)
    rng = np.random.default_rng(830)
    left, right = (rng.normal(size=s).astype(dtype) for s in shapes)
    a, b = _array(left, cuda_provider), _array(right, cuda_provider)
    operation = pacho.compile(lambda x, y: x @ y)
    for result in (a @ b, operation(a, b)):
        _assert_result(
            result,
            left @ right,
            rtol=3e-5 if dtype == np.float32 else 2e-11,
            atol=3e-5 if dtype == np.float32 else 2e-12,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cuda_regular_batch_uses_one_native_call_and_retains_buffers(
    cuda_backend, monkeypatch, dtype
):
    _check_dtype(cuda_backend, dtype)
    calls = []
    name = "sgemm_strided_batched" if dtype == np.float32 else "dgemm_strided_batched"
    original = getattr(cuda_backend._cublas, name)

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(cuda_backend._cublas, name, counted)
    a = _array(np.ones((3, 5, 7), dtype), cuda_backend)
    b = _array(np.ones((3, 7, 9), dtype), cuda_backend)
    c = a @ b
    assert len(calls) == 1
    assert all(x._data in cuda_backend._pending for x in (a, b, c))
    _assert_result(c, np.full((3, 5, 9), 7, dtype))


@pytest.mark.parametrize("shapes", [((3, 33, 17), (3, 17, 65)), ((2, 1, 33, 17), (1, 3, 17, 65))])
def test_cuda_tiled_batch_overwrites_poison_and_keeps_inputs_alive(
    cuda_provider, shapes, monkeypatch
):
    rng = np.random.default_rng(842)
    left, right = (rng.uniform(-0.5, 0.5, size=s).astype(np.float32) for s in shapes)
    actual, expected = _poisoned_matmul(cuda_provider, left, right, np.float32, np.nan, monkeypatch)
    assert actual._data in cuda_provider._pending
    _assert_result(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("length", [4096, 4097, 8193])
def test_cuda_norm_chunk_boundaries_and_nonfinite_precedence(cuda_provider, dtype, length):
    _check_dtype(cuda_provider, dtype)
    small, large = (1e-25, 1e25) if dtype == np.float32 else (1e-200, 1e200)
    host = np.zeros((6, length), dtype=dtype)
    host[0, [0, -1]] = large
    host[1, [0, -1]] = small
    host[2, -1] = np.inf
    host[3, 0], host[3, -1] = np.nan, np.inf
    host[4, 0], host[4, -1] = np.inf, np.nan
    host[5, [0, -1]] = -0.0
    value = _array(host, cuda_provider)
    calculate = pacho.compile(lambda a: pacho.linalg.norm(a, axis=1, keepdims=True))
    expected = np.array(
        [large * np.sqrt(2), small * np.sqrt(2), np.inf, np.nan, np.nan, 0.0], dtype=dtype
    ).reshape(6, 1)
    for result in (pacho.linalg.norm(value, axis=1, keepdims=True), calculate(value)):
        _assert_result(result, expected, rtol=2e-6, atol=0)
        assert result._backend is cuda_provider
        assert not np.signbit(pacho.asnumpy(result)[5, 0])


@pytest.mark.parametrize(
    "method,name", [("cg", "conjugate_gradient"), ("jacobi", "jacobi"), ("sd", "steepest_descent")]
)
def test_cuda_example_solvers_and_prepared_rebinding(cuda_provider, method, name):
    matrix = _array(np.diag([2, 3, 4]).astype(np.float32), cuda_provider)
    rhs = _array([2, 6, 12], cuda_provider, dtype=np.float32)
    ordinary = getattr(solvers, name)(matrix, rhs, rtol=1e-5, maxiter=100)
    system = solvers.prepare(matrix, method=method)
    for result, expected in (
        (ordinary, [1, 2, 3]),
        (system.solve(rhs, rtol=1e-5, maxiter=100), [1, 2, 3]),
        (system.solve(rhs * 2, rtol=1e-5, maxiter=100), [2, 4, 6]),
    ):
        assert result.converged and result.iterations > 0, result.reason
        assert len(result.residual_history) == result.iterations + 1
        assert result.x._backend is cuda_provider
        _assert_result(result.x, np.asarray(expected, dtype=np.float32), rtol=5e-5)
    with pytest.raises(ValueError, match="finite"):
        system.solve(_array([np.inf, 1, 2], cuda_provider, dtype=np.float32))
    if method == "jacobi":
        invalid = _array(np.diag([0, 1, 2]).astype(np.float32), cuda_provider)
        with pytest.raises(ValueError, match="diagonal"):
            solvers.prepare(invalid, method=method)


def test_cuda_operations_do_not_download_or_fall_back_to_numpy(cuda_backend, monkeypatch):
    host = np.arange(12, dtype=np.float32).reshape(3, 4) / 5
    value = _array(host, cuda_backend)
    original_asarray = np.asarray

    def forbid_array_conversion(candidate, *args, **kwargs):
        if isinstance(candidate, pacho.ndarray):
            raise TypeError("CUDA operation attempted an implicit NumPy conversion")
        return original_asarray(candidate, *args, **kwargs)

    def forbid_download(*args, **kwargs):
        raise AssertionError("CUDA operation attempted a host download")

    monkeypatch.setattr(np, "asarray", forbid_array_conversion)
    monkeypatch.setattr(cuda_backend, "download", forbid_download)
    results = (
        value + 1.0,
        np.add(value, 2.0),
        pacho.sum(value, axis=0),
        pacho.mean(value, axis=1),
        value[::-1, ::2],
        value.T,
        value @ value.T,
        pacho.linalg.norm(value),
        pacho.compile(lambda a: pacho.linalg.norm(a, axis=1))(value),
    )
    assert all(result.backend == "cuda" and result._backend is cuda_backend for result in results)


def test_cuda_and_numpy_arrays_require_explicit_transfer(cuda_backend):
    device_value = _array([1.0, 2.0, 3.0], cuda_backend, dtype=np.float32)
    numpy_value = _array([4.0, 5.0, 6.0], pacho.get_backend("numpy"), dtype=np.float32)
    with pytest.raises(BackendMismatchError):
        device_value + numpy_value
    with pytest.raises(BackendMismatchError):
        pacho.asarray(device_value, backend="numpy")
    restored = pacho.to_backend(device_value, "numpy")
    _assert_result(restored, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_cuda_bool_fill_rebinds_without_changing_previous_output(cuda_backend):
    first = pacho.full(33, True, dtype=np.bool_, backend=cuda_backend)
    second = pacho.full(33, False, dtype=np.bool_, backend=cuda_backend)
    np.testing.assert_array_equal(pacho.asnumpy(first), np.ones(33, dtype=np.bool_))
    np.testing.assert_array_equal(pacho.asnumpy(second), np.zeros(33, dtype=np.bool_))


def test_cuda_calls_restore_the_callers_current_context(cuda_backend):
    driver = cuda_backend.driver

    def current_context():
        status, context = driver.cuCtxGetCurrent()
        assert int(status) == 0
        try:
            return int(context)
        except (TypeError, ValueError):
            return repr(context)

    before = current_context()
    value = pacho.asarray(np.arange(8, dtype=np.float32), backend=cuda_backend)
    pacho.asnumpy(value + 1.0)
    compiled = pacho.compile(lambda x: ((x + 1) * (x - 2)).sum())
    for _ in range(3):
        _assert_result(compiled(value), np.asarray(96, dtype=np.float32))
        assert current_context() == before
    with pytest.raises(ValueError, match="scope failure"), cuda_backend.execution_scope():
        _assert_result(compiled(value), np.asarray(96, dtype=np.float32))
        raise ValueError("scope failure")
    assert cuda_backend._activation_depth == 0
    cuda_backend.synchronize()
    after = current_context()
    assert after == before


def test_cuda_synchronize_clears_pending_references(cuda_backend):
    value = pacho.asarray(np.arange(128, dtype=np.float32), backend=cuda_backend)
    result = value + 1.0
    assert cuda_backend._pending
    cuda_backend.synchronize()
    assert not cuda_backend._pending
    _assert_result(result, np.arange(128, dtype=np.float32) + 1)


def test_cuda_inflight_inputs_are_retained_until_synchronize(cuda_backend):
    value = pacho.asarray(np.arange(128, dtype=np.float32), backend=cuda_backend)
    pointer = value._data.ptr
    allocation = weakref.ref(value._data)
    value + 1.0
    del value
    gc.collect()
    assert allocation() is not None
    replacement = pacho.empty((128,), dtype=np.float32, backend=cuda_backend)
    assert replacement._data.ptr != pointer
    cuda_backend.synchronize()


def test_cuda_release_cached_memory_preserves_live_arrays_and_clears_pool(cuda_backend):
    live = pacho.asarray(np.arange(16, dtype=np.float32), backend=cuda_backend)
    temporary = pacho.asarray(np.arange(16, dtype=np.float32) + 1, backend=cuda_backend)
    del temporary
    gc.collect()
    cuda_backend.synchronize()
    cuda_backend.release_cached_memory()
    assert cuda_backend._pool_items == 0
    assert cuda_backend._cached_bytes == 0
    _assert_result(live, np.arange(16, dtype=np.float32))


def test_cuda_pool_respects_item_and_byte_limits(cuda_backend):
    previous_limit = cuda_backend._pool_limit
    try:
        cuda_backend.release_cached_memory()
        cuda_backend._pool_limit = 1024
        for _ in range(12):
            temporary = pacho.empty((128,), dtype=np.float32, backend=cuda_backend)
            del temporary
        gc.collect()
        cuda_backend.synchronize()
        assert cuda_backend._pool_items <= 128
        assert cuda_backend._cached_bytes <= cuda_backend._pool_limit
    finally:
        cuda_backend._pool_limit = previous_limit
        cuda_backend.release_cached_memory()


def test_cuda_close_is_idempotent_and_invalidates_live_arrays(cuda_backend):
    from openpacho._cuda import CUDABackend

    owned = CUDABackend(device=_cuda_device(), blas=cuda_backend.blas)
    live = pacho.asarray([1.0, 2.0, 3.0], dtype=np.float32, backend=owned)
    try:
        owned.close()
        owned.close()
        assert owned._closed
        assert owned._context is None
        assert owned._stream is None
        assert live._data.ptr == 0
        with pytest.raises(RuntimeError):
            pacho.asnumpy(live)
        with pytest.raises(RuntimeError):
            live + 1.0
    finally:
        owned.close()


def test_cuda_compile_rebinds_scalars_preserves_outputs_and_fuses(cuda_backend, monkeypatch):
    calls = []
    original_elementwise = cuda_backend.elementwise

    def track_elementwise(name, args, shape, dtype):
        calls.append(name)
        return original_elementwise(name, args, shape, dtype)

    monkeypatch.setattr(cuda_backend, "elementwise", track_elementwise)

    @pacho.compile
    def affine(value, scale):
        shifted = value * scale
        return shifted, shifted + 1.0

    host = np.arange(12, dtype=np.float32).reshape(3, 4)
    value = _array(host, cuda_backend)
    first, first_shifted = affine(value, 2.0)
    second, second_shifted = affine(value, 3.5)

    _assert_result(first, host * 2.0)
    _assert_result(first_shifted, host * 2.0 + 1.0)
    _assert_result(second, host * 3.5)
    _assert_result(second_shifted, host * 3.5 + 1.0)
    assert affine.cache_info()["misses"] == 1
    assert affine.cache_info()["hits"] >= 1
    assert not calls  # typed groups launch directly, without eager elementwise dispatch
    explanation = affine.explain(value, 4.0)
    assert explanation["backend"] == "cuda"
    assert [step["kind"] for step in explanation["steps"]] == ["fused", "fused"]
    _assert_result(first, host * 2.0)


def _run_installed_subprocess(script):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PACHO_REQUIRE_CUDA", None)
    return subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_cpu_import_does_not_load_optional_cuda_modules():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        forbidden = {"cuda", "nvmath", "cupy"}

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in forbidden:
                    raise RuntimeError(f"forbidden optional import: {fullname}")
                return None

        sys.meta_path.insert(0, BlockOptionalImports())
        import openpacho

        assert not any(name.partition(".")[0] in forbidden for name in sys.modules)
        assert openpacho.get_backend("numpy").name == "numpy"
        """
    )
    completed = _run_installed_subprocess(script)
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_cuda_gemm_does_not_import_cupy(cuda_provider):
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import numpy as np
        import sys

        class BlockCuPy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cupy" or fullname.startswith("cupy."):
                    raise RuntimeError(f"forbidden optional import: {{fullname}}")
                return None

        sys.meta_path.insert(0, BlockCuPy())
        import openpacho

        backend = openpacho.get_backend(
            "cuda", device={json.dumps(_cuda_device())}, blas={json.dumps(cuda_provider.blas)}
        )
        left = openpacho.asarray(np.arange(15, dtype=np.float32).reshape(3, 5), backend=backend)
        right = openpacho.asarray(np.arange(20, dtype=np.float32).reshape(5, 4), backend=backend)
        result = openpacho.matmul(left, right)
        np.testing.assert_allclose(openpacho.asnumpy(result), np.asarray(left.to_host()) @ np.asarray(right.to_host()))
        assert not any(name == "cupy" or name.startswith("cupy.") for name in sys.modules)
        backend.close()
        """
    )
    completed = _run_installed_subprocess(script)
    assert completed.returncode == 0, completed.stderr or completed.stdout
