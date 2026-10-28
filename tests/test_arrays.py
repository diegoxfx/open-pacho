"""NumPy semantic conformance tests for the public array API."""

from __future__ import annotations

import numpy as np
import pytest

import openpacho as op
from openpacho.backends import get_backend
from openpacho.exceptions import BackendMismatchError, UnsupportedDtypeError


def _array(value, backend, dtype=None):
    return op.array(value, dtype=dtype, backend=backend)


def _host(value):
    return op.asnumpy(value)


def _skip_if_unsupported_dtype(backend, dtype):
    dtype = np.dtype(dtype)
    if backend.name == "opencl" and dtype == np.dtype(np.float64) and not backend.fp64:
        pytest.skip("selected OpenCL device does not support float64")


def _assert_result(actual, expected, *, rtol=1e-5, atol=1e-6):
    expected = np.asarray(expected)
    result = _host(actual)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype, (result.dtype, expected.dtype)
    if expected.dtype.kind in "fc":
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol, equal_nan=True)
    else:
        np.testing.assert_array_equal(result, expected)


def test_metadata_roundtrip_and_host_copy(backend):
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    array = _array(source, backend)

    source[...] = -1
    assert array.shape == (2, 3)
    assert array.dtype == np.dtype(np.float32)
    assert array.ndim == 2
    assert array.size == 6
    assert array.nbytes == source.nbytes
    assert array.backend == backend.name
    assert array.device == backend.device
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(_host(array), expected)

    downloaded = _host(array)
    downloaded[...] = 99
    np.testing.assert_array_equal(_host(array), expected)


def test_constructors_match_numpy(backend):
    expected_base = np.arange(6, dtype=np.float32).reshape(2, 3)
    cases = (
        (op.zeros((2, 3), dtype=np.float32, backend=backend), np.zeros((2, 3), dtype=np.float32)),
        (op.ones((2, 3), dtype=np.float32, backend=backend), np.ones((2, 3), dtype=np.float32)),
        (
            op.full((2, 3), -4, dtype=np.int32, backend=backend),
            np.full((2, 3), -4, dtype=np.int32),
        ),
        (
            op.arange(-2, 6, 2, dtype=np.int32, backend=backend),
            np.arange(-2, 6, 2, dtype=np.int32),
        ),
        (
            op.linspace(-1, 1, num=5, dtype=np.float32, backend=backend),
            np.linspace(-1, 1, num=5, dtype=np.float32),
        ),
        (
            op.eye(3, 4, k=-1, dtype=np.float32, backend=backend),
            np.eye(3, 4, k=-1, dtype=np.float32),
        ),
        (
            op.zeros_like(_array(expected_base, backend)),
            np.zeros_like(expected_base),
        ),
        (
            op.ones_like(_array(expected_base, backend)),
            np.ones_like(expected_base),
        ),
    )
    for actual, expected in cases:
        _assert_result(actual, expected)

    empty = op.empty((2, 3), dtype=np.float32, backend=backend)
    assert empty.shape == (2, 3)
    assert empty.dtype == np.dtype(np.float32)
    assert empty.nbytes == 2 * 3 * np.dtype(np.float32).itemsize


@pytest.mark.parametrize(
    "dtype, values",
    (
        (np.bool_, [False, True, True]),
        (np.int32, [-3, 0, 5]),
        (np.int64, [-3, 0, 5]),
        (np.uint32, [0, 3, 5]),
        (np.uint64, [0, 3, 5]),
        (np.float32, [-3.5, 0, 5.25]),
        (np.float64, [-3.5, 0, 5.25]),
    ),
)
def test_supported_dtype_roundtrip(backend, dtype, values):
    _skip_if_unsupported_dtype(backend, dtype)
    expected = np.asarray(values, dtype=dtype)
    _assert_result(_array(expected, backend), expected)


def test_scalar_and_empty_shapes(backend):
    scalar_host = np.asarray(3.5, dtype=np.float32)
    scalar = _array(scalar_host, backend)
    assert scalar.shape == ()
    assert scalar.ndim == 0
    assert scalar.size == 1
    assert scalar.nbytes == np.dtype(np.float32).itemsize
    assert scalar.item() == pytest.approx(3.5)
    _assert_result(scalar + 2, np.asarray(5.5, dtype=np.float32))

    empty_host = np.empty((0, 3), dtype=np.float32)
    empty = _array(empty_host, backend)
    assert empty.shape == (0, 3)
    assert empty.size == 0
    assert empty.nbytes == 0
    _assert_result(empty + 1, empty_host)
    _assert_result(empty * 2, empty_host)
    _assert_result(op.sum(empty), np.sum(empty_host))
    mean = _host(op.mean(empty))
    assert mean.shape == ()
    assert np.isnan(mean)


def test_elementwise_broadcasting_and_reverse_scalars(backend):
    host = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bias = np.asarray([0.5, -1, 2], dtype=np.float32)
    array = _array(host, backend)
    other = _array(bias, backend)

    for actual, expected in (
        (array + other, host + bias),
        (array - other, host - bias),
        (array * other, host * bias),
        (array / (other + 3), host / (bias + 3)),
        (2 + array, 2 + host),
        (2 - array, 2 - host),
        (2 * array, 2 * host),
        (2 / array, 2 / host),
        (array**2.0, host**2.0),
    ):
        _assert_result(actual, expected)

    weak_scalar = array + 0.5
    assert weak_scalar.dtype == np.dtype(np.float32)
    _assert_result(weak_scalar, host + 0.5)


def test_elementwise_ufuncs_comparisons_and_dispatch(backend):
    values = np.asarray([-2.0, -0.25, 0.5, 2.0], dtype=np.float32)
    positive_values = np.asarray([0.25, 1, 2, 4], dtype=np.float32)
    array = _array(values, backend)
    positive = _array(positive_values, backend)

    for name in ("sqrt", "exp", "log", "sin", "cos", "tanh", "abs"):
        function = getattr(op, name)
        source = positive if name in ("sqrt", "log") else array
        expected = getattr(np, name)(positive_values if name in ("sqrt", "log") else values)
        _assert_result(function(source), expected)

    for actual, expected in (
        (array > 0, values > 0),
        (array <= 0, values <= 0),
        (array == 0, values == 0),
        (op.isfinite(array), np.isfinite(values)),
        (np.isfinite(array), np.isfinite(values)),
    ):
        _assert_result(actual, expected)


def test_division_preserves_nan_and_infinity(backend):
    numerator = np.asarray([1.0, 0.0, -1.0], dtype=np.float32)
    denominator = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.divide(numerator, denominator)
    actual = _host(_array(numerator, backend) / _array(denominator, backend))
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.isposinf(actual[0])
    assert np.isnan(actual[1])
    assert np.isneginf(actual[2])


def test_noncontiguous_upload_and_basic_slicing_are_copies(backend):
    contiguous = np.arange(24, dtype=np.float32).reshape(4, 6)
    noncontiguous = contiguous[:, ::2]
    array = _array(noncontiguous, backend)
    np.testing.assert_array_equal(_host(array), noncontiguous)

    sliced = array[::-1, 1:5:2]
    expected = noncontiguous[::-1, 1:5:2]
    _assert_result(sliced, expected)
    downloaded = _host(sliced)
    downloaded[...] = -100
    _assert_result(sliced, expected)

    newaxis = array[None, :, 0]
    _assert_result(newaxis, noncontiguous[None, :, 0])

    with pytest.raises(TypeError):
        array[0, 0] = 1


def test_transpose_and_reshape_match_numpy(backend):
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    array = _array(source, backend)
    _assert_result(array.T, source.T)
    _assert_result(array.transpose(1, 0), source.transpose(1, 0))
    _assert_result(array.reshape(3, 2), source.reshape(3, 2))
    _assert_result(op.reshape(array, (6,)), source.reshape(6))


def test_reductions_match_numpy_for_large_arrays_and_axis_tuples(backend):
    source = (np.arange(17 * 19, dtype=np.float32).reshape(17, 19) - 100) / 7
    array = _array(source, backend)
    for axis, keepdims in ((None, False), (0, False), (1, True), ((0, 1), False), ((0, 1), True)):
        _assert_result(
            op.sum(array, axis=axis, keepdims=keepdims),
            np.sum(source, axis=axis, keepdims=keepdims),
        )
        _assert_result(
            op.mean(array, axis=axis, keepdims=keepdims),
            np.mean(source, axis=axis, keepdims=keepdims),
            rtol=2e-5,
            atol=2e-6,
        )


@pytest.mark.parametrize("dtype", (np.int32, np.uint32))
def test_integer_reduction_default_dtypes(backend, dtype):
    source = np.arange(17 * 19, dtype=dtype).reshape(17, 19)
    array = _array(source, backend)
    expected = np.sum(source, axis=0)
    actual = op.sum(array, axis=0)
    assert actual.dtype == expected.dtype
    _assert_result(actual, expected)


def test_reduction_validation(backend):
    array = _array(np.arange(6, dtype=np.float32).reshape(2, 3), backend)
    with pytest.raises(ValueError):
        op.sum(array, axis=(0, 0))
    with pytest.raises(ValueError):
        op.sum(array, axis=2)
    with pytest.raises(NotImplementedError):
        op.sum(array, out=array)


@pytest.mark.parametrize(
    "left, right",
    (
        (np.asarray([1, 2, 3], dtype=np.float32), np.asarray([4, 5, 6], dtype=np.float32)),
        (
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(12, dtype=np.float32).reshape(3, 4),
        ),
        (
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.asarray([1, 2, 3], dtype=np.float32),
        ),
        (
            np.asarray([1, 2, 3], dtype=np.float32),
            np.arange(12, dtype=np.float32).reshape(3, 4),
        ),
    ),
)
def test_dot_and_matmul_rank_combinations(backend, left, right):
    expected = np.matmul(left, right)
    actual = op.matmul(_array(left, backend), _array(right, backend))
    _assert_result(actual, expected, rtol=2e-5, atol=2e-6)
    _assert_result(
        op.dot(_array(left, backend), _array(right, backend)),
        np.dot(left, right),
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "left, right",
    (
        (np.zeros((0, 3), dtype=np.float32), np.zeros((3, 2), dtype=np.float32)),
        (np.zeros((2, 0), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)),
        (np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)),
        (np.zeros((2, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)),
    ),
)
def test_zero_dimension_matmul(backend, left, right):
    expected = np.matmul(left, right)
    actual = op.matmul(_array(left, backend), _array(right, backend))
    _assert_result(actual, expected)


def test_matmul_validation(backend):
    left = _array(np.ones((2, 3), dtype=np.float32), backend)
    with pytest.raises(ValueError):
        op.matmul(left, _array(np.ones((2, 2), dtype=np.float32), backend))
    with pytest.raises(ValueError):
        op.matmul(_array(np.ones((1, 1, 1), dtype=np.float32), backend), left)


def test_diag_extracts_two_dimensional_matrix_diagonals(backend):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    array = _array(source, backend)
    for k in (-2, -1, 0, 1, 3):
        _assert_result(op.diag(array, k=k), np.diag(source, k=k))
    with pytest.raises(NotImplementedError):
        op.diag(_array(np.arange(3, dtype=np.float32), backend))


def test_implicit_host_conversion_and_unsupported_dispatch_are_explicit(backend):
    array = _array(np.arange(3, dtype=np.float32), backend)
    with pytest.raises(TypeError):
        np.asarray(array)
    with pytest.raises(TypeError):
        np.array(array)
    with pytest.raises(ValueError):
        bool(array)
    with pytest.raises(TypeError):
        np.concatenate((array, array))
    with pytest.raises(TypeError):
        np.stack((array, array))
    with pytest.raises(TypeError):
        np.add(array, array, out=np.empty(3, dtype=np.float32))


def test_integer_power_is_explicitly_unsupported(backend):
    array = _array(np.asarray([1, 2, 3], dtype=np.int32), backend)
    with pytest.raises(NotImplementedError):
        op.power(array, 2)


def test_numpy_and_opencl_backend_mismatch_is_rejected(opencl_backend):
    numpy_backend = get_backend("numpy")
    numpy_array = _array([1, 2, 3], numpy_backend, dtype=np.float32)
    opencl_array = _array([1, 2, 3], opencl_backend, dtype=np.float32)
    with pytest.raises(BackendMismatchError):
        numpy_array + opencl_array


def test_to_backend_roundtrip_is_explicit(opencl_backend):
    numpy_backend = get_backend("numpy")
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    host_array = _array(source, numpy_backend)
    device_array = op.to_backend(host_array, opencl_backend)
    assert device_array.backend == "opencl"
    restored = op.to_backend(device_array, numpy_backend)
    assert restored.backend == "numpy"
    _assert_result(restored, source)


def test_distinct_opencl_devices_reject_mixed_contexts(opencl_backend):
    del opencl_backend
    device_ids = [info.id for info in op.devices()]
    if len(device_ids) < 2:
        pytest.skip("requires at least two OpenCL devices")
    first = get_backend("opencl", device=device_ids[0])
    second = get_backend("opencl", device=device_ids[1])
    left = _array([1, 2], first, dtype=np.float32)
    right = _array([3, 4], second, dtype=np.float32)
    with pytest.raises(BackendMismatchError):
        left + right


def test_opencl_float64_capability_is_reported(opencl_backend):
    if opencl_backend.fp64:
        array = _array([1.0], opencl_backend, dtype=np.float64)
        assert array.dtype == np.dtype(np.float64)
    else:
        with pytest.raises(UnsupportedDtypeError):
            _array([1.0], opencl_backend, dtype=np.float64)
