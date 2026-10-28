"""Focused regression coverage for NumPy promotion and boundary semantics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import openpacho as pacho
from openpacho.backends import get_backend
from openpacho.exceptions import UnsupportedDtypeError


def _array(value, backend, dtype=None):
    return pacho.asarray(value, dtype=dtype, backend=backend)


def _host(value):
    return pacho.asnumpy(value)


def _skip_float64(backend):
    if backend.name == "opencl" and not backend.fp64:
        pytest.skip("selected OpenCL device does not support float64")


def _assert_same(actual, expected, *, rtol=1e-5, atol=1e-6):
    expected = np.asarray(expected)
    result = _host(actual)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype, (result.dtype, expected.dtype)
    if expected.dtype.kind in "fc":
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol, equal_nan=True)
    else:
        np.testing.assert_array_equal(result, expected)


def test_mixed_dtype_arithmetic_uses_numpy_promotion(backend):
    _skip_float64(backend)
    int_values = np.asarray([-3, 2, 7], dtype=np.int32)
    float64_values = np.asarray([0.5, -2.0, 1.25], dtype=np.float64)
    float32_values = float64_values.astype(np.float32)

    for left, right in (
        (int_values, float64_values),
        (float32_values, float64_values),
    ):
        left_array = _array(left, backend)
        right_array = _array(right, backend)
        for actual, expected in (
            (left_array + right_array, left + right),
            (left_array - right_array, left - right),
            (left_array * right_array, left * right),
        ):
            _assert_same(actual, expected)


def test_integer_true_division_promotes_to_float64(backend):
    _skip_float64(backend)
    left = np.asarray([1, -3, 7], dtype=np.int32)
    right = np.asarray([2, 2, -4], dtype=np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.true_divide(left, right)
    actual = _array(left, backend) / _array(right, backend)
    _assert_same(actual, expected)
    assert actual.dtype == np.dtype(np.float64)


def test_boolean_arithmetic_and_large_boolean_reduction(backend):
    values = np.ones(256, dtype=bool)
    array = _array(values, backend)
    _assert_same(array + array, values + values)
    _assert_same(pacho.sum(array, dtype=bool), np.sum(values, dtype=bool))
    assert pacho.sum(array, dtype=bool).item() is True


def test_astype_bool_matches_numpy_for_nan_and_signed_values(backend):
    values = np.asarray([0.0, 2.0, -1.0, np.nan], dtype=np.float32)
    _assert_same(_array(values, backend).astype(bool), values.astype(bool))


@pytest.mark.parametrize(
    "dtype, values",
    (
        (np.float16, [1.0]),
        (np.int8, [1]),
        (np.uint16, [1]),
        (np.complex64, [1 + 2j]),
        (object, [1]),
        (np.dtype([("value", np.float32)]), [(1.0,)]),
    ),
)
def test_unsupported_dtypes_fail_explicitly(backend, dtype, values):
    with pytest.raises(UnsupportedDtypeError):
        _array(np.asarray(values, dtype=dtype), backend)


def test_non_native_dtype_fails_explicitly(backend):
    non_native = np.dtype(">f4")
    if non_native.isnative:
        pytest.skip("test host uses big-endian byte order")
    values = np.asarray([1.0, 2.0], dtype=non_native)
    with pytest.raises(UnsupportedDtypeError):
        _array(values, backend)


def test_numpy_ndarray_left_operand_dispatches_to_array_backend(backend):
    host = np.asarray([[4, 5, 6], [7, 8, 9]], dtype=np.float32)
    array = _array(np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.float32), backend)
    for actual, expected in (
        (host + array, host + _host(array)),
        (host - array, host - _host(array)),
        (host * array, host * _host(array)),
        (host / array, host / _host(array)),
        (np.add(host, array), np.add(host, _host(array))),
    ):
        _assert_same(actual, expected)


def test_multiple_rank_broadcasting_and_incompatible_shapes(backend):
    left_host = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
    right_host = np.arange(12, dtype=np.float32).reshape(4, 3)
    left = _array(left_host, backend)
    right = _array(right_host, backend)
    _assert_same(left + right, left_host + right_host)

    scalar_host = np.asarray(2.0, dtype=np.float32)
    _assert_same(left * _array(scalar_host, backend), left_host * scalar_host)
    with pytest.raises(ValueError):
        left + _array(np.ones((2, 2), dtype=np.float32), backend)


def test_negative_empty_and_integer_slices_match_numpy(backend):
    source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    array = _array(source, backend)
    for key in (
        (..., slice(None, None, -1)),
        (slice(None, None, -1), slice(None), slice(None, None, -2)),
        (slice(1, 1), slice(None), slice(None)),
        (slice(None), slice(2, 2, -1), slice(None)),
        (-1, -1, -1),
    ):
        _assert_same(array[key], source[key])


def test_scalar_reduction_axes_follow_numpy(backend):
    source = np.asarray(7, dtype=np.int32)
    array = _array(source, backend)
    for axis in (None, 0, -1, ()):
        _assert_same(pacho.sum(array, axis=axis), np.sum(source, axis=axis))

    scalar_float = np.asarray(2.5, dtype=np.float32)
    scalar_array = _array(scalar_float, backend)
    for axis in (None, ()):
        _assert_same(pacho.mean(scalar_array, axis=axis), np.mean(scalar_float, axis=axis))
    for axis in (0, -1):
        with pytest.raises(ValueError):
            np.mean(scalar_float, axis=axis)
        with pytest.raises(ValueError):
            pacho.mean(scalar_array, axis=axis)

    with pytest.raises(ValueError):
        pacho.sum(array, axis=(0,))


def test_negative_and_duplicate_reduction_axes(backend):
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    array = _array(source, backend)
    for axis, keepdims in ((-1, False), ((-1, 0), False), ((0, -1), True), (-2, True)):
        _assert_same(
            pacho.sum(array, axis=axis, keepdims=keepdims),
            np.sum(source, axis=axis, keepdims=keepdims),
        )
    with pytest.raises(ValueError):
        pacho.sum(array, axis=(-1, 2))
    with pytest.raises(ValueError):
        pacho.sum(array, axis=(0, -3))


def test_numpy_norm_scalar_and_supported_3d_axes(backend):
    scalar = np.asarray(-3.5, dtype=np.float32)
    _assert_same(np.linalg.norm(_array(scalar, backend)), np.linalg.norm(scalar))

    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 5
    array = _array(tensor, backend)
    _assert_same(np.linalg.norm(array), np.linalg.norm(tensor), rtol=2e-5, atol=2e-6)
    _assert_same(
        np.linalg.norm(array, axis=(1, 2)),
        np.linalg.norm(tensor, axis=(1, 2)),
        rtol=2e-5,
        atol=2e-6,
    )

    with pytest.raises(ValueError):
        np.linalg.norm(array, axis=(0, 1, 2))


def test_use_backend_restores_nested_contexts_and_exceptions(opencl_backend):
    original = pacho.get_backend()
    numpy_backend = get_backend("numpy")
    with pacho.use_backend(numpy_backend) as selected:
        assert selected is numpy_backend
        assert pacho.get_backend() is numpy_backend
        with pacho.use_backend(opencl_backend) as nested:
            assert nested is opencl_backend
            assert pacho.get_backend() is opencl_backend
        assert pacho.get_backend() is numpy_backend
        with pytest.raises(RuntimeError), pacho.use_backend(opencl_backend):
            raise RuntimeError("restore context")
        assert pacho.get_backend() is numpy_backend
    assert pacho.get_backend() is original


def test_use_backend_restores_after_exception_without_opencl(numpy_backend):
    original = pacho.get_backend()
    with pytest.raises(RuntimeError), pacho.use_backend(numpy_backend):
        raise RuntimeError("restore context")
    assert pacho.get_backend() is original


def test_import_succeeds_when_optional_pyopencl_is_missing():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    # The subprocess must resolve the installed/editable distribution rather
    # than importing directly from the checkout's src directory.
    environment.pop("PYTHONPATH", None)
    environment.pop("PACHO_REQUIRE_OPENCL", None)
    script = """
import builtins

real_import = builtins.__import__
def block_pyopencl(name, *args, **kwargs):
    if name == "pyopencl" or name.startswith("pyopencl."):
        raise ImportError("simulated missing optional dependency")
    return real_import(name, *args, **kwargs)

builtins.__import__ = block_pyopencl
import openpacho
array = openpacho.asarray([1, 2], dtype=openpacho.float32)
assert openpacho.asnumpy(array).tolist() == [1.0, 2.0]
try:
    openpacho.devices()
except openpacho.BackendUnavailableError:
    pass
else:
    raise AssertionError("devices() should report a missing optional runtime")
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout.strip().endswith("ok")
