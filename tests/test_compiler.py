"""Focused contract tests for ``openpacho.compile``.

The compiler deliberately covers a value-oriented subset of NumPy.  These tests
exercise that subset through the public API on the same NumPy/OpenCL fixtures as
the rest of the conformance suite.
"""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import openpacho as pacho


def _array(value, backend, dtype=None):
    return pacho.asarray(value, dtype=dtype, backend=backend)


def _assert_result(actual, expected, *, rtol=2e-5, atol=2e-6):
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    result = pacho.asnumpy(actual)
    if expected.dtype.kind in "fc":
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol, equal_nan=True)
    else:
        np.testing.assert_array_equal(result, expected)


def test_compile_public_forms(backend):
    def source(a):
        return 2 * a + 1

    direct = pacho.compile(source)

    @pacho.compile
    def decorated(a):
        return 2 * a + 1

    @pacho.compile()
    def factory_decorated(a):
        return 2 * a + 1

    value = np.arange(7, dtype=np.float32)
    array = _array(value, backend)
    for compiled, name in (
        (direct, "source"),
        (decorated, "decorated"),
        (factory_decorated, "factory_decorated"),
    ):
        assert compiled.__name__ == name
        _assert_result(compiled(array), 2 * value + 1)


def test_compile_native_numpy_passthrough():
    calls = []

    def source(a):
        calls.append("python")
        return np.sin(a) + 2

    compiled = pacho.compile(source)
    value = np.arange(5, dtype=np.float32)
    result = compiled(value)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.sin(value) + 2)
    assert calls == ["python"]


def test_compile_retraces_only_on_cache_miss(backend):
    trace_calls = []

    def source(a):
        trace_calls.append("capture")
        return a * 3 - 1

    compiled = pacho.compile(source)
    value = _array(np.arange(9, dtype=np.float32), backend)
    _assert_result(compiled(value), np.arange(9, dtype=np.float32) * 3 - 1)
    _assert_result(compiled(value), np.arange(9, dtype=np.float32) * 3 - 1)

    assert trace_calls == ["capture"]
    assert compiled.cache_info()["misses"] == 1
    assert compiled.cache_info()["hits"] == 1


def test_compile_rebinds_dynamic_scalars_with_numpy_weak_promotion(backend):
    def blend(a, b, rate):
        return (1 - rate) * a + rate * b

    compiled = pacho.compile(blend)
    left = np.linspace(-2, 2, 11, dtype=np.float32)
    right = np.linspace(3, -1, 11, dtype=np.float32)
    left_array, right_array = _array(left, backend), _array(right, backend)

    for rate in (0.25, 0.75):
        expected = (1 - rate) * left + rate * right
        actual = compiled(left_array, right_array, rate)
        _assert_result(actual, expected)
        assert actual.dtype == expected.dtype == np.dtype(np.float32)

    info = compiled.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 1


def test_compile_shared_ndarray_dispatch_and_tuple_outputs(backend):
    def mixed(x, matrix):
        row_sums = np.sum(x, axis=1)
        column_means = np.mean(x, axis=0, keepdims=True)
        product = np.matmul(matrix, matrix.T)
        flat = np.reshape(x, (6,))
        transposed = np.transpose(x)
        diagonal = np.diag(matrix, k=1)
        length = np.linalg.norm(x)
        return row_sums, column_means, product, flat, transposed, diagonal, length

    compiled = pacho.compile(mixed)
    x = np.arange(6, dtype=np.float32).reshape(2, 3) - 1
    matrix = np.arange(9, dtype=np.float32).reshape(3, 3)
    outputs = compiled(_array(x, backend), _array(matrix, backend))
    expected = (
        np.sum(x, axis=1),
        np.mean(x, axis=0, keepdims=True),
        np.matmul(matrix, matrix.T),
        np.reshape(x, (6,)),
        np.transpose(x),
        np.diag(matrix, k=1),
        np.linalg.norm(x),
    )

    assert isinstance(outputs, tuple)
    assert len(outputs) == len(expected)
    for actual, reference in zip(outputs, expected):
        _assert_result(actual, reference)


@pytest.mark.parametrize("operation", ("reshape", "reduce"))
def test_compile_zero_dim_asarray_flows_through_non_elementwise(backend, operation):
    def scalar_flow(_):
        scalar = pacho.asarray(2, dtype=np.float32)
        if operation == "reshape":
            return np.reshape(scalar, (1,))
        return np.sum(scalar)

    compiled = pacho.compile(scalar_flow)
    dummy = _array(np.zeros(1, dtype=np.float32), backend)
    expected_scalar = np.asarray(2, dtype=np.float32)
    expected = expected_scalar.reshape(1) if operation == "reshape" else np.sum(expected_scalar)
    _assert_result(compiled(dummy), expected)


def test_compile_zero_dim_zeros_like_can_be_reduced(backend):
    def zero_flow(_):
        scalar = pacho.asarray(2, dtype=np.float32)
        return np.sum(pacho.zeros_like(scalar))

    compiled = pacho.compile(zero_flow)
    dummy = _array(np.zeros(1, dtype=np.float32), backend)
    _assert_result(compiled(dummy), np.asarray(0, dtype=np.float32))


def test_compile_tuple_outputs_reuse_an_input(backend):
    def pair(a, b):
        shared = a + b
        return shared, shared * shared, np.sum(shared)

    compiled = pacho.compile(pair)
    left = np.arange(12, dtype=np.float32).reshape(3, 4)
    right = np.full((3, 4), 2, dtype=np.float32)
    outputs = compiled(_array(left, backend), _array(right, backend))
    shared = left + right

    assert isinstance(outputs, tuple)
    for actual, reference in zip(outputs, (shared, shared * shared, np.sum(shared))):
        _assert_result(actual, reference)


def test_compile_cache_tracks_shape_and_dtype(backend):
    compiled = pacho.compile(lambda a: a + 1)
    values = np.arange(6, dtype=np.float32)

    _assert_result(compiled(_array(values, backend)), values + 1)
    _assert_result(compiled(_array(values.copy(), backend)), values + 1)
    reshaped = values.reshape(2, 3)
    _assert_result(compiled(_array(reshaped, backend)), reshaped + 1)
    integers = values.astype(np.int32)
    _assert_result(compiled(_array(integers, backend)), integers + 1)

    info = compiled.cache_info()
    assert info["misses"] == 3
    assert info["hits"] == 1
    assert info["size"] == 3


def test_compile_cache_size_is_bounded(backend):
    compiled = pacho.compile(lambda a: a + 1, cache_size=2)
    for length in (3, 4, 5):
        value = np.arange(length, dtype=np.float32)
        _assert_result(compiled(_array(value, backend)), value + 1)

    info = compiled.cache_info()
    assert info["max_size"] == 2
    assert info["size"] == 2
    assert info["misses"] == 3

    first = np.arange(3, dtype=np.float32)
    _assert_result(compiled(_array(first, backend)), first + 1)
    assert compiled.cache_info()["misses"] == 4
    assert compiled.cache_info()["size"] == 2


def test_compile_static_shape_argument_creates_specialized_plans(backend):
    def reshape_to(a, shape):
        return np.reshape(a, shape)

    compiled = pacho.compile(reshape_to, static_argnames=("shape",))
    value = np.arange(6, dtype=np.float32)
    array = _array(value, backend)
    _assert_result(compiled(array, (2, 3)), value.reshape(2, 3))
    _assert_result(compiled(array, (3, 2)), value.reshape(3, 2))

    info = compiled.cache_info()
    assert info["misses"] == 2
    assert info["hits"] == 0
    assert info["size"] == 2


def test_compile_static_axis_argument_creates_specialized_reductions(backend):
    def reduce_axis(a, axis):
        return np.sum(a, axis=axis)

    compiled = pacho.compile(reduce_axis, static_argnames="axis")
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    array = _array(value, backend)
    _assert_result(compiled(array, 0), np.sum(value, axis=0))
    _assert_result(compiled(array, 1), np.sum(value, axis=1))
    _assert_result(compiled(array, 0), np.sum(value, axis=0))

    info = compiled.cache_info()
    assert info["misses"] == 2
    assert info["hits"] == 1


def test_compile_requires_dynamic_shape_metadata_to_be_static(backend):
    compiled = pacho.compile(lambda a, shape: np.reshape(a, shape))
    array = _array(np.arange(6, dtype=np.float32), backend)
    with pytest.raises(TypeError, match="static_argnames"):
        compiled(array, (2, 3))


def test_compile_cache_separates_backend_plans(opencl_backend):
    numpy_array = _array(np.arange(5, dtype=np.float32), "numpy")
    opencl_array = _array(np.arange(5, dtype=np.float32), opencl_backend)
    compiled = pacho.compile(lambda a: a * 2)

    _assert_result(compiled(numpy_array), np.arange(5, dtype=np.float32) * 2)
    opencl_result = compiled(opencl_array)
    assert opencl_result.backend == "opencl"
    _assert_result(opencl_result, np.arange(5, dtype=np.float32) * 2)
    assert compiled.cache_info()["misses"] == 2
    assert compiled.cache_info()["size"] == 2


def test_compile_rejects_mixed_backend_inputs(opencl_backend):
    left = _array([1, 2, 3], "numpy", dtype=np.float32)
    right = _array([4, 5, 6], opencl_backend, dtype=np.float32)
    compiled = pacho.compile(lambda a, b: a + b)

    with pytest.raises(pacho.BackendMismatchError):
        compiled(left, right)


def test_compile_clear_cache_resets_plans_and_statistics(backend):
    compiled = pacho.compile(lambda a: a * 4)
    value = np.arange(4, dtype=np.float32)
    array = _array(value, backend)
    _assert_result(compiled(array), value * 4)
    _assert_result(compiled(array), value * 4)
    assert compiled.cache_info()["size"] == 1

    compiled.clear_cache()
    assert compiled.cache_info() == {"hits": 0, "misses": 0, "size": 0, "max_size": 32}
    _assert_result(compiled(array), value * 4)
    assert compiled.cache_info()["misses"] == 1


def test_compile_explain_reports_plan_without_host_observations(backend):
    trace_calls = []

    def source(a):
        trace_calls.append("capture")
        return np.sum(a, axis=0) + 1

    compiled = pacho.compile(source)
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    array = _array(value, backend)
    details = compiled.explain(array)

    assert details["backend"] == backend.name
    assert details["device"] == backend.device
    assert details["outputs"] == 1
    assert details["steps"]
    assert details["host_observations"] == 0
    assert details["python_retrace_on_cache_hit"] is False
    assert trace_calls == ["capture"]

    compiled(array)
    assert trace_calls == ["capture"]


def test_compile_does_not_transfer_during_capture_or_execution(opencl_backend, monkeypatch):
    array = _array(np.arange(17, dtype=np.float32), opencl_backend)
    compiled = pacho.compile(lambda a: np.sqrt(a * a + 1))

    def forbidden(*args, **kwargs):
        raise AssertionError("compile attempted an implicit host transfer")

    with monkeypatch.context() as patch:
        patch.setattr(opencl_backend, "upload", forbidden)
        patch.setattr(opencl_backend, "download", forbidden)
        result = compiled(array)
        assert result.backend == "opencl"

    np.testing.assert_allclose(
        pacho.asnumpy(result), np.sqrt(np.arange(17, dtype=np.float32) ** 2 + 1), rtol=2e-6
    )


def test_compile_plan_does_not_retain_input_arrays(backend):
    compiled = pacho.compile(lambda a: a * 2 + 1)
    source = _array(np.arange(13, dtype=np.float32), backend)
    source_ref = weakref.ref(source)
    result = compiled(source)
    del source
    gc.collect()

    assert source_ref() is None
    _assert_result(result, np.arange(13, dtype=np.float32) * 2 + 1)


def test_compile_concurrent_calls_share_one_plan(backend):
    trace_calls = []

    def source(a, scale):
        trace_calls.append("capture")
        return a * scale + 1

    compiled = pacho.compile(source)
    values = [np.arange(23, dtype=np.float32) + i for i in range(8)]
    arrays = [_array(value, backend) for value in values]

    def run(pair):
        value, array = pair
        return compiled(array, 0.5), value * 0.5 + 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, zip(values, arrays)))

    for actual, expected in results:
        _assert_result(actual, expected)
    assert trace_calls == ["capture"]
    assert compiled.cache_info()["misses"] == 1
    assert compiled.cache_info()["hits"] == len(values) - 1


def test_compile_rejects_input_mutation(backend):
    def mutating(a):
        a += 1
        return a

    array = _array(np.arange(4, dtype=np.float32), backend)
    with pytest.raises((TypeError, NotImplementedError)):
        pacho.compile(mutating)(array)


def test_compile_rejects_nested_compiled_calls(backend):
    inner = pacho.compile(lambda a: a * 2)
    outer = pacho.compile(lambda a: inner(a) + 1)
    array = _array(np.arange(4, dtype=np.float32), backend)

    with pytest.raises(TypeError, match="Nested compiled"):
        outer(array)
