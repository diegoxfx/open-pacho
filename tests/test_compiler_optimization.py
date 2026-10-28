"""Regression tests for compiled fusion boundaries and mapped reductions."""

import numpy as np
import pytest

import openpacho as pacho


def assert_same(actual, expected):
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if expected.dtype.kind == "f":
        np.testing.assert_allclose(pacho.asnumpy(actual), expected, rtol=3e-5, atol=3e-5)
    else:
        np.testing.assert_array_equal(pacho.asnumpy(actual), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint32, np.bool_])
@pytest.mark.parametrize(
    "shape,axes",
    [
        ((4099,), None),
        ((5, 32), (-1,)),
        ((3, 2, 33), (1, 2)),
        ((9, 7, 39), (0, 2)),
        ((67, 3, 69), (0, 2)),
        ((2, 3), ()),
        ((), None),
        ((2, 0, 3), (1,)),
        ((0, 4097), (1,)),
    ],
)
def test_compiled_mapped_reduction(backend, dtype, shape, axes):
    size = int(np.prod(shape))
    host = (np.arange(size) % 7).astype(dtype).reshape(shape)

    @pacho.compile
    def total(a):
        mapped = a == 1 if dtype is np.bool_ else a * a + 2
        return np.sum(mapped, axis=axes, keepdims=True)

    expected_map = host == 1 if dtype is np.bool_ else host * host + 2
    actual = total(pacho.asarray(host, backend=backend))
    assert_same(actual, np.sum(expected_map, axis=axes, keepdims=True))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_map_mean_broadcast_and_rebinding(backend, dtype):
    host = np.arange(4 * 67 * 9, dtype=dtype).reshape(4, 67, 9) / 1000
    bias = np.arange(9, dtype=dtype)

    @pacho.compile
    def mean(a, b, rate):
        return np.mean((a + b) * rate, axis=(0, 2))

    a, b = [pacho.asarray(x, backend=backend) for x in (host, bias)]
    for rate in (0.5, 1.75):
        assert_same(mean(a, b, rate), np.mean((host + bias) * rate, axis=(0, 2)))
    assert mean.cache_info()["misses"] == 1


def test_map_float32_rounds_before_float64_accumulation(backend):
    host = np.linspace(0.7, 3.3, 9001, dtype=np.float32)
    compiled = pacho.compile(lambda a: np.sum(a * a, dtype=np.float64))
    result = compiled(pacho.asarray(host, backend=backend))
    assert result.dtype == np.dtype("float64")
    np.testing.assert_allclose(
        pacho.asnumpy(result), np.sum(host * host, dtype=np.float64), rtol=1e-12
    )


def test_map_boolean_accumulator(backend):
    host = np.zeros(9001, dtype=np.int32)
    host[-1] = 1
    compiled = pacho.compile(lambda a: np.sum(a > 0, dtype=np.bool_))
    assert_same(compiled(pacho.asarray(host, backend=backend)), np.sum(host > 0, dtype=np.bool_))


def test_shared_return_preserves_intermediate(backend):
    host = np.arange(4133, dtype=np.float32) / 1000

    @pacho.compile
    def shared(a):
        x = a + 2
        return x, np.sum(x * x), x - 1

    results = shared(pacho.asarray(host, backend=backend))
    for actual, expected in zip(results, (host + 2, np.sum((host + 2) ** 2), host + 1)):
        assert_same(actual, expected)


def test_repeated_operand_fuses_without_full_temporary(opencl_backend, monkeypatch):
    host = np.arange(9001, dtype=np.float32) / 1000
    array = pacho.asarray(host, backend=opencl_backend)

    @pacho.compile
    def total(a, rate):
        x = a + rate
        return np.sum(x * x)

    allocated_shapes = []
    original = opencl_backend.empty

    def record(shape, dtype):
        allocated_shapes.append(tuple(shape))
        return original(shape, dtype)

    monkeypatch.setattr(opencl_backend, "empty", record)
    actual = total(array, 2.0)
    assert_same(actual, np.sum((host + 2) ** 2))
    assert host.shape not in allocated_shapes
    assert [step["kind"] for step in total.explain(array, 2.0)["steps"]] == ["map_reduce"]


def test_repeated_elementwise_operand_single_group(opencl_backend):
    @pacho.compile
    def expression(a):
        x = a + 1
        return x * x

    host = np.arange(17, dtype=np.float32)
    array = pacho.asarray(host, backend=opencl_backend)
    assert_same(expression(array), (host + 1) ** 2)
    assert len(expression.explain(array)["steps"]) == 1


def test_returned_map_is_materialized_once(opencl_backend):
    @pacho.compile
    def expression(a):
        x = a * 2
        return x, np.sum(x)

    host = np.arange(9001, dtype=np.float32)
    array = pacho.asarray(host, backend=opencl_backend)
    actual, total = expression(array)
    assert_same(actual, host * 2)
    assert_same(total, np.sum(host * 2))
    assert [step["kind"] for step in expression.explain(array)["steps"]] == ["fused", "reduce"]
