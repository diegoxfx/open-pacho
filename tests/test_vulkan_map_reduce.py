"""Native Vulkan mapped reductions and a CPU-only code-generation harness."""

import os
from collections import OrderedDict
from threading import RLock

import numpy as np
import pytest

import openpacho as pacho
from openpacho._array import ndarray
from openpacho._compiler import ELEMENTWISE, capture_call
from openpacho._fusion import ScalarExpression
from openpacho._ops import HostScalar
from openpacho._vulkan import VulkanBackend
from openpacho.exceptions import BackendUnavailableError


@pytest.fixture(scope="module")
def vk_map_backend():
    try:
        backend = VulkanBackend(os.environ.get("PACHO_VULKAN_DEVICE"))
    except BackendUnavailableError as exc:
        if os.environ.get("PACHO_REQUIRE_VULKAN", "").lower() in ("1", "true", "yes"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    yield backend
    backend.synchronize()


def assert_same(actual, expected):
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if expected.dtype.kind == "f":
        np.testing.assert_allclose(pacho.asnumpy(actual), expected, rtol=3e-5, atol=3e-5)
    else:
        np.testing.assert_array_equal(pacho.asnumpy(actual), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.bool_])
@pytest.mark.parametrize(
    "shape,axis,keepdims",
    [
        ((4099,), None, False),
        ((5, 32), -1, True),
        ((9, 7, 39), (0, 2), False),
        ((67, 3, 69), (2, 0), True),
        ((2, 3), (), False),
        ((), None, True),
        ((2, 0, 3), (1,), True),
        ((0, 4097), (1,), False),
    ],
)
def test_axes_shapes_and_dtype(vk_map_backend, dtype, shape, axis, keepdims):
    if dtype == np.float64 and not vk_map_backend.fp64:
        pytest.skip("Optional float64 unavailable")
    if dtype in (np.int32, np.bool_) and not vk_map_backend.int64:
        pytest.skip("NumPy integer sum requires optional shaderInt64")
    host = (np.arange(int(np.prod(shape))) % 7).astype(dtype).reshape(shape)

    @pacho.compile
    def total(a):
        mapped = a == 1 if dtype is np.bool_ else a * a + 2
        return np.sum(mapped, axis=axis, keepdims=keepdims)

    a = pacho.asarray(host, backend=vk_map_backend)
    expected_map = host == 1 if dtype is np.bool_ else host * host + 2
    assert_same(total(a), np.sum(expected_map, axis=axis, keepdims=keepdims))
    assert [step["kind"] for step in total.explain(a)["steps"]] == ["map_reduce"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mean_broadcast_dynamic_scalars(vk_map_backend, dtype):
    if dtype == np.float64 and not vk_map_backend.fp64:
        pytest.skip("Optional float64 unavailable")
    host = np.arange(4 * 67 * 9, dtype=dtype).reshape(4, 67, 9) / 1000
    bias = np.arange(9, dtype=dtype)
    operation = pacho.compile(lambda a, b, rate: np.mean((a + b) * rate, axis=(0, 2)))
    a, b = [pacho.asarray(x, backend=vk_map_backend) for x in (host, bias)]
    for rate in (0.5, 1.75):
        assert_same(operation(a, b, rate), np.mean((host + bias) * rate, axis=(0, 2)))
    assert operation.cache_info()["misses"] == 1


def test_map_rounds_float32_before_float64_sum(vk_map_backend):
    if not vk_map_backend.fp64:
        pytest.skip("Optional float64 unavailable")
    host = np.linspace(0.7, 3.3, 9001, dtype=np.float32)
    operation = pacho.compile(lambda a: np.sum(a * a, dtype=np.float64))
    actual = operation(pacho.asarray(host, backend=vk_map_backend))
    assert actual.dtype == np.dtype("float64")
    np.testing.assert_allclose(
        pacho.asnumpy(actual), np.sum(host * host, dtype=np.float64), rtol=1e-12
    )


@pytest.mark.parametrize("active", [False, True])
def test_boolean_accumulator_and_chunk_boundary(vk_map_backend, active):
    host = np.zeros(9001, dtype=np.int32)
    if active:
        host[-1] = 1
    operation = pacho.compile(lambda a: np.sum(a > 0, dtype=np.bool_))
    assert_same(
        operation(pacho.asarray(host, backend=vk_map_backend)), np.sum(host > 0, dtype=np.bool_)
    )


def test_compact_partials_without_host_read(vk_map_backend, monkeypatch):
    host = np.arange(3 * 9001, dtype=np.float32).reshape(3, 9001) / 10000
    a = pacho.asarray(host, backend=vk_map_backend)
    operation = pacho.compile(lambda a, rate: np.sum((a + rate) * (a + rate), axis=1))
    allocations = []
    original = vk_map_backend.empty

    def record(shape, dtype):
        allocations.append(tuple(shape))
        return original(shape, dtype)

    def forbid(*args):
        raise AssertionError("Mapped reduction attempted a host observation")

    with monkeypatch.context() as patch:
        patch.setattr(vk_map_backend, "empty", record)
        patch.setattr(vk_map_backend, "download", forbid)
        actual = operation(a, 2.0)
    assert all(np.prod(shape) < host.size for shape in allocations)
    assert (3, 3) in allocations
    assert_same(actual, np.sum((host + 2) * (host + 2), axis=1))


def test_returned_producer_is_preserved(vk_map_backend):
    @pacho.compile
    def operation(a):
        x = a * 2
        return x, np.sum(x)

    host = np.arange(4099, dtype=np.float32)
    a = pacho.asarray(host, backend=vk_map_backend)
    mapped, total = operation(a)
    assert_same(mapped, host * 2)
    assert_same(total, np.sum(host * 2))
    assert [step["kind"] for step in operation.explain(a)["steps"]] == ["fused", "reduce"]


def test_unsupported_double_transcendental_is_explicit(vk_map_backend):
    if not vk_map_backend.fp64:
        pytest.skip("Optional float64 unavailable")
    operation = pacho.compile(lambda a: np.sum(np.exp(a)))
    a = pacho.asarray(np.ones(7, np.float64), backend=vk_map_backend)
    with pytest.raises(NotImplementedError, match="float64 exp"):
        operation(a)


def _capture_reduction(backend, shape):
    a = ndarray._wrap(None, shape, np.dtype("float32"), backend)
    plan, bindings = capture_call(
        lambda a, rate: np.sum((a + rate) * (a + rate), axis=-1), (a, 2.0), {}
    )
    node = plan.outputs[0]
    group, leaves, seen = [], [], set()

    def visit(current):
        if id(current) in seen:
            return
        seen.add(id(current))
        if current.op not in ELEMENTWISE:
            leaves.append(current)
        else:
            for child in current.inputs:
                visit(child)
            group.append(current)

    visit(node.inputs[0])
    operands = []
    for leaf in leaves:
        if leaf.op == "input":
            operands.append(bindings[leaf.value.index])
        else:
            value = leaf.value
            operands.append(
                HostScalar(value.evaluate(bindings), leaf.dtype)
                if isinstance(value, ScalarExpression)
                else value
            )
    return node, group, leaves, operands


def test_source_harness_compact_storage_and_bounded_value_only_cache():
    backend = VulkanBackend.__new__(VulkanBackend)
    backend._device = backend._instance = None
    backend._lock = RLock()
    backend._fusion_bodies = OrderedDict()
    backend.fp64 = backend.int64 = True
    backend.device = "CPU-only source harness"
    allocations, dispatches = [], []
    backend.empty = lambda shape, dtype: allocations.append(tuple(shape)) or object()
    backend._run = lambda body, count, buffers, scalars, **kwargs: dispatches.append(
        (body, count, kwargs)
    )
    backend.reduce = lambda a, axes, keepdims, dtype: object()
    node, group, leaves, operands = _capture_reduction(backend, (3, 9001))
    backend.fused_reduce(node, group, leaves, operands)
    assert allocations == [(3, 3)]
    assert dispatches[0][1] == 9
    assert dispatches[0][2]["cooperative"]
    assert "precise float" in dispatches[0][0]
    assert "pc.s0" in dispatches[0][0]

    for width in range(1, 140):
        backend.fused_reduce(*_capture_reduction(backend, (2, width)))
    assert len(backend._fusion_bodies) <= 128

    def plain_metadata(value):
        if isinstance(value, tuple):
            return all(plain_metadata(part) for part in value)
        return isinstance(value, (str, int, bool))

    assert all(
        plain_metadata(key) and plain_metadata(value)
        for key, value in backend._fusion_bodies.items()
    )
    # The largest permitted mapped domain must not form (chunk+1)*4096
    # or increment an absolute r past INT_MAX in its final cooperative chunk.
    limit = 2**31 - 1
    backend.fused_reduce(*_capture_reduction(backend, (limit,)))
    source = dispatches[-1][0]
    assert f"for(int j=lane;j<min(4096,{limit}-chunk*4096);j+=64)" in source
    assert "int r=chunk*4096+j;" in source
    assert "(chunk+1)*4096" not in source
    assert "r+=64" not in source
    start = ((limit - 1) // 4096) * 4096
    final_indices = [
        int(np.int32(start) + np.int32(j))
        for lane in range(64)
        for j in range(lane, min(4096, limit - start), 64)
    ]
    assert sorted(final_indices) == list(range(start, limit))
    with pytest.raises(ValueError, match="indexing limits"):
        backend.fused_reduce(*_capture_reduction(backend, (65536, 65536)))
