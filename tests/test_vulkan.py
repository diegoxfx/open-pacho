"""Optional real-driver Vulkan conformance, with no CPU execution fallback."""

import os

import numpy as np
import pytest

import openpacho as pacho
from examples import solvers
from openpacho._array import ndarray
from openpacho._fusion import HostScalar
from openpacho._vulkan import VulkanBackend, vulkan_devices
from openpacho.exceptions import BackendUnavailableError, UnsupportedDtypeError


@pytest.fixture(scope="module")
def vk_backend(request):
    try:
        backend = VulkanBackend(
            os.environ.get("PACHO_VULKAN_DEVICE"),
            _descriptor_mode=getattr(request, "param", "auto"),
        )
    except BackendUnavailableError as exc:
        if os.environ.get("PACHO_REQUIRE_VULKAN", "").lower() in ("1", "true", "yes"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    yield backend
    backend.synchronize()


def array(backend, value):
    value = np.asarray(value)
    return ndarray._wrap(backend.upload(value), value.shape, value.dtype, backend)


def result(backend, data, shape, dtype):
    return backend.download(ndarray._wrap(data, shape, dtype, backend))


@pytest.mark.parametrize(
    "dtype", ["bool", "int32", "uint32", "int64", "uint64", "float32", "float64"]
)
def test_transfer_and_broadcast(vk_backend, dtype):
    backend = vk_backend
    try:
        dtype = backend.validate_dtype(dtype)
    except UnsupportedDtypeError:
        pytest.skip("Optional 64-bit device capability unavailable")
    host = np.arange(12).reshape(3, 4).astype(dtype)
    a = array(backend, host)
    np.testing.assert_array_equal(backend.download(a), host)
    b = array(backend, np.arange(4).astype(dtype))
    out = backend.elementwise("add", [a, b], host.shape, dtype)
    np.testing.assert_array_equal(
        result(backend, out, host.shape, dtype), host + np.arange(4).astype(dtype)
    )


@pytest.mark.parametrize(
    "name", ["add", "subtract", "multiply", "true_divide", "power", "equal", "less"]
)
def test_binary_float(vk_backend, name):
    x = np.linspace(0.2, 2, 17, dtype=np.float32)
    y = np.linspace(0.4, 1.5, 17, dtype=np.float32)
    expected = getattr(np, name)(x, y)
    out = vk_backend.elementwise(
        name, [array(vk_backend, x), array(vk_backend, y)], x.shape, expected.dtype
    )
    np.testing.assert_allclose(
        result(vk_backend, out, x.shape, expected.dtype), expected, rtol=2e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    "name",
    ["negative", "positive", "absolute", "sqrt", "exp", "log", "sin", "cos", "tanh", "isfinite"],
)
def test_unary_float(vk_backend, name):
    x = np.linspace(0.2, 2, 17, dtype=np.float32)
    expected = getattr(np, name)(x)
    out = vk_backend.elementwise(name, [array(vk_backend, x)], x.shape, expected.dtype)
    np.testing.assert_allclose(
        result(vk_backend, out, x.shape, expected.dtype), expected, rtol=2e-6, atol=1e-6
    )


def test_scalars_are_dynamic_push_constants(vk_backend):
    host = np.arange(11, dtype=np.float32)
    a = array(vk_backend, host)
    pipeline_count = None
    for scalar in (2, 7, -3):
        out = vk_backend.elementwise(
            "multiply", [a, HostScalar(scalar, np.float32)], a.shape, a.dtype
        )
        np.testing.assert_array_equal(result(vk_backend, out, a.shape, a.dtype), host * scalar)
        if pipeline_count is None:
            pipeline_count = len(vk_backend._pipelines)
        assert len(vk_backend._pipelines) == pipeline_count


@pytest.mark.parametrize("axes", [(0,), (1,), (0, 2), (0, 1, 2), ()])
def test_reduction_axes(vk_backend, axes):
    host = np.arange(3000, dtype=np.float32).reshape(3, 10, 100) / 3000
    a = array(vk_backend, host)
    expected = np.sum(host, axis=axes)
    out = vk_backend.reduce(a, axes, False, a.dtype)
    np.testing.assert_allclose(
        result(vk_backend, out, expected.shape, a.dtype), expected, rtol=3e-6, atol=2e-5
    )


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), ()])
def test_empty_and_scalar(vk_backend, shape):
    a = array(vk_backend, np.zeros(shape, dtype=np.float32))
    out = vk_backend.full(shape, 2, a.dtype)
    np.testing.assert_array_equal(
        result(vk_backend, out, shape, a.dtype), np.full(shape, 2, dtype=np.float32)
    )
    reduced = vk_backend.reduce(a, tuple(range(a.ndim)), False, a.dtype)
    assert result(vk_backend, reduced, (), a.dtype) == 0


@pytest.mark.parametrize(
    "left,right", [((3, 7), (7, 5)), ((7,), (7,)), ((3, 7), (7,)), ((7,), (7, 5))]
)
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "bool"])
def test_matrix_products(vk_backend, left, right, dtype):
    if dtype == "float64" and not vk_backend.fp64:
        pytest.skip("Optional float64 unavailable")
    x = (np.arange(np.prod(left)).reshape(left) % 3).astype(dtype)
    y = (np.arange(np.prod(right)).reshape(right) % 4).astype(dtype)
    expected = x @ y
    out = vk_backend.matmul(
        array(vk_backend, x), array(vk_backend, y), expected.shape, np.dtype(dtype)
    )
    np.testing.assert_allclose(result(vk_backend, out, expected.shape, np.dtype(dtype)), expected)


def test_gather_negative_strides(vk_backend):
    host = np.arange(12, dtype=np.float32).reshape(3, 4)
    a = array(vk_backend, host)
    out = vk_backend.gather(a, (4, 3), 8, (1, -4))
    np.testing.assert_array_equal(result(vk_backend, out, (4, 3), a.dtype), host[::-1].T)


def test_pending_lifetime_and_queue_dependencies(vk_backend):
    host = np.arange(19, dtype=np.float32)
    a = array(vk_backend, host)
    for _ in range(80):
        a = ndarray._wrap(
            vk_backend.elementwise("add", [a, HostScalar(1, a.dtype)], a.shape, a.dtype),
            a.shape,
            a.dtype,
            vk_backend,
        )
    np.testing.assert_array_equal(vk_backend.download(a), host + 80)
    assert not vk_backend._pending
    vk_backend.release_cached_memory()
    np.testing.assert_array_equal(vk_backend.download(a), host + 80)


def test_unsupported_dtype_and_double_transcendental(vk_backend):
    with pytest.raises(UnsupportedDtypeError):
        vk_backend.validate_dtype("complex64")
    if vk_backend.fp64:
        a = array(vk_backend, np.ones(3, np.float64))
        with pytest.raises(NotImplementedError, match="float64 exp"):
            vk_backend.elementwise("exp", [a], a.shape, a.dtype)


def test_discovery(vk_backend):
    assert any(device.name == vk_backend.device for device in vulkan_devices())


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nonfinite_values(vk_backend, dtype):
    if dtype == "float64" and not vk_backend.fp64:
        pytest.skip("Optional float64 unavailable")
    for value in (np.nan, np.inf, -np.inf, -0.0):
        out = vk_backend.full((3,), value, np.dtype(dtype))
        actual = result(vk_backend, out, (3,), np.dtype(dtype))
        np.testing.assert_array_equal(actual, np.full(3, value, dtype=dtype))
        if value == 0:
            assert np.signbit(actual).all()


def test_compile_failure_is_explicit_and_backend_survives(vk_backend):
    with pytest.raises(RuntimeError, match="shader compilation failed"):
        vk_backend._pipeline("this is not a shader", 0, 0)
    host = np.arange(7, dtype=np.float32)
    np.testing.assert_array_equal(vk_backend.download(array(vk_backend, host)), host)


def test_public_backend_selection(vk_backend):
    assert pacho.get_backend(vk_backend) is vk_backend
    found = pacho.devices("vulkan")
    selected_id = next(item.id for item in found if item.name == vk_backend.device)
    selected = pacho.get_backend("vulkan", device=selected_id)
    assert selected.name == "vulkan" and selected.device == vk_backend.device
    with pacho.use_backend(vk_backend):
        actual = pacho.asarray(np.arange(5, dtype=np.float32))
        assert actual._backend is vk_backend
        np.testing.assert_array_equal(pacho.asnumpy(actual), np.arange(5, dtype=np.float32))


def test_public_compile_rebinds_scalar_and_new_arrays(vk_backend):
    @pacho.compile
    def blend(a, b, rate):
        return (1 - rate) * a + rate * b

    for offset, rate in ((0, 0.25), (3, 0.75)):
        left = np.arange(11, dtype=np.float32) + offset
        right = np.linspace(-2, 2, 11, dtype=np.float32)
        actual = blend(
            pacho.asarray(left, backend=vk_backend),
            pacho.asarray(right, backend=vk_backend),
            rate,
        )
        assert actual._backend is vk_backend
        np.testing.assert_allclose(pacho.asnumpy(actual), (1 - rate) * left + rate * right)
    assert blend.cache_info()["misses"] == 1
    assert blend.cache_info()["hits"] == 1


def test_public_compile_matrix_reductions(vk_backend):
    @pacho.compile()
    def statistics(a, b):
        product = a @ b
        return product, np.sum(product * product, axis=1), np.mean(product, axis=0)

    host_a = np.arange(21, dtype=np.float32).reshape(3, 7) / 7
    host_b = np.arange(28, dtype=np.float32).reshape(7, 4) / 9
    a, b = pacho.asarray(host_a, backend=vk_backend), pacho.asarray(host_b, backend=vk_backend)
    product = host_a @ host_b
    actual = statistics(a, b)
    expected = product, np.sum(product * product, axis=1), np.mean(product, axis=0)
    for value, reference in zip(actual, expected):
        assert value._backend is vk_backend
        np.testing.assert_allclose(pacho.asnumpy(value), reference, rtol=3e-6, atol=1e-5)
    explanation = statistics.explain(a, b)
    assert explanation["backend"] == "vulkan"
    assert explanation["host_observations"] == 0
    assert {step["kind"] for step in explanation["steps"]} >= {"matmul", "reduce"}


@pytest.mark.parametrize("method", ["conjugate_gradient", "jacobi", "steepest_descent"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_example_solvers_converge(vk_backend, method, dtype):
    if dtype == np.float64 and not vk_backend.fp64:
        pytest.skip("Optional float64 unavailable")
    rng = np.random.default_rng(47)
    raw = rng.uniform(-0.1, 0.1, (9, 9))
    matrix = (raw + raw.T + 4 * np.eye(9)).astype(dtype)
    exact = np.linspace(-1, 2, 9).astype(dtype)
    rhs = matrix @ exact
    a, b = pacho.asarray(matrix, backend=vk_backend), pacho.asarray(rhs, backend=vk_backend)
    tolerance = 2e-5 if dtype == np.float32 else 1e-10
    solution = getattr(solvers, method)(a, b, rtol=tolerance, maxiter=100)
    assert solution.converged and solution.reason == "converged"
    assert solution.x._backend is vk_backend
    actual = pacho.asnumpy(solution.x)
    np.testing.assert_allclose(
        actual, np.linalg.solve(matrix, rhs), rtol=tolerance * 5, atol=tolerance * 5
    )
    assert np.linalg.norm(rhs - matrix @ actual) <= tolerance * np.linalg.norm(rhs) * 1.1
    np.testing.assert_array_equal(pacho.asnumpy(a), matrix)
    np.testing.assert_array_equal(pacho.asnumpy(b), rhs)


@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 3, 17), (2, 17, 9)),
        ((2, 1, 9, 13), (3, 13, 11)),
        ((13,), (2, 13, 9)),
        ((2, 9, 13), (13,)),
        ((2, 1, 9, 13), (3, 13, 1)),
        ((13,), (13,)),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.bool_])
def test_batched_matmul_broadcast_and_tile_edges(vk_backend, shapes, dtype):
    if dtype == np.float64 and not vk_backend.fp64:
        pytest.skip("Optional float64 unavailable")
    rng = np.random.default_rng(91)
    left = rng.integers(-2, 3, shapes[0]).astype(dtype)
    right = rng.integers(-2, 3, shapes[1]).astype(dtype)
    expected = left @ right
    a, b = array(vk_backend, left), array(vk_backend, right)
    data = vk_backend.matmul(a, b, expected.shape, expected.dtype)
    np.testing.assert_allclose(result(vk_backend, data, expected.shape, expected.dtype), expected)


def test_pool_reuses_only_completed_unreferenced_buffers(vk_backend):
    import gc
    import weakref

    backend = vk_backend
    backend.release_cached_memory()
    a = array(backend, np.arange(257, dtype=np.float32))
    # The upload owns a pending reference; dropping the array cannot recycle it.
    storage = weakref.ref(a._data)
    del a
    gc.collect()
    assert storage() is not None
    assert backend._cached_bytes == 0
    backend.synchronize()
    gc.collect()
    assert storage() is None
    assert backend._cached_bytes > 0
    before = backend._cached_bytes
    data = backend.empty((257,), np.float32)
    assert backend._cached_bytes == before - data.allocation_size
    a = ndarray._wrap(data, (257,), np.float32, backend)
    # Releasing cached allocations cannot invalidate independently live arrays.
    backend.release_cached_memory()
    assert backend._cached_bytes == 0
    assert not backend._free_commands
    assert not backend._free_descriptor_pools
    allocation_size = data.allocation_size
    del a, data
    gc.collect()
    assert backend._cached_bytes == allocation_size
    backend.release_cached_memory()


def test_pool_bounds_memory_and_resource_counts(vk_backend):
    backend = vk_backend
    backend.release_cached_memory()
    previous = backend._pool_limit
    backend._pool_limit = 128 * 1024
    try:
        data = [backend.empty((8192,), np.float32) for _ in range(8)]
        data.clear()
        assert 0 < backend._cached_bytes <= 128 * 1024
        assert backend._pool_items <= 128
    finally:
        backend._pool_limit = previous
        backend.release_cached_memory()


def test_fused_bool_cast_shared_input_and_dynamic_values(vk_backend):
    @pacho.compile
    def evaluate(a, scalar):
        mask = a > scalar
        return (mask + mask).astype(np.int32), (a * a + scalar) / 2

    host = np.arange(67, dtype=np.float32)
    a = pacho.asarray(host, backend=vk_backend)
    for scalar in (2.0, 19.0):
        actual = evaluate(a, scalar)
        expected = ((host > scalar) + (host > scalar)).astype(np.int32), (host * host + scalar) / 2
        for value, reference in zip(actual, expected):
            np.testing.assert_array_equal(pacho.asnumpy(value), reference)
    kinds = [step["kind"] for step in evaluate.explain(a, 2.0)["steps"]]
    assert kinds.count("fused") == 2


def test_fused_float64_transcendental_stays_explicit(vk_backend):
    if not vk_backend.fp64:
        pytest.skip("Optional float64 unavailable")
    operation = pacho.compile(lambda a: np.exp(a) + 1)
    a = pacho.asarray(np.ones(7, np.float64), backend=vk_backend)
    with pytest.raises(NotImplementedError, match="float64 exp"):
        operation(a)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.bool_])
@pytest.mark.parametrize(
    "shapes", [((4096,), (4096,)), ((65539,), (65539,)), ((2, 1, 4099), (4099, 1))]
)
def test_long_dot_uses_device_partials(vk_backend, dtype, shapes):
    if dtype == np.float64 and not vk_backend.fp64:
        pytest.skip("Optional float64 unavailable")
    rng = np.random.default_rng(82)
    left = rng.integers(-2, 3, shapes[0]).astype(dtype)
    right = rng.integers(-2, 3, shapes[1]).astype(dtype)
    a, b = array(vk_backend, left), array(vk_backend, right)
    expected = left @ right
    data = vk_backend.matmul(a, b, expected.shape, expected.dtype)
    np.testing.assert_array_equal(
        result(vk_backend, data, expected.shape, expected.dtype), expected
    )


def test_dispatch_uses_cached_immutable_capabilities(vk_backend):
    backend = vk_backend
    backend.release_cached_memory()
    properties, memory = backend._properties, backend._memory
    # Only initialization should traverse the binding's dynamic native structs.
    backend._properties = backend._memory = None
    try:
        host = np.arange(73, dtype=np.float32)
        a = array(backend, host)
        data = backend.elementwise("add", [a, HostScalar(2, a.dtype)], a.shape, a.dtype)
        np.testing.assert_array_equal(result(backend, data, a.shape, a.dtype), host + 2)
    finally:
        backend._properties, backend._memory = properties, memory


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.bool_])
@pytest.mark.parametrize("scalar", [False, True])
def test_staging_owns_host_data_and_rebinds_readback(vk_backend, dtype, scalar):
    if dtype == np.float64 and not vk_backend.fp64 or dtype == np.int64 and not vk_backend.int64:
        pytest.skip("Optional 64-bit capability unavailable")
    host = (
        np.asarray(3, dtype=dtype)
        if scalar
        else np.arange(70).astype(dtype).reshape(7, 10)[:, ::-2]
    )
    expected = host.copy()
    a = array(vk_backend, host)
    host[...] = 0  # Mutating the original before GPU completion must be safe.
    np.testing.assert_array_equal(vk_backend.download(a), expected)
    b = array(vk_backend, np.full_like(expected, 1))
    np.testing.assert_array_equal(vk_backend.download(b), np.full_like(expected, 1))
    np.testing.assert_array_equal(vk_backend.download(a), expected)


def test_memory_type_preference_preserves_requirements():
    from openpacho._vulkan import _memory_type_index

    # Arbitrary bits: visible=1, coherent=2, cached=4. A cached noncoherent
    # type cannot be chosen without implementing host cache invalidation.
    assert _memory_type_index((3, 5, 7), 0b111, 3, 4) == 2
    assert _memory_type_index((3, 5, 7), 0b011, 3, 4) == 0
    assert _memory_type_index((3, 7), 0b11, 3) == 0
    with pytest.raises(BackendUnavailableError, match="memory type"):
        _memory_type_index((1, 5), 0b11, 3, 4)


@pytest.mark.parametrize("vk_backend", ["sets"], indirect=True)
def test_descriptor_sets_reuse_after_completion_and_pipeline_eviction(vk_backend, monkeypatch):
    backend = vk_backend
    backend.release_cached_memory()
    a = array(backend, np.arange(173, dtype=np.float32))
    backend.synchronize()
    allocated = []
    original = backend.vk.vkAllocateDescriptorSets

    def allocate(*args):
        sets = original(*args)
        allocated.append(sets)
        return sets

    monkeypatch.setattr(backend.vk, "vkAllocateDescriptorSets", allocate)
    previous = set(backend._pipelines)
    x, y = a + a, a + a
    # Both commands are in flight. They must own different descriptor sets.
    assert len(allocated) == 2
    assert allocated[0][0] != allocated[1][0]
    np.testing.assert_array_equal(pacho.asnumpy(x), np.arange(173, dtype=np.float32) * 2)
    backend.synchronize()
    source = next(iter(set(backend._pipelines) - previous))
    backend._destroy_pipeline(backend._pipelines.pop(source))
    # Rebuilding a pipeline must not invalidate its previously allocated sets.
    z = y + y
    np.testing.assert_array_equal(pacho.asnumpy(z), np.arange(173, dtype=np.float32) * 4)
    assert len(allocated) == 2
    assert backend._descriptor_layouts


def test_compiled_commands_share_submit_without_changing_results(vk_backend, monkeypatch):
    backend = vk_backend
    a = array(backend, np.arange(64, dtype=np.float32))
    operation = pacho.compile(lambda x: np.sum(x * x) + 2)
    operation(a)  # Compile and allocate resources outside counters.
    backend.synchronize()
    submitted = []
    markers = []
    original = backend.vk.vkQueueSubmit

    def submit(queue, count, infos, fence):
        submitted.extend(int(infos[index].commandBufferCount) for index in range(count))
        if count == 0:
            assert fence != backend.vk.ffi.NULL
            markers.append(fence)
        return original(queue, count, infos, fence)

    monkeypatch.setattr(backend.vk, "vkQueueSubmit", submit)
    result = operation(a)
    assert submitted == [1]  # dependent dispatches share one recorded command buffer
    assert not backend._recorded_commands
    np.testing.assert_array_equal(
        pacho.asnumpy(result), np.sum(np.arange(64, dtype=np.float32) ** 2) + 2
    )
    assert len(markers) == 1  # one completion marker at readback, not per dispatch


def test_batch_sync_exception_and_pending_limit_keep_queue_usable(vk_backend):
    backend = vk_backend
    a = array(backend, np.arange(37, dtype=np.float32))
    with pytest.raises(RuntimeError, match="test interruption"), backend._batch_commands():
        for _ in range(70):
            a = a + 1
            assert len(backend._pending) + len(backend._recorded_commands) < 64
            assert backend._queued_operations < 64
        np.testing.assert_array_equal(pacho.asnumpy(a), np.arange(37, dtype=np.float32) + 70)
        a = a + 2
        raise RuntimeError("test interruption")
    assert backend._batch_depth == 0 and not backend._recorded_commands
    assert backend._recording_command is None
    np.testing.assert_array_equal(pacho.asnumpy(a + 3), np.arange(37, dtype=np.float32) + 75)
