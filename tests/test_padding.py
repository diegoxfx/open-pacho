"""Conformance tests for opt-in physical padding around custom GEMM.

Padding is a storage and dispatch detail.  These tests keep the public result
shape and dtype visible, exercise the private backend factory through an
independent backend, and cover the three optional GPU adapters when a device is
available.
"""

from __future__ import annotations

import gc
import os
import weakref
from contextlib import contextmanager

import numpy as np
import pytest

import openpacho as pacho
from openpacho.backends import NumpyBackend
from openpacho.exceptions import BackendUnavailableError


def _array(value, backend="numpy", dtype=None):
    return pacho.asarray(value, dtype=dtype, backend=backend)


def _oracle(left, right, dtype=np.float32):
    """Use a wider host accumulator for the float32 GPU comparisons."""

    result = np.matmul(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    return np.asarray(result, dtype=dtype)


def _round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _expected_padded_shapes(left_shape, right_shape, output_shape):
    m, k = left_shape[-2:]
    right_k, n = right_shape[-2:]
    assert k == right_k
    mp, kp, npad = _round_up(m, 32), _round_up(k, 16), _round_up(n, 32)
    return (
        left_shape[:-2] + (mp, kp),
        right_shape[:-2] + (kp, npad),
        output_shape[:-2] + (mp, npad),
    )


def _padding_step(details):
    steps = [step for step in details["steps"] if step["kind"] == "padded_matmul"]
    assert len(steps) == 1, details["steps"]
    return steps[0]


def _shape_tuple(value):
    return tuple(tuple(dimension for dimension in shape) for shape in value)


@pytest.mark.parametrize(
    "flag",
    [1, 0, np.int64(1), np.int32(0), None, "true", np.array(True)],
    ids=["int-one", "int-zero", "numpy-int64", "numpy-int32", "none", "string", "array"],
)
def test_padding_flags_require_strict_booleans(flag):
    left = _array(np.ones((2, 2), dtype=np.float32))
    right = _array(np.ones((2, 2), dtype=np.float32))

    with pytest.raises(TypeError, match="padding"):
        pacho.matmul(left, right, padding=flag)
    with pytest.raises(TypeError, match="padding"):
        pacho.compile(lambda a, b: a + b, padding=flag)


@pytest.mark.parametrize("flag", [False, True, np.bool_(False), np.bool_(True)])
def test_padding_flags_accept_python_and_numpy_booleans(flag):
    left_host = np.arange(4, dtype=np.float32).reshape(2, 2)
    right_host = np.eye(2, dtype=np.float32)
    left, right = _array(left_host), _array(right_host)

    eager = pacho.matmul(left, right, padding=flag)
    np.testing.assert_array_equal(pacho.asnumpy(eager), left_host @ right_host)

    compiled = pacho.compile(lambda a, b: pacho.matmul(a, b), padding=flag)
    result = compiled(left, right)
    np.testing.assert_array_equal(pacho.asnumpy(result), left_host @ right_host)


def test_compile_padding_preserves_native_numpy_passthrough():
    calls = []

    @pacho.compile(padding=True)
    def native(left, right):
        calls.append("python")
        return np.matmul(left, right)

    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = native(left, right)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, left @ right)
    assert calls == ["python"]


@pytest.mark.parametrize(
    "shapes",
    [
        ((33, 17), (17, 65)),
        ((31, 16), (16, 31)),
        ((0, 17), (17, 65)),
        ((33, 17), (17, 0)),
        ((17,), (17, 65)),
        ((33, 17), (17,)),
    ],
)
def test_numpy_padding_is_a_native_noop_for_ordinary_cases(shapes):
    rng = np.random.default_rng(1201)
    left_host, right_host = (rng.normal(size=shape).astype(np.float32) for shape in shapes)
    left, right = _array(left_host), _array(right_host)
    backend = left._backend

    ordinary = pacho.matmul(left, right)
    hinted = pacho.matmul(left, right, padding=True)
    expected = left_host @ right_host

    assert ordinary.backend == hinted.backend == "numpy"
    assert ordinary._backend is backend and hinted._backend is backend
    np.testing.assert_array_equal(pacho.asnumpy(ordinary), expected)
    np.testing.assert_array_equal(pacho.asnumpy(hinted), expected)

    compiled = pacho.compile(lambda a, b: pacho.matmul(a, b), padding=True)
    details = compiled.explain(left, right)
    assert details["padding"] is True
    assert not any(step["kind"] == "padded_matmul" for step in details["steps"])
    np.testing.assert_array_equal(pacho.asnumpy(compiled(left, right)), expected)


def test_numpy_float64_padding_keeps_the_original_path():
    rng = np.random.default_rng(1202)
    left_host = rng.normal(size=(33, 17)).astype(np.float64)
    right_host = rng.normal(size=(17, 65)).astype(np.float64)
    left, right = _array(left_host), _array(right_host)

    compiled = pacho.compile(lambda a, b: pacho.matmul(a, b), padding=True)
    details = compiled.explain(left, right)
    assert details["padding"] is True
    assert not any(step["kind"] == "padded_matmul" for step in details["steps"])

    result = compiled(left, right)
    assert result.dtype == np.dtype("float64")
    np.testing.assert_allclose(pacho.asnumpy(result), left_host @ right_host)


class _NullPaddingBackend(NumpyBackend):
    """A backend with the seam but an explicit capability decline."""

    name = "null-padding"
    device = "test CPU"

    def __init__(self):
        self.factory_calls = 0
        self.matmul_calls = 0

    def prepare_padded_matmul(self, ashape, bshape, shape, dtype):
        del ashape, bshape, shape, dtype
        self.factory_calls += 1

    def matmul(self, a, b, shape, dtype):
        self.matmul_calls += 1
        return super().matmul(a, b, shape, dtype)


def test_padding_factory_none_falls_back_to_the_backend_matmul():
    backend = _NullPaddingBackend()
    left_host = np.arange(33 * 17, dtype=np.float32).reshape(33, 17) / 13
    right_host = np.arange(17 * 65, dtype=np.float32).reshape(17, 65) / 17
    left, right = _array(left_host, backend), _array(right_host, backend)

    result = pacho.matmul(left, right, padding=True)

    assert backend.factory_calls == 1
    assert backend.matmul_calls == 1
    np.testing.assert_allclose(pacho.asnumpy(result), left_host @ right_host)


class _RecordingPaddedPlan:
    def __init__(self, output_shape, dtype):
        self.output_shape = tuple(output_shape)
        self.dtype = np.dtype(dtype)
        self.run_calls = 0

    def run(self, left, right):
        self.run_calls += 1
        left_data = left._data.reshape(left.shape) if hasattr(left, "_data") else np.asarray(left)
        right_data = (
            right._data.reshape(right.shape) if hasattr(right, "_data") else np.asarray(right)
        )
        return np.asarray(left_data @ right_data, dtype=self.dtype).reshape(-1).copy()

    def explain(self):
        return {
            "input_shapes": (),
            "padded_input_shapes": (),
            "output_shape": self.output_shape,
            "padded_output_shape": self.output_shape,
            "scratch_bytes": 0,
            "alignment": (32, 16, 32),
            "bounds_checks_removed": True,
        }


class _RecordingPaddedBackend(NumpyBackend):
    """An arbitrary backend name proves the routing seam is capability based."""

    name = "opaque-test-backend"
    device = "test accelerator"

    def __init__(self):
        self.factory_calls = []
        self.plans = []

    def prepare_padded_matmul(self, ashape, bshape, shape, dtype):
        plan = _RecordingPaddedPlan(shape, dtype)
        self.factory_calls.append((tuple(ashape), tuple(bshape), tuple(shape), np.dtype(dtype)))
        self.plans.append(plan)
        return plan


@pytest.mark.parametrize("dispatches", [1, 2, 4])
def test_execution_scope_uses_padding_dispatches_not_logical_steps(dispatches):
    class ScopedBackend(_RecordingPaddedBackend):
        execution_scope_min_steps = 2
        scopes = 0

        @contextmanager
        def execution_scope(self):
            self.scopes += 1
            yield

        def prepare_padded_matmul(self, *args):
            plan = super().prepare_padded_matmul(*args)
            plan.dispatch_count = dispatches
            return plan

    backend = ScopedBackend()
    a = _array(np.ones((33, 17), dtype=np.float32), backend)
    b = _array(np.ones((17, 65), dtype=np.float32), backend)
    call = pacho.compile(lambda a, b: a @ b, padding=True).prepare(a, b)
    np.testing.assert_array_equal(pacho.asnumpy(call(a, b)), np.full((33, 65), 17))
    assert backend.scopes == int(dispatches >= 2)


def test_capability_factory_is_cached_and_prepared_calls_rebind_without_retaining_inputs():
    backend = _RecordingPaddedBackend()
    left_host = np.arange(33 * 17, dtype=np.float32).reshape(33, 17) / 19
    right_host = np.arange(17 * 65, dtype=np.float32).reshape(17, 65) / 23
    left = _array(left_host, backend)
    right = _array(right_host, backend)

    compiled = pacho.compile(lambda a, b: pacho.matmul(a, b), padding=True)
    left_ref, right_ref = weakref.ref(left), weakref.ref(right)
    prepared = compiled.prepare(left, right)

    assert len(backend.factory_calls) == 1
    assert backend.factory_calls[0][:3] == (left.shape, right.shape, (33, 65))
    del left, right
    gc.collect()
    assert left_ref() is None and right_ref() is None

    left_first = _array(left_host, backend)
    right_first = _array(right_host, backend)
    first = prepared(left_first, right_first)
    first_host = pacho.asnumpy(first)
    left_second = _array(left_host + 1, backend)
    right_second = _array(right_host - 1, backend)
    second = prepared(left_second, right_second)

    assert len(backend.factory_calls) == 1
    assert backend.plans[0].run_calls == 2
    assert first._data is not second._data
    np.testing.assert_array_equal(pacho.asnumpy(first), first_host)
    np.testing.assert_allclose(pacho.asnumpy(first), left_host @ right_host)
    np.testing.assert_allclose(pacho.asnumpy(second), (left_host + 1) @ (right_host - 1))

    # The ordinary compiled entry point hits the same cached plan.
    compiled(left_first, right_first)
    assert compiled.cache_info()["misses"] == 1
    assert compiled.cache_info()["hits"] == 1
    details = prepared.explain()
    assert details["padding"] is True
    assert _padding_step(details)["kind"] == "padded_matmul"


def test_explicit_padding_inside_default_compile_is_captured():
    backend = _RecordingPaddedBackend()
    left_host = np.ones((33, 17), dtype=np.float32)
    right_host = np.ones((17, 65), dtype=np.float32)
    left, right = _array(left_host, backend), _array(right_host, backend)

    @pacho.compile
    def product(a, b):
        return pacho.matmul(a, b, padding=True)

    details = product.explain(left, right)
    assert details["padding"] is False
    step = _padding_step(details)
    assert len(backend.factory_calls) == 1
    assert step["kind"] == "padded_matmul"
    np.testing.assert_array_equal(pacho.asnumpy(product(left, right)), left_host @ right_host)


def test_default_padding_never_calls_factory_and_failure_is_not_hidden(monkeypatch):
    backend = _RecordingPaddedBackend()
    left = _array(np.ones((33, 17), dtype=np.float32), backend)
    right = _array(np.ones((17, 65), dtype=np.float32), backend)

    def broken(*args):
        raise RuntimeError("padding preparation failed")

    monkeypatch.setattr(backend, "prepare_padded_matmul", broken)
    pacho.matmul(left, right)
    pacho.compile(lambda a, b: a @ b)(left, right)
    with pytest.raises(RuntimeError, match="padding preparation failed"):
        pacho.matmul(left, right, padding=True)
    with pytest.raises(RuntimeError, match="padding preparation failed"):
        pacho.compile(lambda a, b: a @ b, padding=True)(left, right)


def _padding_gpu_required(name):
    return os.environ.get(f"PACHO_REQUIRE_{name.upper()}", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.fixture(
    scope="module", params=("opencl", "vulkan", "cuda"), ids=("opencl", "vulkan", "cuda")
)
def gpu_backend(request):
    name = request.param
    try:
        if name == "opencl":
            backend = pacho.get_backend(
                "opencl",
                device=os.environ.get("PACHO_OPENCL_DEVICE") or None,
                blas="custom",
            )
        elif name == "vulkan":
            backend = pacho.get_backend(
                "vulkan", device=os.environ.get("PACHO_VULKAN_DEVICE") or None
            )
        else:
            backend = pacho.get_backend(
                "cuda", device=os.environ.get("PACHO_CUDA_DEVICE") or "0", blas="custom"
            )
    except (BackendUnavailableError, ImportError) as exc:
        message = f"{name} unavailable{f' ({exc})' if str(exc) else ''}"
        if _padding_gpu_required(name):
            pytest.fail(message)
        pytest.skip(message)
    yield backend
    backend.synchronize()


def _require_register_gemm(backend):
    if not getattr(backend, "_register_gemm", False):
        pytest.skip(f"{backend.name} device does not advertise register GEMM")


@pytest.mark.parametrize(
    "shapes",
    [
        ((33, 17), (17, 65)),
        ((65, 47), (47, 67)),
        ((64, 32), (32, 96)),
        ((2, 1, 33, 17), (1, 3, 17, 65)),
    ],
)
def test_gpu_padding_rounds_to_tiles_and_preserves_logical_results(gpu_backend, shapes):
    _require_register_gemm(gpu_backend)
    rng = np.random.default_rng(8301)
    left_host, right_host = (
        rng.uniform(-0.5, 0.5, size=shape).astype(np.float32) for shape in shapes
    )
    left_before, right_before = left_host.copy(), right_host.copy()
    left, right = _array(left_host, gpu_backend), _array(right_host, gpu_backend)
    provider = getattr(gpu_backend, "blas", None)

    compiled = pacho.compile(lambda a, b: pacho.matmul(a, b), padding=True)
    details = compiled.explain(left, right)
    step = _padding_step(details)
    metadata = step["padding"]
    output_shape = np.matmul(left_host, right_host).shape
    expected_input_shapes = (left.shape, right.shape)
    expected_padded = _expected_padded_shapes(*expected_input_shapes, output_shape)

    assert details["padding"] is True
    assert tuple(step["shape"]) == output_shape
    assert _shape_tuple(metadata["input_shapes"]) == expected_input_shapes
    assert _shape_tuple(metadata["padded_input_shapes"]) == expected_padded[:2]
    assert tuple(metadata["output_shape"]) == output_shape
    assert tuple(metadata["padded_output_shape"]) == expected_padded[2]
    assert tuple(metadata["alignment"]) == (32, 16, 32)
    assert metadata["bounds_checks_removed"] is True
    if expected_padded == expected_input_shapes + (output_shape,):
        assert metadata["scratch_bytes"] == 0
    else:
        assert metadata["scratch_bytes"] > 0

    eager = pacho.matmul(left, right, padding=True)
    first = compiled(left, right)
    prepared = compiled.prepare(left, right)
    second = prepared(left, right)
    expected = _oracle(left_host, right_host)
    for result in (eager, first, second):
        assert result.shape == output_shape
        assert result.dtype == np.dtype("float32")
        np.testing.assert_allclose(pacho.asnumpy(result), expected, rtol=5e-4, atol=5e-4)

    # The copy in the padding path must leave the logical inputs alone.  Keep
    # both result objects alive across the second call to catch aliasing.
    first_snapshot = pacho.asnumpy(first)
    again = prepared(left, right)
    assert first._data is not again._data
    np.testing.assert_array_equal(pacho.asnumpy(left), left_before)
    np.testing.assert_array_equal(pacho.asnumpy(right), right_before)
    np.testing.assert_array_equal(pacho.asnumpy(first), first_snapshot)
    assert getattr(gpu_backend, "blas", None) == provider

    # The real pad kernels must rebind new input buffers on cache hits too.
    changed_left = _array(left_host + np.float32(0.25), gpu_backend)
    changed_right = _array(right_host - np.float32(0.125), gpu_backend)
    changed = prepared(changed_left, changed_right)
    np.testing.assert_allclose(
        pacho.asnumpy(changed),
        _oracle(left_host + np.float32(0.25), right_host - np.float32(0.125)),
        rtol=5e-4,
        atol=5e-4,
    )
    np.testing.assert_array_equal(pacho.asnumpy(first), first_snapshot)


@pytest.mark.parametrize(
    "shapes",
    [((33, 17), (17, 65)), ((2, 1, 33, 17), (1, 3, 17, 65))],
)
def test_gpu_padding_crops_before_sum_and_mean(gpu_backend, shapes):
    _require_register_gemm(gpu_backend)
    rng = np.random.default_rng(8302)
    left_host, right_host = (
        rng.uniform(-0.5, 0.5, size=shape).astype(np.float32) for shape in shapes
    )
    left, right = _array(left_host, gpu_backend), _array(right_host, gpu_backend)

    @pacho.compile(padding=True)
    def product_and_reductions(a, b):
        product = pacho.matmul(a, b)
        return product, pacho.sum(product), pacho.mean(product)

    details = product_and_reductions.explain(left, right)
    _padding_step(details)
    product, total, average = product_and_reductions(left, right)
    expected = _oracle(left_host, right_host)

    np.testing.assert_allclose(pacho.asnumpy(product), expected, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(pacho.asnumpy(total), np.sum(expected), rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(pacho.asnumpy(average), np.mean(expected), rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gpu_padding_rejects_ineligible_dtype_or_small_matrix(gpu_backend, dtype):
    _require_register_gemm(gpu_backend)
    if np.dtype(dtype) == np.dtype("float64") and not getattr(gpu_backend, "fp64", True):
        pytest.skip("selected device does not support float64")
    rng = np.random.default_rng(8303)
    # Float64 is tested at an otherwise eligible size so this branch proves
    # the dtype policy itself declines padding.  Float32 uses the small-M
    # fallback.
    rows = 33 if np.dtype(dtype) == np.dtype("float64") else 31
    left_host = rng.normal(size=(rows, 17)).astype(dtype)
    right_host = rng.normal(size=(17, 65)).astype(dtype)
    left, right = _array(left_host, gpu_backend), _array(right_host, gpu_backend)
    compiled = pacho.compile(lambda a, b: pacho.matmul(a, b), padding=True)

    details = compiled.explain(left, right)
    assert not any(step["kind"] == "padded_matmul" for step in details["steps"])
    result = compiled(left, right)
    expected = np.matmul(left_host, right_host)
    np.testing.assert_allclose(pacho.asnumpy(result), expected, rtol=5e-4, atol=5e-4)


def test_gpu_padding_preserves_special_value_classification_and_signed_zero(gpu_backend):
    _require_register_gemm(gpu_backend)
    left_host = np.zeros((33, 17), dtype=np.float32)
    right_host = np.zeros((17, 65), dtype=np.float32)
    left_host[0, 0], right_host[0, 0] = np.nan, 1.0
    left_host[1, 1], right_host[1, 1], right_host[1, 2] = np.inf, 0.0, 1.0
    left_host[2, 2], right_host[2, 3] = -np.inf, 1.0
    left_host[3, 3], right_host[3, 4] = -np.float32(0.0), 1.0
    left_host[4, 4], right_host[4, 5] = np.float32(0.0), -1.0
    left_host[5, 5], right_host[5, 6] = np.nextafter(np.float32(0), np.float32(1)), 1.0
    left_host[6, 6], right_host[6, 7] = np.finfo(np.float32).tiny, 1.0
    left_host[7, 7], right_host[7, 8] = np.finfo(np.float32).max, 1.0

    left, right = _array(left_host, gpu_backend), _array(right_host, gpu_backend)
    padded = pacho.asnumpy(pacho.matmul(left, right, padding=True))
    ordinary = pacho.asnumpy(pacho.matmul(left, right, padding=False))

    for predicate in (np.isnan, np.isinf, np.isfinite):
        np.testing.assert_array_equal(predicate(padded), predicate(ordinary))
    for index in ((3, 4), (4, 5)):
        assert np.signbit(padded[index]) == np.signbit(ordinary[index])
    # Compare the crafted finite lanes against the existing unpadded route.
    # This allows a device to flush a denormal in both routes while still
    # detecting a padding-specific change in the result or its signed zero.
    for index in ((3, 4), (4, 5), (5, 6), (6, 7), (7, 8)):
        np.testing.assert_array_equal(padded[index], ordinary[index])
    assert np.isfinite(padded[5, 6]) == np.isfinite(ordinary[5, 6])
    assert np.isfinite(padded[6, 7]) == np.isfinite(ordinary[6, 7])
    assert np.isfinite(padded[7, 8]) == np.isfinite(ordinary[7, 8])
    np.testing.assert_array_equal(pacho.asnumpy(left), left_host)
    np.testing.assert_array_equal(pacho.asnumpy(right), right_host)


def test_vendor_blas_providers_decline_padding_without_initialization():
    """Vendor providers remain the owner of ordinary GEMM dispatch."""

    from openpacho._cuda import CUDABackend
    from openpacho._opencl import OpenCLBackend

    shapes = ((33, 17), (17, 65), (33, 65))
    opencl = OpenCLBackend.__new__(OpenCLBackend)
    opencl._blas = object()
    opencl._register_gemm = True
    assert opencl.prepare_padded_matmul(*shapes, np.dtype("float32")) is None
    assert opencl._blas is not None

    cuda = CUDABackend.__new__(CUDABackend)
    cuda.blas = "cublas"
    cuda._register_gemm = True
    assert cuda.prepare_padded_matmul(*shapes, np.dtype("float32")) is None
    assert cuda.blas == "cublas"
