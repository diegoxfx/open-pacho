"""CPU-only routing and source-template checks for the RNG adapters."""

from math import prod

import numpy as np

import openpacho as p
from openpacho import _random_cuda, _random_opencl, _random_vulkan
from openpacho.backends import NumpyBackend


def test_public_generator_uses_backend_implementation_without_name_registry():
    class ExampleBackend(NumpyBackend):
        name = "independent-device"

    actual = p.random.default_rng(47, backend=ExampleBackend()).normal(size=(3, 5))
    expected = p.random.default_rng(47, backend="numpy").normal(size=(3, 5))
    assert p.get_backend(actual).name == "independent-device"
    np.testing.assert_array_equal(p.asnumpy(actual), p.asnumpy(expected))


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def empty(self, shape, dtype):
        return np.empty(shape, dtype).reshape(-1)

    def _run(self, *args):
        self.calls.append(args)


def test_opencl_routes_dynamic_values_to_cached_source():
    backend = RecordingBackend()
    shape = (3, 4)
    out = _random_opencl.random_data(
        backend, shape, np.dtype("float32"), "uniform", 17, 23, np.float32(-3), np.float32(7)
    )
    first = backend.calls[-1]
    out2 = _random_opencl.random_data(
        backend, shape, np.dtype("float32"), "uniform", 19, 29, np.float32(-5), np.float32(11)
    )
    second = backend.calls[-1]

    assert out.shape == out2.shape == (prod(shape),)
    assert first[0] is second[0]
    assert first[1] == second[1] == prod(shape)
    assert first[2:-1] != second[2:-1]
    assert "mul_hi" in first[0]
    assert "__umulhi" not in first[0]
    assert "0x00000011" not in first[0] and "0x00000017" not in first[0]


def test_cuda_source_cache_includes_count_but_runtime_values_are_dynamic():
    backend = RecordingBackend()
    _random_cuda.random_data(
        backend, (5,), np.dtype("float64"), "exponential", 17, 23, np.float64(0), np.float64(2)
    )
    first = backend.calls[-1]
    _random_cuda.random_data(
        backend, (5,), np.dtype("float64"), "exponential", 19, 29, np.float64(0), np.float64(7)
    )
    second = backend.calls[-1]
    _random_cuda.random_data(
        backend, (6,), np.dtype("float64"), "exponential", 17, 23, np.float64(0), np.float64(2)
    )
    third = backend.calls[-1]

    assert first[0] is second[0]
    assert first[0] is not third[0]
    assert first[1] == second[1] == 1
    assert third[1] == 1
    assert first[2:-1] != second[2:-1]
    assert "__umulhi" in first[0]
    assert "log1p" in first[0]
    assert "0x00000011" not in first[0] and "0x00000017" not in first[0]
    assert ">=5u" in first[0] and ">=6u" in third[0]


def test_vulkan_routes_host_scalars_and_keeps_glsl_log1m_in_adapter():
    backend = RecordingBackend()
    _random_vulkan.random_data(
        backend, (5,), np.dtype("float32"), "normal", 17, 23, np.float32(3), np.float32(2)
    )
    first = backend.calls[-1]
    _random_vulkan.random_data(
        backend, (5,), np.dtype("float32"), "normal", 19, 29, np.float32(7), np.float32(4)
    )
    second = backend.calls[-1]
    _random_vulkan.random_data(
        backend, (6,), np.dtype("float32"), "normal", 17, 23, np.float32(3), np.float32(2)
    )
    third = backend.calls[-1]

    assert first[0] is second[0]
    assert first[0] is third[0]
    assert first[1] == second[1] == 5
    assert third[1] == 6
    assert first[2][0][1] == np.dtype("float32")
    assert [scalar._data.item() for scalar in first[3]] != [
        scalar._data.item() for scalar in second[3]
    ]
    assert "umulExtended" in first[0]
    assert "log1p" not in first[0]
    assert "nl=u+u*u" in first[0]
