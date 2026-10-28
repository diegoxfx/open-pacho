"""CPU-only source and dispatch checks for the rectangular copy adapters."""

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from openpacho import _padding_copy


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def empty(self, shape, dtype):
        return np.empty(np.prod(shape, dtype=np.int64), dtype=dtype)

    def _run(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _array(shape=(2, 3), dtype=np.float32):
    shape = tuple(shape)
    return SimpleNamespace(
        shape=shape,
        dtype=np.dtype(dtype),
        size=np.prod(shape, dtype=np.int64),
        _data=object(),
    )


def test_opencl_expansion_uses_uint_words_and_exact_destination_count():
    backend = RecordingBackend()
    output = _padding_copy.opencl_copy(backend, _array(), (3, 4))

    assert output.shape == (12,)
    source, count, input_storage, output_storage = backend.calls[0][0]
    assert count == 12
    assert input_storage is not None and output_storage is output
    assert "__global const uint" in source
    assert "__global uint" in source
    assert "if(row<2u&&col<3u)" in source
    assert "else x1[i]=0u" in source


def test_cuda_dispatch_has_a_bounded_128_thread_kernel():
    backend = RecordingBackend()
    _padding_copy.cuda_copy(backend, _array((5, 7)), (5, 7))

    source, blocks, *_ = backend.calls[0][0]
    assert blocks == 1
    assert backend.calls[0][1] == {"threads": 128}
    assert "const unsigned int*x0" in source
    assert "unsigned int*x1" in source
    assert "if(i>=35u) return" in source


def test_vulkan_binds_float32_storage_as_uint32_words():
    backend = RecordingBackend()
    _padding_copy.vulkan_copy(backend, _array((2, 3)), (1, 2))

    body, count, buffers = backend.calls[0][0]
    assert count == 2
    assert [dtype for _, dtype in buffers] == [np.dtype("uint32"), np.dtype("uint32")]
    assert "x1[i]=x0" in body


def test_empty_output_and_zero_source_dimension_never_launch_or_read():
    backend = RecordingBackend()
    output = _padding_copy.opencl_copy(backend, _array((2, 0)), (2, 0))
    assert output.size == 0
    assert backend.calls == []

    backend = RecordingBackend()
    _padding_copy.opencl_copy(backend, _array((2, 0)), (2, 3))
    source = backend.calls[0][0][0]
    assert "x1[i]=0u" in source
    assert "x0[" not in source


@pytest.mark.parametrize(
    "shape, expected",
    [((3,), "rank"), ((2, 4, 3), "batch"), ((2, 3), "float32")],
)
def test_validation_rejects_malformed_inputs(shape, expected):
    dtype = np.float64 if expected == "float32" else np.float32
    with pytest.raises((TypeError, ValueError), match=expected):
        destination = (3, 4, 3) if expected == "batch" else (2, 3)
        _padding_copy.opencl_copy(RecordingBackend(), _array(shape, dtype), destination)


def test_cached_layout_normalizes_metadata_and_vulkan_rebinds_current_storage():
    _padding_copy._layout.cache_clear()
    _padding_copy._body.cache_clear()
    backend = RecordingBackend()
    first, second = _array(), _array()
    first_bits = np.array(
        [0x00000001, 0x80000000, 0x7FC12345, 0x7FA12345, 0xFF800000, 0x007FFFFF],
        dtype=np.uint32,
    )
    second_bits = first_bits ^ np.uint32(0x80000000)
    first._data = first_bits.copy().view(np.float32)
    second._data = second_bits.copy().view(np.float32)
    second.shape = [np.int64(2), np.int32(3)]
    second.dtype = "float32"

    first_output = _padding_copy.vulkan_copy(backend, first, (3, 4))
    second_output = _padding_copy.vulkan_copy(backend, second, [np.int32(3), np.int64(4)])

    assert _padding_copy._layout.cache_info().misses == 1
    assert _padding_copy._layout.cache_info().hits == 1
    assert _padding_copy._body.cache_info().misses == 1
    assert _padding_copy._body.cache_info().hits == 1
    first_body, _, first_buffers = backend.calls[0][0]
    second_body, _, second_buffers = backend.calls[1][0]
    assert first_body is second_body
    assert first_buffers[0][0] is first._data
    assert second_buffers[0][0] is second._data
    assert first_buffers[1][0] is first_output
    assert second_buffers[1][0] is second_output
    assert first_output is not second_output
    # Source/body caching must only see geometry, never convert payload words.
    # Actual device copy equivalence is covered by test_padding_metadata.py.
    np.testing.assert_array_equal(first_buffers[0][0].view(np.uint32), first_bits)
    np.testing.assert_array_equal(second_buffers[0][0].view(np.uint32), second_bits)
    assert "x1[i]=0u" in first_body


@pytest.mark.parametrize(
    "attribute, value, error, match",
    [
        ("size", 5, ValueError, "size does not match"),
        ("size", 6.0, TypeError, "size must be an integer"),
        ("shape", (2, 3.0), TypeError, "integer dimensions"),
        ("shape", (2, -3), ValueError, "negative dimensions"),
        ("dtype", np.float64, TypeError, "float32"),
        ("flags", SimpleNamespace(c_contiguous=False), ValueError, "C-contiguous input"),
        ("_data", np.arange(12, dtype=np.float32)[::2], ValueError, "C-contiguous backend"),
    ],
)
def test_cached_layout_does_not_skip_current_input_validation(attribute, value, error, match):
    _padding_copy._layout.cache_clear()
    backend, array = RecordingBackend(), _array()
    _padding_copy.opencl_copy(backend, array, (3, 4))
    assert _padding_copy._layout.cache_info().currsize == 1
    backend.calls.clear()
    setattr(array, attribute, value)

    with pytest.raises(error, match=match):
        _padding_copy.opencl_copy(backend, array, (3, 4))
    assert backend.calls == []


def test_cached_layout_still_requires_storage_on_each_call():
    backend, array = RecordingBackend(), _array()
    _padding_copy.cuda_copy(backend, array, (3, 4))
    backend.calls.clear()
    del array._data

    with pytest.raises(TypeError, match="backend storage"):
        _padding_copy.cuda_copy(backend, array, (3, 4))
    assert backend.calls == []


@pytest.mark.parametrize(
    "source_shape, destination_shape, match",
    [
        ((2**30, 3), (2, 3), "input exceeds"),
        ((2, 3), (2**30, 3), "output exceeds"),
    ],
)
def test_layout_checks_index_limits_before_dispatch(source_shape, destination_shape, match):
    backend = RecordingBackend()
    with pytest.raises(ValueError, match=match):
        _padding_copy.cuda_copy(backend, _array(source_shape), destination_shape)
    assert backend.calls == []


def test_layout_and_body_caches_are_bounded_and_never_keep_input_objects():
    class Tracked:
        pass

    _padding_copy._layout.cache_clear()
    _padding_copy._body.cache_clear()
    array, storage = Tracked(), Tracked()
    array.__dict__.update(_array().__dict__)
    array._data = storage
    array.dtype = np.dtype(np.float32, metadata={"owner": storage})
    array_ref, storage_ref = weakref.ref(array), weakref.ref(storage)
    for width in range(3, 163):
        source, destination, *_ = _padding_copy._validate(array, (3, width))
        _padding_copy._body(source, destination, "uint")
    assert _padding_copy._layout.cache_info().currsize == 128
    assert _padding_copy._body.cache_info().currsize == 128

    del array, storage, _
    gc.collect()
    assert array_ref() is None
    assert storage_ref() is None
