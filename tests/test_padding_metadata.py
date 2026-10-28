"""Metadata and optional-backend bit-exact checks for register-GEMM padding."""

import os

import numpy as np
import pytest

import openpacho as p
from openpacho._padding import prepare_register_padding
from openpacho.exceptions import BackendUnavailableError


class MetadataBackend:
    def __init__(self):
        self.empty_calls = []

    def empty(self, shape, dtype):
        self.empty_calls.append((tuple(shape), np.dtype(dtype)))
        raise AssertionError("metadata preparation must not allocate buffers")


def _plan(ashape, bshape, shape, dtype=np.float32, *, max_buffer_bytes=None, backend=None):
    return prepare_register_padding(
        backend or MetadataBackend(),
        ashape,
        bshape,
        shape,
        dtype,
        lambda *_args: None,
        max_buffer_bytes=max_buffer_bytes,
    )


def test_register_padding_aligns_rectangular_gemm_and_reports_three_copies():
    plan = _plan((33, 17), (17, 65), (33, 65))

    assert plan is not None
    assert plan.padded_input_shapes == ((64, 32), (32, 96))
    assert plan.padded_output_shape == (64, 96)
    explanation = plan.explain()
    assert explanation["scratch_bytes"] == 45056
    assert explanation["copy_dispatches"] == 3
    assert plan.dispatch_count == 4


def test_register_padding_keeps_broadcast_batch_prefixes_unchanged():
    plan = _plan((2, 1, 33, 17), (1, 3, 17, 65), (2, 3, 33, 65))

    assert plan is not None
    assert plan.padded_input_shapes == ((2, 1, 64, 32), (1, 3, 32, 96))
    assert plan.padded_output_shape == (2, 3, 64, 96)


def test_register_padding_for_aligned_gemm_has_no_scratch_or_copy_dispatches():
    plan = _plan((64, 32), (32, 96), (64, 96))

    assert plan is not None
    explanation = plan.explain()
    assert explanation["scratch_bytes"] == 0
    assert explanation["copy_dispatches"] == 0
    assert plan.dispatch_count == 1


@pytest.mark.parametrize(
    "ashape, bshape, shape, dtype",
    [
        ((17,), (17, 65), (65,), np.float32),
        ((31, 16), (16, 32), (31, 32), np.float32),
        ((0, 32), (32, 96), (0, 96), np.float32),
        ((33, 17), (17, 65), (33, 65), np.float64),
    ],
    ids=("vector", "small", "empty", "float64"),
)
def test_register_padding_declines_ineligible_shapes(ashape, bshape, shape, dtype):
    assert _plan(ashape, bshape, shape, dtype) is None


def test_register_padding_declines_index_overflow_before_allocation():
    backend = MetadataBackend()
    assert (
        _plan(
            (2**31, 16),
            (16, 32),
            (2**31, 32),
            backend=backend,
        )
        is None
    )
    assert backend.empty_calls == []


def test_register_padding_declines_buffer_limit_before_allocation():
    backend = MetadataBackend()
    assert (
        _plan(
            (33, 17),
            (17, 65),
            (33, 65),
            max_buffer_bytes=24575,
            backend=backend,
        )
        is None
    )
    assert backend.empty_calls == []


_BIT_PATTERNS = np.array(
    [
        0x00000001,  # +minimum subnormal
        0x80000001,  # -minimum subnormal
        0x007FFFFF,  # +maximum subnormal
        0x807FFFFF,  # -maximum subnormal
        0x00800000,  # +minimum normal
        0x80800000,  # -minimum normal
        0x00000000,  # +zero
        0x80000000,  # -zero
        0x7F800000,  # +infinity
        0xFF800000,  # -infinity
        0x7FC12345,  # positive NaN payload
        0xFFC54321,  # negative NaN payload
    ],
    dtype=np.uint32,
)


def _bit_pattern_array(shape):
    return np.resize(_BIT_PATTERNS, int(np.prod(shape))).reshape(shape).view(np.float32)


def _bits(value):
    return np.asarray(value, dtype=np.float32).view(np.uint32)


@pytest.fixture(params=("opencl", "vulkan", "cuda"), ids=("opencl", "vulkan", "cuda"))
def padding_gpu_backend(request):
    """Missing optional backends skip unless explicitly required by CI."""
    try:
        if request.param == "opencl":
            return p.get_backend(
                "opencl",
                device=os.environ.get("PACHO_OPENCL_DEVICE") or None,
                blas="custom",
            )
        if request.param == "cuda":
            return p.get_backend(
                "cuda",
                device=os.environ.get("PACHO_CUDA_DEVICE") or None,
                blas="custom",
            )
        return p.get_backend("vulkan", device=os.environ.get("PACHO_VULKAN_DEVICE") or None)
    except (BackendUnavailableError, ImportError) as exc:
        if os.environ.get(f"PACHO_REQUIRE_{request.param.upper()}", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            pytest.fail(str(exc))
        pytest.skip(str(exc) or f"{request.param} unavailable")


@pytest.mark.parametrize("batch_shape", [(2,), (2, 1)])
def test_raw_copy_preserves_bits_pads_with_zero_and_crops_exactly(padding_gpu_backend, batch_shape):
    from openpacho import _padding_copy
    from openpacho._array import ndarray

    source_shape = batch_shape + (3, 5)
    packed_shape = batch_shape + (5, 7)
    cropped_shape = batch_shape + (2, 4)
    source_host = _bit_pattern_array(source_shape)
    source = p.asarray(source_host, backend=padding_gpu_backend)
    copy = getattr(_padding_copy, f"{padding_gpu_backend.name}_copy")

    packed_data = copy(padding_gpu_backend, source, packed_shape)
    packed = ndarray._wrap(packed_data, packed_shape, np.dtype("float32"), padding_gpu_backend)
    packed_bits = _bits(p.asnumpy(packed))
    expected_packed = np.zeros(packed_shape, dtype=np.uint32)
    expected_packed[(...,) + (slice(0, 3), slice(0, 5))] = _bits(source_host)
    np.testing.assert_array_equal(packed_bits, expected_packed)

    cropped_data = copy(padding_gpu_backend, packed, cropped_shape)
    cropped = ndarray._wrap(cropped_data, cropped_shape, np.dtype("float32"), padding_gpu_backend)
    np.testing.assert_array_equal(_bits(p.asnumpy(cropped)), _bits(source_host)[..., :2, :4])


__all__ = []
