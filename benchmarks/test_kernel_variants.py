"""Candidate resource/border constraints independent of GPU availability."""

import pytest

from benchmarks.kernel_variants import VARIANTS, Tile, source


def test_padding_memory_cost_is_explicit():
    assert Tile().shared_bytes == 4096
    assert Tile(pad_left=1).shared_bytes == 4224
    assert Tile(pad_left=1, pad_right=1).shared_bytes == 4288
    assert Tile(transpose_left=True, pad_left=1).shared_bytes == 4160
    assert Tile(k=32).shared_bytes == 8192


def test_specialization_preserves_each_partial_axis():
    for backend in ("opencl", "vulkan", "cuda"):
        aligned, _, _ = source(backend, 64, 32, 64, Tile(specialize_bounds=True))
        assert "ar<" not in aligned and "bk<" not in aligned and "if(row" not in aligned
        partial, _, _ = source(backend, 65, 33, 67, Tile(specialize_bounds=True))
        for condition in ("ar<65", "ak<33", "bk<33", "bc<67", "if(row"):
            assert condition in partial
        k_tail, _, _ = source(backend, 64, 33, 64, Tile(specialize_bounds=True))
        assert "ar<" not in k_tail and "ak<33" in k_tail and "bk<33" in k_tail


@pytest.mark.parametrize("backend", ["opencl", "cuda", "vulkan"])
def test_candidate_loads_preserve_float32_baseline(backend):
    # A double zero in a conditional silently penalized the old OpenCL control.
    # Every generated tuning candidate must use the corrected arithmetic type.
    for tile in VARIANTS.values():
        body, _, _ = source(backend, 65, 33, 67, tile)
        assert body.count(":0.0f") == 2
        assert "=0.0;" not in body


@pytest.mark.parametrize("shape", [(0, 16, 32), (32, 0, 32), (2**30, 32, 32)])
def test_invalid_index_geometry(shape):
    with pytest.raises(ValueError):
        source("opencl", *shape, Tile())
