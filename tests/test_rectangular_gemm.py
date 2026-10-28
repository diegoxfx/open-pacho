"""Small-row tiles preserve ownership, borders, arithmetic and fallback regions."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

import openpacho as p
from openpacho._gemm_rect import select_tile, source
from openpacho.exceptions import BackendUnavailableError


@pytest.mark.parametrize("name,tile", [("opencl", (16, 32, 16)), ("cuda", (16, 16, 32))])
def test_source_uses_bounded_float32_loads_and_cooperative_barriers(name, tile):
    text, groups = source(name, (33, 4081), (4081, 257), tile)
    assert groups == 3 * ((257 + tile[1] - 1) // tile[1])
    assert text.count(":0.0f") == 2
    assert "ak<4081" in text and "bk<4081" in text
    assert "return" not in text
    assert "double" not in text
    assert text.count("barrier" if name == "opencl" else "__syncthreads") == 2


def test_selection_retains_small_aligned_batched_and_wide_fallbacks():
    backend = SimpleNamespace(_small_row_gemm_config=((16, 32, 16), 4081))
    assert select_tile(backend, (65, 4081), (4081, 257)) == (16, 32, 16)
    for a, b in (
        ((31, 8191), (8191, 2051)),
        ((96, 8191), (8191, 2051)),
        ((65, 4080), (4080, 2051)),
        ((64, 8192), (8192, 2048)),
        ((65, 8191), (8191, 4096)),
        ((65, 8191), (8191, 255)),
        ((2, 65, 8191), (8191, 2051)),
    ):
        assert select_tile(backend, a, b) is None
    assert select_tile(SimpleNamespace(), (65, 8191), (8191, 2051)) is None


def test_rectangular_source_cache_and_unsupported_tile():
    source.cache_clear()
    for n in range(256, 390):
        source("opencl", (65, 4095), (4095, n), (16, 32, 16))
    assert source.cache_info().currsize == 128
    with pytest.raises(ValueError, match="tile"):
        source("opencl", (65, 4095), (4095, 257), (8, 8, 8))
    with pytest.raises(ValueError, match="OpenCL and CUDA"):
        source("vulkan", (65, 4095), (4095, 257), (16, 32, 16))


@pytest.fixture(params=("opencl", "cuda"))
def rectangular_backend(request):
    try:
        return p.get_backend(request.param, blas="custom")
    except BackendUnavailableError as exc:
        if os.environ.get(f"PACHO_REQUIRE_{request.param.upper()}") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


@pytest.mark.parametrize("m,k,n", [(33, 4081, 257), (65, 4095, 1023), (95, 8191, 2051)])
def test_public_rectangular_path_rebinds_without_copies_or_rounding_changes(
    rectangular_backend, monkeypatch, m, k, n
):
    backend = rectangular_backend
    rng = np.random.default_rng(613)
    # Dyadic products/sums remain exactly representable for this finite range.
    host = [rng.integers(-2, 3, shape).astype("float32") / 16 for shape in ((m, k), (k, n))]
    a, b = (p.asarray(h, backend=backend) for h in host)
    twice_a = p.asarray(host[0] * 2, backend=backend)
    config = backend._small_row_gemm_config
    monkeypatch.setattr(backend, "_small_row_gemm_config", None)
    baseline = p.asnumpy(a @ b)
    monkeypatch.setattr(backend, "_small_row_gemm_config", config)
    assert select_tile(backend, a.shape, b.shape) is not None
    compiled = p.compile(lambda a, b: a @ b).prepare(a, b)
    with monkeypatch.context() as guard:
        guard.setattr(backend, "download", lambda *args: pytest.fail("Hidden download"))
        guard.setattr(backend, "gather", lambda *args: pytest.fail("Unexpected packing/crop"))
        first, second = compiled(a, b), compiled(twice_a, b)
    np.testing.assert_array_equal(p.asnumpy(first).view("uint32"), baseline.view("uint32"))
    np.testing.assert_array_equal(p.asnumpy(first), host[0].astype("float64") @ host[1])
    np.testing.assert_array_equal(p.asnumpy(second), baseline * 2)
    assert first._data is not second._data


def test_rectangular_special_values_match_direct_kernel(rectangular_backend, monkeypatch):
    backend = rectangular_backend
    left = np.zeros((33, 4081), dtype="float32")
    right = np.zeros((4081, 257), dtype="float32")
    left[0, 0], left[1, 0], left[2, 0] = np.inf, np.nan, np.nextafter(np.float32(0), np.float32(1))
    right[0, :3] = [0, 1, -1]
    a, b = (p.asarray(h, backend=backend) for h in (left, right))
    config = backend._small_row_gemm_config
    monkeypatch.setattr(backend, "_small_row_gemm_config", None)
    expected = p.asnumpy(a @ b)
    monkeypatch.setattr(backend, "_small_row_gemm_config", config)
    actual = p.asnumpy(a @ b)
    np.testing.assert_array_equal(actual, expected)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view("uint32"), expected[finite].view("uint32"))


def test_explicit_padding_keeps_its_aligned_kernel(rectangular_backend, monkeypatch):
    from openpacho import _gemm_rect

    backend = rectangular_backend
    a = p.ones((33, 4081), backend=backend, dtype="float32")
    b = p.ones((4081, 257), backend=backend, dtype="float32")
    monkeypatch.setattr(_gemm_rect, "source", lambda *a, **k: pytest.fail("Wrong padding kernel"))
    result = p.matmul(a, b, padding=True)
    np.testing.assert_array_equal(p.asnumpy(result), np.full((33, 257), 4081, "float32"))
