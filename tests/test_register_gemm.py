"""Private output blocks preserve batching, borders and public ownership."""

import os
import re

import numpy as np
import pytest

import openpacho as p
from openpacho.exceptions import BackendUnavailableError


@pytest.fixture(params=("opencl", "vulkan"))
def gemm_backend(request):
    try:
        return p.get_backend(request.param)
    except BackendUnavailableError as exc:
        if os.environ.get(f"PACHO_REQUIRE_{request.param.upper()}") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


@pytest.mark.parametrize(
    "shapes",
    [
        ((32, 16), (16, 32)),
        ((65, 47), (47, 67)),
        ((2, 1, 33, 65), (1, 3, 65, 39)),
        ((128, 128), (128, 128)),
        ((64, 32), (32, 96)),
        ((2, 1, 64, 32), (1, 3, 32, 64)),
        ((64, 33), (33, 64)),
        ((65, 32), (32, 64)),
    ],
)
def test_register_gemm_accuracy_and_fallback(gemm_backend, shapes, monkeypatch):
    rng = np.random.default_rng(5107)
    host = [rng.uniform(-1, 1, shape).astype(np.float32) for shape in shapes]
    expected = host[0].astype(np.float64) @ host[1].astype(np.float64)
    a, b = [p.asarray(v, backend=gemm_backend) for v in host]
    fused = p.compile(lambda a, b: a @ b)
    first = a @ b
    second = fused(a, b)
    monkeypatch.setattr(gemm_backend, "_register_gemm", False)
    previous = a @ b
    for result in (first, second, previous):
        assert result.dtype == np.dtype("float32") and result.shape == expected.shape
        np.testing.assert_allclose(p.asnumpy(result), expected, rtol=3e-5, atol=3e-5)


def test_register_cache_metadata_only():
    from openpacho._gemm import _source

    _source.cache_clear()
    for n in range(32, 165):
        _source("vulkan", (n, 16), (16, 32))
    assert _source.cache_info().currsize == 128


def test_core_bounds_specialization_preserves_partial_axes():
    from openpacho._gemm import _source

    for name in ("opencl", "vulkan", "cuda"):
        complete, _ = _source(name, (64, 32), (32, 96), specialize_bounds=True)
        assert "if(row+" not in complete and "?x0" not in complete and "?x1" not in complete
        partial, _ = _source(name, (65, 33), (33, 67), specialize_bounds=True)
        guarded, _ = _source(name, (65, 33), (33, 67))
        assert partial == guarded
        k_tail, _ = _source(name, (64, 33), (33, 64), specialize_bounds=True)
        assert "ak<33" in k_tail and "bk<33" in k_tail and "ar<" not in k_tail


@pytest.mark.parametrize("name", ["opencl", "cuda", "vulkan"])
def test_guarded_loads_do_not_promote_float32_to_double(name):
    from openpacho._gemm import _source

    source, _ = _source(name, (65, 33), (33, 67))
    assert "?x0" in source and "?x1" in source
    assert source.count(":0.0f") == 2
    assert not re.search(r"\b0\.0\b(?!f)", source)
