"""Residual GEMV preserves the product rounding boundary and resident execution."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

import openpacho as p
from openpacho._array import ndarray
from openpacho._gemv import residual_matvec_data
from openpacho.exceptions import BackendUnavailableError, UnsupportedDtypeError


@pytest.fixture(scope="module", params=("opencl", "vulkan"))
def backend(request):
    try:
        return p.get_backend(request.param)
    except BackendUnavailableError as exc:
        if os.environ.get(f"PACHO_REQUIRE_{request.param.upper()}") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "shape", ((0, 5), (3, 0), (7, 3), (5, 63), (9, 64), (3, 129), (19, 513), (1, 4097))
)
def test_residual_matches_ordinary_backend_rounding(backend, dtype, shape, monkeypatch):
    try:
        backend.validate_dtype(dtype)
    except UnsupportedDtypeError as exc:
        pytest.skip(str(exc))
    rng = np.random.default_rng(814)
    a = p.asarray(rng.uniform(-1, 1, shape).astype(dtype), backend=backend)
    x = p.asarray(rng.uniform(-1, 1, shape[1]).astype(dtype), backend=backend)
    product = p.asnumpy(a @ x)
    # Make subtraction sensitive to any changed matvec rounding.
    host_b = (
        np.nextafter(product, np.full_like(product, np.inf)) if shape[1] else np.ones_like(product)
    )
    b = p.asarray(host_b, backend=backend)
    expected = p.asnumpy(b - a @ x)
    original_run = backend._run
    calls = []

    def run(*args, **kwargs):
        calls.append(1)
        return original_run(*args, **kwargs)

    def forbidden_download(*args):
        pytest.fail("residual helper downloaded an intermediate")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "_run", run)
        patch.setattr(backend, "download", forbidden_download)
        out = residual_matvec_data(backend, a, x, b)
    actual = p.asnumpy(ndarray._wrap(out, b.shape, b.dtype, backend))
    np.testing.assert_array_equal(actual, expected)
    if shape[0] and shape[1] and not (backend.name == "vulkan" and shape == (1, 4097)):
        assert len(calls) == 1
    if not shape[0]:
        assert not calls
    assert out is not b._data


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_residual_nonfinite_and_cancellation(backend, dtype):
    try:
        backend.validate_dtype(dtype)
    except UnsupportedDtypeError as exc:
        pytest.skip(str(exc))
    a = np.zeros((5, 128), dtype=dtype)
    a[0, 0], a[1, 0], a[2, 0] = np.inf, -np.inf, np.nan
    a[3, :4] = np.asarray([1e6, 1, -1e6, 2], dtype=dtype)
    a[4, :2] = [3, -3]
    host_b = np.asarray([np.inf, 1, 2, 3, 0], dtype=dtype)
    a, x, b = [p.asarray(v, backend=backend) for v in (a, np.ones(128, dtype), host_b)]
    expected = p.asnumpy(b - a @ x)
    out = residual_matvec_data(backend, a, x, b)
    np.testing.assert_array_equal(
        p.asnumpy(ndarray._wrap(out, b.shape, b.dtype, backend)), expected
    )


def test_residual_rejects_incompatible_contracts_without_driver():
    backend = SimpleNamespace(name="opencl", _blas=None)

    def value(shape, dtype="float32", owner=backend):
        return SimpleNamespace(shape=shape, ndim=len(shape), dtype=np.dtype(dtype), _backend=owner)

    valid = (value((3, 4)), value((4,)), value((3,)))
    for args in (
        (value((4,)), valid[1], valid[2]),
        (valid[0], value((5,)), valid[2]),
        (valid[0], valid[1], value((3,), "float64")),
        tuple(value(v.shape, "int32") for v in valid),
        (valid[0], valid[1], value((3,), owner=object())),
    ):
        with pytest.raises(ValueError):
            residual_matvec_data(backend, *args)
    backend._blas = object()
    with pytest.raises(NotImplementedError, match="custom"):
        residual_matvec_data(backend, *valid)


def test_compiler_fuses_private_product_but_preserves_shared_outputs(backend):
    a = p.asarray(np.arange(3 * 129, dtype=np.float32).reshape(3, 129) / 100, backend=backend)
    x = p.asarray(np.ones(129, dtype=np.float32), backend=backend)
    b = p.asarray(np.ones(3, dtype=np.float32), backend=backend)
    operation = p.compile(lambda a, x, b: b - a @ x)
    assert [s["kind"] for s in operation.explain(a, x, b)["steps"]] == ["residual_matvec"]
    np.testing.assert_array_equal(p.asnumpy(operation(a, x, b)), p.asnumpy(b - a @ x))

    @p.compile
    def shared(a, x, b):
        product = a @ x
        return product, b - product

    assert "residual_matvec" not in [s["kind"] for s in shared.explain(a, x, b)["steps"]]
    product, residual = shared(a, x, b)
    np.testing.assert_array_equal(p.asnumpy(product), p.asnumpy(a @ x))
    np.testing.assert_array_equal(p.asnumpy(residual), p.asnumpy(b - a @ x))


def test_compiler_keeps_clblast_provider():
    try:
        backend = p.get_backend("opencl", blas="clblast")
    except BackendUnavailableError as exc:
        pytest.skip(str(exc))
    a = p.asarray(np.eye(4, dtype=np.float32), backend=backend)
    x = p.asarray(np.ones(4, dtype=np.float32), backend=backend)
    b = p.asarray(np.full(4, 2, dtype=np.float32), backend=backend)
    operation = p.compile(lambda a, x, b: b - a @ x)
    assert "residual_matvec" not in [s["kind"] for s in operation.explain(a, x, b)["steps"]]
    np.testing.assert_array_equal(p.asnumpy(operation(a, x, b)), np.ones(4, dtype=np.float32))
