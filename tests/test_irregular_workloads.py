"""Rectangular, odd and non-power-of-two shapes across providers and plans."""

import os

import numpy as np
import pytest

import openpacho as p
from examples import solvers
from openpacho.exceptions import BackendUnavailableError, UnsupportedDtypeError


@pytest.fixture(scope="module", params=("numpy", "opencl", "opencl-clblast", "vulkan"))
def irregular_backend(request):
    name = request.param
    try:
        return (
            p.get_backend("opencl", blas="clblast")
            if name == "opencl-clblast"
            else p.get_backend(name)
        )
    except BackendUnavailableError as exc:
        required = (
            bool(os.environ.get("OPENPACHO_CLBLAST_LIBRARY"))
            if name == "opencl-clblast"
            else os.environ.get(f"PACHO_REQUIRE_{name.upper()}") == "1"
        )
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))


@pytest.fixture(params=(np.float32, np.float64))
def dtype(irregular_backend, request):
    try:
        return irregular_backend.validate_dtype(request.param)
    except UnsupportedDtypeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize(
    "shapes",
    [
        ((31, 15), (15, 33)),
        ((33, 17), (17, 65)),
        ((65, 63), (63, 97)),
        ((127, 65), (65, 35)),
        ((45, 30), (30, 70)),
        ((3, 1, 33, 17), (1, 5, 17, 35)),
    ],
)
def test_rectangular_tile_boundaries_eager_compiled_and_rebound(irregular_backend, dtype, shapes):
    rng = np.random.default_rng(831)
    host = [rng.uniform(-1, 1, shape).astype(dtype) for shape in shapes]
    a, b = [p.asarray(x, backend=irregular_backend) for x in host]
    operation = p.compile(lambda left, right: left @ right)
    expected = host[0].astype(np.float64) @ host[1].astype(np.float64)
    tolerance = 3e-5 if dtype == np.dtype("float32") else 5e-13
    for result in (a @ b, operation(a, b)):
        assert result.shape == expected.shape and result.dtype == dtype
        np.testing.assert_allclose(p.asnumpy(result), expected, rtol=tolerance, atol=tolerance)
    # Reuse the same plan with different values, while a prior public result lives.
    later = operation(-a, b)
    np.testing.assert_allclose(p.asnumpy(later), -expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(p.asnumpy(result), expected, rtol=tolerance, atol=tolerance)
    assert operation.cache_info()["misses"] == 1
    np.testing.assert_array_equal(p.asnumpy(a), host[0])
    np.testing.assert_array_equal(p.asnumpy(b), host[1])


@pytest.mark.parametrize("shape", ((31, 63), (65, 129), (127, 257)))
def test_rectangular_gemv_residual_and_axis_reductions(irregular_backend, dtype, shape):
    rng = np.random.default_rng(832)
    ah = rng.uniform(-1, 1, shape).astype(dtype)
    xh = rng.uniform(-1, 1, shape[1]).astype(dtype)
    bh = rng.uniform(-1, 1, shape[0]).astype(dtype)
    a, x, b = [p.asarray(v, backend=irregular_backend) for v in (ah, xh, bh)]

    @p.compile
    def operation(a, x, b):
        residual = b - a @ x
        return residual, np.sum(a * a, axis=0), np.sum(a * a, axis=1)

    actual = operation(a, x, b)
    reference = (
        bh.astype(float) - ah.astype(float) @ xh.astype(float),
        np.sum(ah.astype(float) ** 2, axis=0),
        np.sum(ah.astype(float) ** 2, axis=1),
    )
    tolerance = 3e-5 if dtype == np.dtype("float32") else 5e-13
    for result, expected in zip(actual, reference):
        assert result.shape == expected.shape and result.dtype == dtype
        np.testing.assert_allclose(p.asnumpy(result), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("n", (31, 65, 129))
@pytest.mark.parametrize("method", ("jacobi", "conjugate_gradient", "steepest_descent"))
def test_odd_dense_spd_solvers_true_residual_and_preparation(irregular_backend, dtype, n, method):
    rng = np.random.default_rng(833)
    u = rng.uniform(-0.1, 0.1, n)
    ah = (np.diag(np.linspace(2, 4, n)) + np.outer(u, u)).astype(dtype)
    exact = rng.uniform(-1, 1, n).astype(dtype)
    bh = ah @ exact
    a, b = [p.asarray(v, backend=irregular_backend) for v in (ah, bh)]
    rtol = 1e-5 if dtype == np.dtype("float32") else 1e-10
    ordinary = getattr(solvers, method)(a, b, rtol=rtol, maxiter=250)
    selected = {"conjugate_gradient": "cg", "steepest_descent": "sd"}.get(method, method)
    prepared = solvers.prepare(a, method=selected).solve(b, rtol=rtol, maxiter=250)
    reference = np.linalg.solve(ah.astype(float), bh.astype(float))
    for result in (ordinary, prepared):
        assert result.converged and result.iterations > 0
        assert len(result.residual_history) == result.iterations + 1
        actual = p.asnumpy(result.x)
        independent = np.linalg.norm(bh.astype(float) - ah.astype(float) @ actual.astype(float))
        assert independent <= 1.02 * rtol * np.linalg.norm(bh.astype(float))
        np.testing.assert_allclose(actual, reference, rtol=rtol * 5, atol=rtol * 5)
    assert ordinary.iterations == prepared.iterations
    np.testing.assert_array_equal(p.asnumpy(a), ah)
    np.testing.assert_array_equal(p.asnumpy(b), bh)
