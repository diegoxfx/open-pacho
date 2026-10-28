"""Norms must not square extreme inputs before scaling."""

import math
import os

import numpy as np
import pytest

import openpacho as p
from examples import solvers
from openpacho.exceptions import BackendUnavailableError


@pytest.fixture(params=("numpy", "opencl", "vulkan", "cuda-custom"))
def norm_backend(request):
    name = "cuda" if request.param == "cuda-custom" else request.param
    options = (
        {"blas": "custom", "device": os.environ.get("PACHO_CUDA_DEVICE") or "0"}
        if name == "cuda"
        else {}
    )
    try:
        return p.get_backend(name, **options)
    except BackendUnavailableError as exc:
        if os.environ.get(f"PACHO_REQUIRE_{name.upper()}") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


@pytest.mark.parametrize(
    "dtype,small,large", [(np.float32, 1e-25, 1e25), (np.float64, 1e-200, 1e200)]
)
@pytest.mark.parametrize(
    "shape,axis", [((2,), None), ((3, 5, 7), (2, 0)), ((4099,), None), ((2, 0, 5), 1), ((), None)]
)
def test_scaled_norm(norm_backend, dtype, small, large, shape, axis):
    if dtype == np.float64 and not getattr(norm_backend, "fp64", True):
        pytest.skip("selected device does not support float64")
    for scale in (small, large, 0.0):
        host = np.full(shape, scale, dtype=dtype)
        a = p.asarray(host, backend=norm_backend)
        axes = (
            tuple(range(len(shape))) if axis is None else (axis,) if isinstance(axis, int) else axis
        )
        count = math.prod(shape[i] for i in axes)
        outshape = tuple(1 if i in axes else d for i, d in enumerate(shape))
        expected = np.full(
            outshape, abs(float(np.asarray(scale, dtype=dtype))) * math.sqrt(count), dtype=dtype
        )

        @p.compile
        def calculate(x):
            return p.linalg.norm(x, axis=axis, keepdims=True)

        for result in (p.linalg.norm(a, axis=axis, keepdims=True), calculate(a)):
            assert result.dtype == np.dtype(dtype) and result.shape == expected.shape
            np.testing.assert_allclose(p.asnumpy(result), expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize(
    "values,expected",
    [([0.0, 0.0], 0.0), ([np.inf, 1.0], np.inf), ([np.nan, np.inf], np.nan), ([-3.0, 4.0], 5.0)],
)
def test_norm_special(norm_backend, values, expected):
    actual = p.linalg.norm(p.asarray(values, backend=norm_backend)).item()
    assert math.isnan(actual) if math.isnan(expected) else actual == expected


@pytest.mark.parametrize("method", ["conjugate_gradient", "jacobi", "steepest_descent"])
@pytest.mark.parametrize(
    "dtype,scale",
    [(np.float32, 1e-25), (np.float32, 1e25), (np.float64, 1e-200), (np.float64, 1e200)],
)
@pytest.mark.parametrize("scale_matrix", [False, True])
def test_solver_extreme_scaling(norm_backend, method, dtype, scale, scale_matrix):
    if dtype == np.float64 and not getattr(norm_backend, "fp64", True):
        pytest.skip("selected device does not support float64")
    host = np.diag(np.arange(2, 6, dtype=dtype))
    exact = np.arange(1, 5, dtype=dtype)
    if scale_matrix:
        host *= dtype(scale)
    else:
        exact *= dtype(scale)
    rhs = host @ exact
    A, b = (p.asarray(v, backend=norm_backend) for v in (host, rhs))
    rtol = 2e-5 if dtype == np.float32 else 1e-10
    result = getattr(solvers, method)(A, b, rtol=rtol, maxiter=200)
    assert result.converged and result.iterations > 0, result.reason
    actual = p.asnumpy(result.x)
    np.testing.assert_allclose(
        actual / np.max(abs(exact)), exact / np.max(abs(exact)), rtol=rtol * 5, atol=rtol * 5
    )
    assert 0 < result.residual_history[0] < math.inf
