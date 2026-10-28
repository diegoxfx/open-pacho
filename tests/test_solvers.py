import numpy as np
import pytest

import openpacho as p
from examples import solvers


@pytest.mark.parametrize(
    "method", [solvers.conjugate_gradient, solvers.jacobi, solvers.steepest_descent]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_known_spd_solution(backend, method, dtype):
    rng = np.random.default_rng(47)
    raw = rng.uniform(-0.1, 0.1, (9, 9))
    A = (raw + raw.T + 4 * np.eye(9)).astype(dtype)
    exact = np.linspace(-1, 2, 9).astype(dtype)
    b = A @ exact
    da, db = p.asarray(A, backend=backend), p.asarray(b, backend=backend)
    tolerance = 2e-5 if dtype == np.float32 else 1e-10
    result = method(da, db, rtol=tolerance, maxiter=300)
    assert result.converged
    assert result.reason == "converged"
    assert result.x.backend == backend.name
    assert len(result.residual_history) == result.iterations + 1
    actual = p.asnumpy(result.x)
    np.testing.assert_allclose(
        actual, np.linalg.solve(A, b), rtol=tolerance * 5, atol=tolerance * 5
    )
    assert np.linalg.norm(b - A @ actual) <= tolerance * np.linalg.norm(b) * 1.1
    np.testing.assert_allclose(p.asnumpy(da), A)
    np.testing.assert_allclose(p.asnumpy(db), b)


@pytest.mark.parametrize(
    "method", [solvers.conjugate_gradient, solvers.jacobi, solvers.steepest_descent]
)
def test_zero_rhs_and_iteration_limit(backend, method):
    A = p.asarray([[4.0, 1.0], [1.0, 3.0]], backend=backend)
    zero = p.zeros(2, backend=backend)
    result = method(A, zero)
    assert result.converged and result.iterations == 0 and result.residual_norm == 0
    b = p.ones(2, backend=backend)
    result = method(A, b, maxiter=0)
    assert not result.converged and result.iterations == 0 and result.reason == "maxiter"
    result = method(A, b, maxiter=1, rtol=0, atol=0)
    assert not result.converged and result.iterations == 1


def test_exact_initial_guess_and_zero_rhs_nonzero_guess(backend):
    A = p.eye(3, backend=backend)
    b = p.asarray([1.0, 2.0, 3.0], backend=backend)
    result = solvers.conjugate_gradient(A, b, x0=b)
    assert result.converged and result.iterations == 0
    result = solvers.conjugate_gradient(A, p.zeros_like(b), x0=b, atol=1e-12)
    assert result.converged and result.iterations == 1
    np.testing.assert_array_equal(p.asnumpy(b), [1.0, 2.0, 3.0])


@pytest.mark.parametrize("method", [solvers.conjugate_gradient, solvers.steepest_descent])
def test_nonpositive_curvature(backend, method):
    A = p.asarray([[-1.0, 0.0], [0.0, -2.0]], backend=backend)
    result = method(A, p.ones(2, backend=backend))
    assert not result.converged and result.reason == "nonpositive_curvature"


def test_jacobi_zero_diagonal(backend):
    with pytest.raises(ValueError, match="diagonal"):
        solvers.jacobi(
            p.asarray([[0.0, 1.0], [1.0, 2.0]], backend=backend), p.ones(2, backend=backend)
        )


@pytest.mark.parametrize(
    "kwargs", [{"rtol": -1}, {"atol": float("nan")}, {"maxiter": -1}, {"rtol": float("inf")}]
)
def test_bad_tolerances(kwargs):
    with pytest.raises(ValueError):
        solvers.conjugate_gradient([[2.0, 0.0], [0.0, 3.0]], [1.0, 1.0], **kwargs)


@pytest.mark.parametrize(
    "A,b",
    [
        ([[1.0, 2.0]], [1.0]),
        ([[1.0, 0.0], [0.0, 1.0]], [[1.0], [2.0]]),
        ([[float("nan")]], [1.0]),
        ([[1.0]], [float("inf")]),
    ],
)
def test_invalid_system(A, b):
    with pytest.raises(ValueError):
        solvers.conjugate_gradient(A, b)
