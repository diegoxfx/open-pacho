"""Example dense iterative solvers built on OpenPacho's public array API.

These applications belong to the repository examples, not the installed library.
They also serve as fixed workloads for benchmarks and numerical regression tests.

Only scalar convergence diagnostics cross the device boundary during iteration.
CG and steepest descent require symmetric positive-definite matrices. Jacobi
requires a nonzero diagonal and a convergent iteration matrix (strict diagonal
dominance is a sufficient condition). These conditions are not proven by solvers.
"""

import math
import operator
from dataclasses import dataclass

import numpy as np

import openpacho as xp
from openpacho import compile
from openpacho.linalg import norm


@compile
def _jacobi_advance(A, x, b, r, diagonal):
    candidate = x + r / diagonal
    residual = b - A @ candidate
    return candidate, residual, norm(residual)


@compile
def _scaled_advance(A, x, b, numerator, curvature, direction):
    candidate = x + (numerator / curvature) * direction
    residual = b - A @ candidate
    return candidate, residual, norm(residual)


@compile
def _curvature(A, direction):
    return xp.dot(direction, A @ direction)


@compile
def _sd_curvature(A, r, residual_norm):
    direction = r / residual_norm
    return direction, xp.dot(direction, A @ direction)


@compile
def _direction_update(r, new_norm, old_norm, direction):
    return r / new_norm + (new_norm / old_norm) * direction


@compile
def _normalize(r, residual_norm):
    return r / residual_norm


@compile
def _residual_state(A, x, b):
    r = b - A @ x
    return r, norm(r)


def _nonfinite_count(value):
    # Only zero versus nonzero matters, so the input floating dtype suffices
    # even for arrays too large to count every element exactly in float32.
    return xp.sum(xp.equal(xp.isfinite(value), False), dtype=value.dtype)


_compiled_nonfinite_count = compile(_nonfinite_count)


@dataclass(frozen=True)
class SolverResult:
    x: xp.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    residual_history: tuple[float, ...]
    reason: str


def _common_backend(args):
    found = [xp.get_backend(a) for a in args if isinstance(a, xp.ndarray)]
    if found and any(backend is not found[0] for backend in found):
        raise xp.BackendMismatchError(
            "Arrays belong to different backends/devices; use to_backend()"
        )
    return found[0] if found else xp.get_backend()


class PreparedSystem:
    """An owned matrix snapshot with fixed precision and reusable validation.

    Construct through ``solvers.prepare``. Each solve still validates b/x0 and
    checks the true residual at every iteration. Positive definiteness and
    convergence are requirements on the caller, as for the ordinary solvers.
    """

    def __init__(self, A, method):
        if method not in ("cg", "jacobi", "sd"):
            raise ValueError("method must be 'cg', 'jacobi' or 'sd'")
        A = xp.asarray(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or not A.shape[0]:
            raise ValueError("A must be a nonempty square matrix")
        dtype = A.dtype if A.dtype.kind == "f" else np.dtype("float64")
        self._A = xp.array(A, dtype=dtype, copy=True)
        self._method = method
        gpu = A.backend in ("opencl", "vulkan", "cuda")
        count = (_compiled_nonfinite_count if gpu else _nonfinite_count)(self._A)
        if count.item() != 0:
            raise ValueError("A must contain only finite values")
        self._diagonal = xp.diag(self._A) if method == "jacobi" else None
        if self._diagonal is not None and xp.sum(self._diagonal == 0, dtype=dtype).item():
            raise ValueError("Jacobi requires a nonzero diagonal")

    @property
    def shape(self):
        return self._A.shape

    @property
    def dtype(self):
        return self._A.dtype

    @property
    def backend(self):
        return self._A.backend

    @property
    def method(self):
        return self._method

    def solve(self, b, *, x0=None, rtol=None, atol=0.0, maxiter=None):
        return _solve(
            self._method,
            self._A,
            b,
            x0=x0,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            _prepared=self,
        )


def prepare(A, *, method="cg"):
    """Copy and validate A once for repeated solves at A's floating precision.

    Nonfloating matrices use float64. A right-hand side that would promote the
    fixed matrix precision is rejected; prepare an explicitly cast matrix instead.
    """
    return PreparedSystem(A, method)


def _solve(method, A, b, *, x0=None, rtol=None, atol=0.0, maxiter=None, _prepared=None):
    backend = _common_backend((A, b, x0))
    A, b = xp.asarray(A, backend=backend), xp.asarray(b, backend=backend)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not A.shape[0]:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if b.shape != (n,):
        raise ValueError("b must have shape (n,)")
    dtype = np.result_type(A.dtype, b.dtype)
    if dtype.kind != "f":
        dtype = np.dtype("float64")
    if _prepared is not None and dtype != A.dtype:
        raise ValueError("b would change the prepared precision; prepare a cast matrix")
    A, b = A.astype(dtype, copy=False), b.astype(dtype, copy=False)
    x = xp.zeros_like(b) if x0 is None else xp.array(x0, dtype=dtype, backend=backend)
    gpu = backend.name in ("opencl", "vulkan", "cuda")
    if x.shape != (n,):
        raise ValueError("x0 must have shape (n,)")
    inputs = (("A", A), ("b", b), ("x0", x)) if _prepared is None else (("b", b), ("x0", x))
    for name, value in inputs:
        count = (_compiled_nonfinite_count if gpu else _nonfinite_count)(value)
        if count.item() != 0:
            raise ValueError(f"{name} must contain only finite values")
    rtol = (1e-5 if dtype == np.dtype("float32") else 1e-8) if rtol is None else float(rtol)
    atol = float(atol)
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    maxiter = 10 * n if maxiter is None else operator.index(maxiter)
    if maxiter < 0:
        raise ValueError("maxiter must be nonnegative")
    diagonal = (
        _prepared._diagonal if _prepared is not None else xp.diag(A) if method == "jacobi" else None
    )
    if _prepared is None and diagonal is not None and xp.sum(diagonal == 0, dtype=dtype).item():
        raise ValueError("Jacobi requires a nonzero diagonal")
    prepared_calls = {}

    def execute(function, *args):
        call = prepared_calls.get(function)
        if call is None:
            call = prepared_calls[function] = function.prepare(*args)
        return call(*args)

    if gpu:
        r, residual_value = execute(_residual_state, A, x, b)
    else:
        r = b - A @ x
        residual_value = norm(r)
    residual = float(residual_value.item())
    history = [residual]
    tolerance = max(atol, rtol * float(norm(b).item()))

    def result(converged, iterations, reason):
        return SolverResult(x, converged, iterations, history[-1], tuple(history), reason)

    if not math.isfinite(residual) or not math.isfinite(tolerance):
        return result(False, 0, "nonfinite_residual")
    if residual <= tolerance:
        return result(True, 0, "converged")
    # q represents the usual CG direction divided by the current residual norm.
    # This avoids squaring tiny/huge residuals in both alpha and beta. Arithmetic
    # remains in the input dtype; no hidden promotion or relaxed stopping rule.
    q = (
        (execute(_normalize, r, residual_value) if gpu else r / residual)
        if method == "cg"
        else None
    )
    for iteration in range(1, maxiter + 1):
        if method == "jacobi":
            if gpu:
                candidate, candidate_r, new_norm = execute(_jacobi_advance, A, x, b, r, diagonal)
            else:
                candidate = x + r / diagonal
        else:
            if gpu:
                if method == "cg":
                    direction, curvature = q, execute(_curvature, A, q)
                else:
                    direction, curvature = execute(_sd_curvature, A, r, residual_value)
            else:
                direction = q if method == "cg" else r / residual
                curvature = xp.dot(direction, A @ direction)
            if gpu:
                # The host must observe validity before accepting a candidate.
                # Read the existing scalar instead of dispatching a boolean kernel;
                # keep its device value for the typed update and its rounding.
                host_curvature = float(curvature.item())
                if not math.isfinite(host_curvature) or host_curvature <= 0:
                    return result(False, iteration - 1, "nonpositive_curvature")
                candidate, candidate_r, new_norm = execute(
                    _scaled_advance, A, x, b, residual_value, curvature, direction
                )
            else:
                curvature = float(curvature.item())
                if not math.isfinite(curvature) or curvature <= 0:
                    return result(False, iteration - 1, "nonpositive_curvature")
                candidate = x + (residual / curvature) * direction
        if not gpu:
            candidate_r = b - A @ candidate
            new_norm = norm(candidate_r)
        new_residual = float(new_norm.item())
        x, r = candidate, candidate_r
        history.append(new_residual)
        if not math.isfinite(new_residual):
            return result(False, iteration, "nonfinite_residual")
        if new_residual <= tolerance:
            return result(True, iteration, "converged")
        if method == "cg":
            q = (
                execute(_direction_update, r, new_norm, residual_value, q)
                if gpu
                else r / new_residual + (new_residual / residual) * q
            )
        residual_value, residual = new_norm, new_residual
    return result(False, maxiter, "maxiter")


def conjugate_gradient(A, b, *, x0=None, rtol=None, atol=0.0, maxiter=None):
    """Solve SPD A x = b; stop when ||b-Ax|| <= max(atol, rtol*||b||)."""
    return _solve("cg", A, b, x0=x0, rtol=rtol, atol=atol, maxiter=maxiter)


def jacobi(A, b, *, x0=None, rtol=None, atol=0.0, maxiter=None):
    """Solve with Jacobi iteration; return explicit nonconvergence at maxiter."""
    return _solve("jacobi", A, b, x0=x0, rtol=rtol, atol=atol, maxiter=maxiter)


def steepest_descent(A, b, *, x0=None, rtol=None, atol=0.0, maxiter=None):
    """Solve an SPD system with an exact line search on its quadratic energy."""
    return _solve("sd", A, b, x0=x0, rtol=rtol, atol=atol, maxiter=maxiter)
