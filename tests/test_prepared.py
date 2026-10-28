"""Prepared execution preserves binding, ownership and solver checks."""

import gc
import os
import weakref

import numpy as np
import pytest

import openpacho as p
from examples import solvers
from openpacho.exceptions import BackendMismatchError, BackendUnavailableError


@pytest.fixture(params=("numpy", "opencl", "vulkan"))
def prepared_backend(request):
    try:
        return p.backends.get_backend(request.param)
    except BackendUnavailableError as exc:
        if os.environ.get(f"PACHO_REQUIRE_{request.param.upper()}") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def test_prepared_rebinds_and_does_not_retain_examples(prepared_backend):
    @p.compile
    def affine(a, factor=2.0):
        return a * factor + 1

    a = p.asarray(np.arange(12, dtype=np.float32), backend=prepared_backend)
    ref = weakref.ref(a)
    run = affine.prepare(a)
    del a
    gc.collect()
    assert ref() is None
    b = p.ones(12, dtype=np.float32, backend=prepared_backend)
    first = run(b, 3.0)
    second = run(a=b, factor=7.0)
    affine.clear_cache()
    third = run(b)
    for result, expected in ((first, 4), (second, 8), (third, 3)):
        np.testing.assert_array_equal(p.asnumpy(result), np.full(12, expected, np.float32))
    assert affine.cache_info()["size"] == 0
    assert run.explain()["binding_validation"]


def test_prepared_rejects_specialization_changes(prepared_backend):
    @p.compile(static_argnames="axis")
    def total(a, factor=2.0, *, axis=0):
        return p.sum(a * factor, axis=axis)

    a = p.ones((3, 4), dtype=np.float32, backend=prepared_backend)
    run = total.prepare(a)
    np.testing.assert_array_equal(p.asnumpy(run(a)), np.full(4, 6, np.float32))
    for args, kwargs in (
        ((p.ones((4, 3), dtype=np.float32, backend=prepared_backend),), {}),
        ((a.astype(np.float64),), {}),
        ((np.ones((3, 4), np.float32),), {}),
        ((a, 2), {}),
        ((a,), {"axis": 1}),
    ):
        with pytest.raises(ValueError, match="specialization"):
            run(*args, **kwargs)
    with pytest.raises(TypeError):
        run()
    with pytest.raises(TypeError):
        run(a, 2.0, 3)


@pytest.mark.parametrize(
    "method,name", [("cg", "conjugate_gradient"), ("jacobi", "jacobi"), ("sd", "steepest_descent")]
)
def test_prepared_system_matches_solver(prepared_backend, method, name):
    host = np.diag(np.arange(2, 10, dtype=np.float32)) + np.float32(0.03)
    A = p.asarray(host, backend=prepared_backend)
    system = solvers.prepare(A, method=method)
    assert (system.shape, system.dtype, system.backend, system.method) == (
        A.shape,
        A.dtype,
        A.backend,
        method,
    )
    for scale in (1, 3):
        b = p.asarray(np.arange(8, dtype=np.float32) * scale, backend=prepared_backend)
        expected = getattr(solvers, name)(A, b, rtol=1e-5, maxiter=100)
        actual = system.solve(b, rtol=1e-5, maxiter=100)
        assert (actual.converged, actual.iterations, actual.reason) == (
            expected.converged,
            expected.iterations,
            expected.reason,
        )
        np.testing.assert_allclose(
            actual.residual_history, expected.residual_history, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            p.asnumpy(actual.x), np.linalg.solve(host, p.asnumpy(b)), rtol=5e-4, atol=5e-4
        )
    assert system.solve(b, maxiter=0).iterations == 0
    with pytest.raises(ValueError, match="finite"):
        system.solve(p.full(8, np.nan, dtype=np.float32, backend=prepared_backend))
    with pytest.raises(ValueError, match="finite"):
        system.solve(b, x0=p.full(8, np.inf, dtype=np.float32, backend=prepared_backend))
    with pytest.raises(ValueError, match="precision"):
        system.solve(b.astype(np.float64))
    with pytest.raises(ValueError, match="shape"):
        system.solve(b[:2])
    with pytest.raises(ValueError, match="rtol"):
        system.solve(b, rtol=-1)


def test_prepared_system_owns_matrix_snapshot():
    A = p.asarray(np.eye(3), backend="numpy")
    system = solvers.prepare(A, method="jacobi")
    A._data[:] = 0  # Simulate mutation by a native owner of the source buffer.
    np.testing.assert_array_equal(p.asnumpy(system.solve(np.ones(3)).x), np.ones(3))
    with pytest.raises(AttributeError):
        system.dtype = np.dtype("float32")


def test_prepared_system_validation(prepared_backend):
    for host, error in (
        (np.ones((2, 3)), "square"),
        (np.empty((0, 0)), "square"),
        (np.diag([1.0, np.nan]), "finite"),
    ):
        with pytest.raises(ValueError, match=error):
            solvers.prepare(p.asarray(host, backend=prepared_backend))
    with pytest.raises(ValueError, match="diagonal"):
        solvers.prepare(p.zeros((2, 2), backend=prepared_backend), method="jacobi")
    with pytest.raises(ValueError, match="method"):
        solvers.prepare(p.eye(2, backend=prepared_backend), method="invalid")
    system = solvers.prepare(p.eye(2, backend=prepared_backend))
    if prepared_backend.name != "numpy":
        with pytest.raises(BackendMismatchError):
            system.solve(p.ones(2, backend="numpy"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("tile", [None, 8, 0])
@pytest.mark.parametrize(
    "shapes", [((17, 13), (13, 19)), ((2, 1, 9, 15), (1, 3, 15, 11)), ((8, 8), (8, 8))]
)
def test_tiled_opencl_gemm(opencl_backend, dtype, shapes, tile, monkeypatch):
    if tile is not None:
        if tile > opencl_backend._gemm_tile:
            pytest.skip("Device cannot execute this tile size")
        monkeypatch.setattr(opencl_backend, "_gemm_tile", tile)
    rng = np.random.default_rng(711)
    a, b = (rng.integers(-3, 4, size=shape).astype(dtype) for shape in shapes)
    actual = p.asarray(a, backend=opencl_backend) @ p.asarray(b, backend=opencl_backend)
    np.testing.assert_array_equal(p.asnumpy(actual), a @ b)


def test_prepared_system_reuses_only_matrix_validation(monkeypatch):
    A = p.eye(5, backend="numpy")
    original = solvers.xp.isfinite
    checked = []

    def track(value):
        checked.append(value.shape)
        return original(value)

    monkeypatch.setattr(solvers.xp, "isfinite", track)
    system = solvers.prepare(A, method="jacobi")
    assert checked.count((5, 5)) == 1
    for _ in range(2):
        assert system.solve(p.ones(5, backend="numpy")).converged
    assert checked.count((5, 5)) == 1
    assert checked.count((5,)) == 4  # b and x0 remain validated on every solve.
