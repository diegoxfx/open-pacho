"""Numerical and execution contracts for the external oscillator client."""

import gc
import os
import weakref

import numpy as np
import pytest

import openpacho as p
from examples.oscillators import energy, integrate, verlet_step
from openpacho.exceptions import BackendUnavailableError


def _inputs(dtype=np.float64, variant=0):
    # Coupling makes coordinate-wise independent oscillator updates incorrect.
    K = np.asarray([[3.0, -0.6, 0.2], [-0.6, 2.0, -0.4], [0.2, -0.4, 1.0]], dtype=dtype)
    q = np.asarray([0.7, -0.3, 0.2], dtype=dtype) + dtype(variant * 0.15)
    v = np.asarray([-0.2, 0.5, 0.1], dtype=dtype) - dtype(variant * 0.08)
    return K, q, v


def _modal_solution(K, q, v, time):
    # Diagonalize the actual rounded inputs, independently of Verlet's updates.
    eigenvalues, modes = np.linalg.eigh(K.astype(np.float64))
    omega = np.sqrt(eigenvalues)
    position, velocity = modes.T @ q.astype(np.float64), modes.T @ v.astype(np.float64)
    cosine, sine = np.cos(omega * time), np.sin(omega * time)
    return (
        modes @ (position * cosine + velocity * sine / omega),
        modes @ (-omega * position * sine + velocity * cosine),
    )


def test_hand_computed_scalar_step_and_immutability():
    values = tuple(np.asarray(x) for x in ([[4.0]], [1.0], [2.0], [-4.0]))
    originals = tuple(x.copy() for x in values)
    for value in values:
        value.flags.writeable = False
    actual = verlet_step(*values, 0.1)
    for value, expected in zip(actual, ([1.18], [1.564], [-4.72])):
        np.testing.assert_allclose(value, expected, rtol=1e-15)
        assert all(not np.shares_memory(value, original) for original in values)
    for value, original in zip(values, originals):
        np.testing.assert_array_equal(value, original)
    np.testing.assert_equal(energy(*values[:3]), 4.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_coupled_modal_solution_and_second_order_convergence(dtype):
    K, q, v = _inputs(dtype)
    exact = _modal_solution(K, q, v, 1.6)
    errors = []
    for dt, steps in ((0.08, 20), (0.04, 40), (0.02, 80)):
        q1, v1, a1 = integrate(K, q, v, dt, steps)
        errors.append(np.linalg.norm(np.concatenate((q1 - exact[0], v1 - exact[1]))))
        np.testing.assert_allclose(a1, -(K @ q1), rtol=0, atol=0)
    # All three errors are above the float32 floor; halving dt quarters error.
    assert errors[-1] > 5e-5
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2
    np.testing.assert_allclose(q1, exact[0], rtol=0, atol=3e-4)
    np.testing.assert_allclose(v1, exact[1], rtol=0, atol=3e-4)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_bounded_energy_excursion_and_time_reversal(dtype):
    K, q, v = _inputs(dtype)
    a = -(K @ q)
    initial_energy = energy(K, q, v)
    positions, velocities = q, v
    energies = [initial_energy]
    for _ in range(1000):
        positions, velocities, a = verlet_step(K, positions, velocities, a, 0.04)
        energies.append(energy(K, positions, velocities))
    assert max(abs(x / initial_energy - 1) for x in energies) < 0.002
    for _ in range(1000):
        positions, velocities, a = verlet_step(K, positions, velocities, a, -0.04)
    tolerance = 1e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(positions, q, rtol=0, atol=tolerance)
    np.testing.assert_allclose(velocities, v, rtol=0, atol=tolerance)


def test_carried_acceleration_and_exact_gemv_count():
    class CountingMatrix(np.ndarray):
        def __matmul__(self, other):
            products.append(other.copy())
            return np.asarray(self) @ other

    K, q, v = _inputs()
    K = K.view(CountingMatrix)
    products = []
    calls = []

    def checked_step(K, q, v, a, dt):
        np.testing.assert_array_equal(a, -(np.asarray(K) @ q))
        calls.append(a)
        return verlet_step(K, q, v, a, dt)

    integrate(K, q, v, 0.03, 7, step=checked_step)
    assert len(products) == 8
    assert len(calls) == 7
    np.testing.assert_array_equal(products[0], q)
    products.clear()
    q0, v0, a0 = integrate(K, q, v, 0.03, np.int64(0), step=checked_step)
    assert q0 is q and v0 is v
    assert len(products) == 1 and len(calls) == 7
    np.testing.assert_array_equal(a0, -(np.asarray(K) @ q))


@pytest.mark.parametrize("steps", [True, np.bool_(False), 1.5, 2.0, -1])
def test_invalid_step_counts(steps):
    with pytest.raises((TypeError, ValueError), match="nonnegative integer"):
        integrate(*_inputs(), 0.01, steps)


@pytest.mark.parametrize(
    "invalid", ["integer", "mixed", "empty", "nonsquare", "matrix-q", "short-v", "short-a"]
)
def test_metadata_validation(invalid):
    K, q, v = _inputs()
    a = -(K @ q)
    if invalid == "integer":
        K = K.astype(np.int64)
    elif invalid == "mixed":
        v = v.astype(np.float32)
    elif invalid == "empty":
        K, q, v, a = K[:0, :0], q[:0], v[:0], a[:0]
    elif invalid == "nonsquare":
        K = K[:, :-1]
    elif invalid == "matrix-q":
        q = q[None, :]
    elif invalid == "short-v":
        v = v[:-1]
    else:
        a = a[:-1]
    with pytest.raises((TypeError, ValueError)):
        verlet_step(K, q, v, a, 0.01)
    if invalid != "short-a":
        for function in (energy, lambda K, q, v: integrate(K, q, v, 0.01, 1)):
            with pytest.raises((TypeError, ValueError)):
                function(K, q, v)


def _assert_outputs(actual, expected, dtype):
    tolerance = 2e-5 if dtype == np.float32 else 2e-12
    for value, reference in zip(actual, expected):
        host = p.asnumpy(value)
        assert host.dtype == np.dtype(dtype)
        assert host.shape == reference.shape
        np.testing.assert_allclose(host, reference, rtol=tolerance, atol=tolerance)


def _check_execution(backend, dtype):
    compiled = p.compile(verlet_step)
    prepared = None
    for variant, dt in ((0, 0.025), (1, -0.04)):
        inputs = _inputs(dtype, variant)
        originals = tuple(x.copy() for x in inputs)
        resident = tuple(p.asarray(x, backend=backend) for x in inputs)
        K, q, v = resident
        if prepared is None:
            prepared = compiled.prepare(K, q, v, -(K @ q), dt)
        expected = integrate(*inputs, dt, 24)
        for step in (verlet_step, compiled, prepared):
            actual = integrate(*resident, dt, 24, step=step)
            _assert_outputs(actual, expected, dtype)
        _assert_outputs((energy(*resident),), (energy(*inputs),), dtype)
        for value, original in zip(resident, originals):
            np.testing.assert_array_equal(p.asnumpy(value), original)
    assert compiled.cache_info()["size"] == 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_numpy_eager_compiled_prepared_integration(dtype):
    _check_execution(p.get_backend("numpy"), dtype)


def test_prepared_rebinding_nonretention_and_independent_outputs():
    compiled = p.compile(verlet_step)
    K, q, v = tuple(p.asarray(x, backend="numpy") for x in _inputs())
    a = -(K @ q)
    examples = tuple(weakref.ref(x) for x in (K, q, v, a))
    prepared = compiled.prepare(K, q, v, a, 0.03)
    del K, q, v, a
    gc.collect()
    assert all(reference() is None for reference in examples)
    states = []
    snapshots = []
    for variant, dt in ((1, 0.03), (2, -0.07)):
        host = _inputs(variant=variant)
        resident = tuple(p.asarray(x, backend="numpy") for x in host)
        state = integrate(*resident, dt, 12, step=prepared)
        _assert_outputs(state, integrate(*host, dt, 12), np.float64)
        states.append(state)
        snapshots.append(tuple(p.asnumpy(x).copy() for x in state))
    for state, snapshot in zip(states, snapshots):
        _assert_outputs(state, snapshot, np.float64)
    for first, second in zip(*states):
        assert not np.shares_memory(p.asnumpy(first), p.asnumpy(second))
    assert compiled.cache_info()["size"] == 1


@pytest.fixture(params=["opencl", "vulkan", "cuda-custom", "cuda-cublas"])
def oscillators_native_backend(request):
    selected = request.param
    name = "cuda" if selected.startswith("cuda-") else selected
    options = {"blas": selected.split("-", 1)[1]} if name == "cuda" else {}
    if name == "opencl":
        options["blas"] = "custom"
    device = os.environ.get(f"PACHO_{name.upper()}_DEVICE")
    if device:
        options["device"] = device
    try:
        return p.get_backend(name, **options)
    except (BackendUnavailableError, ImportError) as exc:
        if os.environ.get(f"PACHO_REQUIRE_{name.upper()}") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_native_eager_compiled_prepared_integration(oscillators_native_backend, dtype):
    backend = oscillators_native_backend
    if dtype == np.float64 and not getattr(backend, "fp64", True):
        pytest.skip("selected device does not support float64")
    _check_execution(backend, dtype)
