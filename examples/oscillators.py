"""Coupled harmonic oscillators using public NumPy-compatible operations.

The equation is ``q'' = -K @ q`` with unit masses. Inputs must be finite,
same-backend float32 or float64 arrays; K must be symmetric positive definite.
These are caller preconditions, checked neither by transfers nor device scalar
observations. Only dtype and shape metadata are validated here. The timestep
must be finite; negative timesteps are allowed for time reversal. Stability
requires ``abs(dt) * sqrt(max_eigenvalue(K)) < 2``.

Compile or prepare ``verlet_step`` and pass that callable to ``integrate``.
Keep the Python integration loop outside compilation: compiling the whole loop
would unroll it, rather than create a native device loop. Helpers never create
input arrays, transfer data, or mutate arrays. Energy is an optional resident
scalar diagnostic to evaluate outside the integration loop.
"""

import operator

import numpy as np


def _validate(K, q, v, a=None):
    """Validate metadata without observing array values or dynamic scalars."""
    arrays = (K, q, v) if a is None else (K, q, v, a)
    if any(x.dtype not in (np.dtype("float32"), np.dtype("float64")) for x in arrays):
        raise TypeError("oscillator inputs require float32 or float64 arrays")
    if any(x.dtype != K.dtype for x in arrays):
        raise TypeError("oscillator inputs must share one dtype")
    if K.ndim != 2 or K.shape[0] <= 0 or K.shape[0] != K.shape[1]:
        raise ValueError("expected a nonempty square stiffness matrix K")
    if any(x.ndim != 1 or x.shape != (K.shape[0],) for x in arrays[1:]):
        raise ValueError("expected q, v and a vectors matching K")


def verlet_step(K, q, v, a, dt):
    """Return new ``(q, v, a)`` using one velocity-Verlet step and one GEMV.

    The caller carries acceleration between steps and initially supplies
    ``a = -(K @ q)``. Every returned state is new; inputs are unchanged. ``dt``
    can be a Python scalar or a captured dynamic scalar.
    """
    _validate(K, q, v, a)
    vhalf = v + (0.5 * dt) * a
    q1 = q + dt * vhalf
    a1 = -(K @ q1)
    v1 = vhalf + (0.5 * dt) * a1
    return q1, v1, a1


def integrate(K, q, v, dt, steps, *, step=verlet_step):
    """Return final ``(q, v, a)`` after a nonnegative integer number of steps.

    Initialize acceleration exactly once, then carry it through the Python
    loop. With the default step, integration performs exactly ``steps + 1``
    matrix-vector products and no host observations. Zero steps returns the
    original q and v with their computed acceleration. ``step`` may be an eager,
    compiled or prepared ``verlet_step`` callable with the same signature.
    """
    _validate(K, q, v)
    if isinstance(steps, (bool, np.bool_)):
        raise TypeError("steps must be a nonnegative integer")
    try:
        steps = operator.index(steps)
    except TypeError as exc:
        raise TypeError("steps must be a nonnegative integer") from exc
    if steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    a = -(K @ q)
    for _ in range(steps):
        q, v, a = step(K, q, v, a, dt)
    return q, v, a


def energy(K, q, v):
    """Return resident scalar energy ``(v dot v + q dot Kq) / 2``."""
    _validate(K, q, v)
    return 0.5 * ((v * v).sum() + (q * (K @ q)).sum())
