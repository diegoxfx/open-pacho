"""Resident pseudorandom arrays with explicit backend and counter ownership.

Philox4x32-10 uses one counter block per output element, in C order. Streams
do not depend on launch geometry or call partitioning. NumPy's default_rng
uses a different algorithm: equal seeds do not imply equal NumPy samples.
"""

import operator
import secrets
from math import prod
from numbers import Real
from threading import Lock

import numpy as np

from ._array import _shape, ndarray
from .backends import get_backend

__all__ = ["Generator", "default_rng"]


def _outside_capture():
    if getattr(get_backend(), "_is_tracing", False):
        raise NotImplementedError(
            "Random generation is not captured; pass generated arrays as arguments"
        )


def _scalar(value, dtype, name):
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            value = dtype.type(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and representable in {dtype}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite and representable in {dtype}")
    return value


class Generator:
    """A Philox4x32-10 stream pinned to one backend, with a 64-bit seed.

    Calls on one generator are serialized to reserve disjoint counters. Thread
    scheduling decides which caller receives each range. No cryptographic use.
    """

    def __init__(self, seed=None, *, backend=None):
        _outside_capture()
        if seed is None:
            seed = secrets.randbits(64)
        if isinstance(seed, (bool, np.bool_)):
            raise TypeError("seed must be an integer, not bool")
        seed = operator.index(seed)
        if not 0 <= seed < 2**64:
            raise ValueError("seed must be in [0, 2**64)")
        self._backend = get_backend(backend)
        if getattr(self._backend, "_is_tracing", False):
            raise NotImplementedError("Random generation is not captured")
        self._seed, self._counter = seed, 0
        self._lock = Lock()

    def random(self, size=None, dtype=np.float64):
        """Uniform samples in [0, 1); size=None returns a resident 0-D array."""
        return self._generate("random", size, dtype, 0, 1)

    def uniform(self, low=0.0, high=1.0, size=None, *, dtype=np.float64):
        """Uniform [low, high), or low when equal; scalar bounds in output dtype."""
        return self._generate("uniform", size, dtype, low, high)

    def normal(self, loc=0.0, scale=1.0, size=None, *, dtype=np.float64):
        """Normal samples using Box-Muller; scalar location and nonnegative scale."""
        return self._generate("normal", size, dtype, loc, scale)

    def exponential(self, scale=1.0, size=None, *, dtype=np.float64):
        """Exponential samples with nonnegative scalar scale (the mean)."""
        return self._generate("exponential", size, dtype, 0, scale)

    def _generate(self, distribution, size, dtype, first, second):
        _outside_capture()
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("Random generation requires float32 or float64")
        self._backend.validate_dtype(dtype)
        shape = () if size is None else _shape(size)
        count = prod(shape)
        if count > 2**31 - 1 or any(d > np.iinfo(np.intp).max for d in shape):
            raise ValueError("Random output exceeds indexing limits")
        # Validate domains before rounding: tiny negative scales must not turn
        # into valid -0.0, nor inverted bounds into equal representable bounds.
        if distribution in ("normal", "exponential") and isinstance(second, Real) and second < 0:
            raise ValueError("scale must be nonnegative")
        if (
            distribution == "uniform"
            and isinstance(first, Real)
            and isinstance(second, Real)
            and second < first
        ):
            raise ValueError("uniform requires high >= low")
        first, second = (
            _scalar(first, dtype, "first parameter"),
            _scalar(second, dtype, "second parameter"),
        )
        if distribution == "uniform":
            with np.errstate(over="ignore"):
                width = second - first
            if second < first or not np.isfinite(width):
                raise ValueError("uniform requires high >= low and finite high-low in output dtype")
        elif distribution in ("normal", "exponential") and second < 0:
            raise ValueError("scale must be nonnegative")
        # Resolve support before allocation or stream reservation. GLSL has no
        # double log/cos here; never silently compute in float32 or on CPU.
        generate = getattr(self._backend, "random_data", None)
        if generate is None:
            raise NotImplementedError("This backend does not implement random generation")
        capabilities = getattr(self._backend, "random_distributions", lambda dtype: ())
        if distribution not in capabilities(dtype):
            raise NotImplementedError(
                f"{self._backend.name} does not support {distribution} with {dtype}"
            )
        with self._lock:
            if self._counter + count > 2**64:
                raise OverflowError("Philox stream exhausted; create a generator with a new seed")
            start = self._counter
            # Consume before dispatch: an uncertain asynchronous failure must
            # never lead to reuse of a counter range. Empty calls consume none.
            self._counter += count
            data = generate(shape, dtype, distribution, self._seed, start, first, second)
        return ndarray._wrap(data, shape, dtype, self._backend)


def default_rng(seed=None, *, backend=None):
    """Construct an independent generator; no global random state or GPU import."""
    return Generator(seed, backend=backend)
