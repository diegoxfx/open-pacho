"""Vector and Frobenius norms (other norm orders deliberately unsupported)."""

import numpy as np

from . import _array as xp


def norm(x, ord=None, axis=None, keepdims=False):
    x = xp.asarray(x)
    if isinstance(axis, tuple) and len(axis) not in (1, 2):
        raise ValueError("norm requires one vector axis or two matrix axes")
    if x.ndim == 0 and axis is not None:
        raise ValueError("axis out of bounds for scalar norm")
    if ord is not None:
        if (
            ord == "fro"
            and (x.ndim == 2 and axis is None or isinstance(axis, tuple) and len(axis) == 2)
            or ord == 2
            and (x.ndim == 1 and axis is None or isinstance(axis, (int, np.integer)))
        ):
            pass
        else:
            raise NotImplementedError(
                "Only Euclidean vector and Frobenius matrix norms are supported"
            )
    if x.dtype.kind != "f":
        x = x.astype(np.float64)
    axes = xp._axes(axis, x.ndim)
    shape = (
        tuple(1 if i in axes else d for i, d in enumerate(x.shape))
        if keepdims
        else tuple(d for i, d in enumerate(x.shape) if i not in axes)
    )
    data = x._backend.norm(x, axes, keepdims, x.dtype)
    return xp.ndarray._wrap(data, shape, x.dtype, x._backend)


xp._FUNCTIONS[np.linalg.norm] = norm
