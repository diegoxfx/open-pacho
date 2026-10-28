"""Shared elementwise operation contract for eager execution and graph capture."""

import numpy as np

UFUNCS = {
    name: getattr(np, name)
    for name in (
        "add",
        "subtract",
        "multiply",
        "true_divide",
        "power",
        "negative",
        "positive",
        "absolute",
        "sqrt",
        "exp",
        "log",
        "sin",
        "cos",
        "tanh",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "isfinite",
    )
}


class HostScalar:
    _host_scalar = True
    shape = ()

    def __init__(self, value, dtype):
        self.dtype = np.dtype(dtype)
        self._data = np.asarray(value, dtype=self.dtype)[()]


def resolve_elementwise(ufunc, inputs):
    if ufunc is np.where:
        branches = []
        for value in inputs[1:]:
            if getattr(value, "weak_type", None) is not None:
                branches.append(value.sample)
            elif type(value) in (bool, int, float, complex):
                branches.append(value)
            else:
                branches.append(value.dtype if hasattr(value, "dtype") else np.asarray(value).dtype)
        dtype = np.result_type(*branches)
        shape = np.broadcast_shapes(
            *(value.shape if hasattr(value, "shape") else np.shape(value) for value in inputs)
        )
        return (np.dtype("bool"), dtype, dtype, dtype), shape
    if ufunc not in UFUNCS.values():
        raise NotImplementedError(f"Unsupported elementwise operation {ufunc.__name__}")
    types = tuple(
        getattr(a, "weak_type", None)
        or (
            type(a)
            if type(a) in (int, float, complex)
            else a.dtype
            if hasattr(a, "dtype")
            else np.asarray(a).dtype
        )
        for a in inputs
    )
    resolved = ufunc.resolve_dtypes(types + (None,))
    if ufunc is np.power and resolved[-1].kind in "iu":
        raise NotImplementedError(
            "Integer power is not supported; cast to float or multiply explicitly"
        )
    shape = np.broadcast_shapes(*(getattr(a, "shape", ()) for a in inputs))
    return resolved, shape
