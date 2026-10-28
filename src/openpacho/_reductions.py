"""Scalar reduction algebra shared by eager and compiled GPU kernels.

Extrema propagate NaNs. Zero ties are associative: min prefers negative zero,
max prefers positive zero, while a single sign is preserved. Integer comparisons
and neutral values stay in their storage type, including 64-bit limits.
"""

import numpy as np


def _cast(value, ctype, dialect):
    return f"{ctype}({value})" if dialect == "vulkan" else f"(({ctype})({value}))"


def identity(operation, dtype, ctype, dialect):
    """Typed neutral expression for sum/min/max; never a float surrogate for int."""
    dtype = np.dtype(dtype)
    cast = lambda value: _cast(value, ctype, dialect)
    if operation == "sum":
        return cast("0")
    if operation not in ("min", "max"):
        raise ValueError(f"Unknown reduction operation: {operation}")
    if dtype.kind == "b":
        return cast("1" if operation == "min" else "0")
    if dtype.kind == "f":
        infinity = (
            "uintBitsToFloat(0x7f800000u)"
            if dialect == "vulkan"
            else "__int_as_float(0x7f800000)"
            if dialect == "cuda"
            else "INFINITY"
        )
        return cast(infinity if operation == "min" else f"(-{infinity})")
    if dtype.kind == "u":
        return f"(~{cast('0')})" if operation == "min" else cast("0")
    # Construct the int64 lower bound from int32-sized constants; no oversized
    # unsuffixed literal, float conversion, negation of INT_MIN or signed shift.
    lower = cast("-2147483647-1")
    if dtype.itemsize == 8:
        lower = f"({lower}*{cast('65536')}*{cast('65536')})"
    return f"(-({lower}+{cast('1')}))" if operation == "min" else lower


def combine(operation, dtype, ctype, dialect, left, right):
    """Typed scalar combine expression, valid for both lane and tree merges."""
    dtype = np.dtype(dtype)
    if operation not in ("sum", "min", "max"):
        raise ValueError(f"Unknown reduction operation: {operation}")
    left, right = f"({left})", f"({right})"
    if dtype.kind == "b":
        operator = "&&" if operation == "min" else "||"
        return _cast(f"({left} != 0) {operator} ({right} != 0)", ctype, dialect)
    if operation == "sum":
        return f"({left}+{right})"
    comparator = "<" if operation == "min" else ">"
    ordinary = f"({left} {comparator} {right} ? {left} : {right})"
    if dtype.kind != "f":
        return ordinary
    if dialect == "vulkan":
        bits = f"floatBitsToUint({left})" if dtype.itemsize == 4 else f"unpackDouble2x32({left}).y"
        negative = f"(({bits} & 0x80000000u) != 0u)"
    else:
        negative = f"signbit({left})"
    tie = (
        f"({negative} ? {left} : {right})"
        if operation == "min"
        else f"({negative} ? {right} : {left})"
    )
    finite = f"(({left} == 0 && {right} == 0) ? {tie} : {ordinary})"
    return f"(isnan({left}) ? {left} : (isnan({right}) ? {right} : {finite}))"


def numpy_extreme(value, axes, keepdims, operation):
    """NumPy extrema with the same deliberate signed-zero tie contract as GPUs."""
    if operation not in ("min", "max"):
        raise ValueError(f"Unknown extreme reduction operation: {operation}")
    function = np.min if operation == "min" else np.max
    result = function(value, axis=axes, keepdims=keepdims)
    if value.dtype.kind == "f" and np.any(result == 0):
        negative = np.signbit(value)
        if operation == "min":
            sign = np.any((value == 0) & negative, axis=axes, keepdims=keepdims)
        else:
            sign = ~np.any((value == 0) & ~negative, axis=axes, keepdims=keepdims)
        zero = np.copysign(np.zeros_like(result), np.where(sign, -1, 1))
        result = np.where(result == 0, zero, result).astype(value.dtype, copy=False)
    return np.asarray(result, dtype=value.dtype).reshape(-1)
