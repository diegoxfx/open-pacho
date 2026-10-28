"""Raw-bit rectangular copies used by the opt-in GEMM padding path.

The device allocations remain float32 storage, but these kernels bind the same
four-byte words as unsigned integers.  That is deliberate: a copy must retain
all float32 bit patterns, including signed zeroes, subnormals, and NaN payloads.
"""

import operator
from functools import lru_cache
from math import prod

import numpy as np

_MAX_INDEX = 2**31 - 1
_UINT32 = np.dtype("uint32")
_FLOAT32 = np.dtype("float32")


def _dimensions(value, label):
    """Normalize a shape while keeping malformed dimensions out of kernels."""

    try:
        result = tuple(operator.index(dimension) for dimension in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a sequence of integer dimensions") from exc
    if any(dimension < 0 for dimension in result):
        raise ValueError(f"{label} cannot contain negative dimensions")
    return result


def _checked_count(shape, label):
    count = prod(shape)
    if count > _MAX_INDEX:
        raise ValueError(f"{label} exceeds the GPU indexing limit")
    return count


@lru_cache(maxsize=128)
def _layout(source_shape, destination_shape, dtype_string):
    """Cache immutable geometry only, after dimensions/dtype are normalized."""

    if len(source_shape) < 2 or len(destination_shape) < 2:
        raise ValueError("padding copies require rank >= 2")
    if len(source_shape) != len(destination_shape):
        raise ValueError("input and output ranks must match")
    if source_shape[:-2] != destination_shape[:-2]:
        raise ValueError("input and output leading batch dimensions must match")
    if dtype_string != _FLOAT32.str:
        raise TypeError("padding copies require float32 input")
    return (
        _checked_count(source_shape, "input"),
        _checked_count(destination_shape, "output"),
    )


def _validate(a, shape):
    """Validate every current input; cache only the immutable layout checks."""

    try:
        source_shape = _dimensions(a.shape, "input shape")
        dtype = np.dtype(a.dtype)
    except AttributeError as exc:
        raise TypeError("input must expose shape and dtype metadata") from exc
    destination_shape = _dimensions(shape, "output shape")
    # A dtype's optional metadata can hold arbitrary objects, even array/storage
    # owners. Cache its primitive descriptor, never the dtype object itself.
    source_count, destination_count = _layout(source_shape, destination_shape, dtype.str)

    # openpacho.ndarray is always C-contiguous.  Honor an exposed NumPy-style
    # flag when a recording/fake array supplies one, without requiring GPU
    # storage objects to implement host-array metadata.
    flags = getattr(a, "flags", None)
    if flags is not None:
        try:
            contiguous = flags.c_contiguous
        except AttributeError:
            try:
                contiguous = flags["C_CONTIGUOUS"]
            except (KeyError, TypeError, IndexError):
                contiguous = None
        if contiguous is False:
            raise ValueError("padding copies require a C-contiguous input")

    if not hasattr(a, "_data"):
        raise TypeError("input must expose backend storage as _data")
    storage = a._data
    if isinstance(storage, np.ndarray) and not storage.flags.c_contiguous:
        raise ValueError("padding copies require C-contiguous backend storage")
    declared_size = getattr(a, "size", source_count)
    try:
        declared_size = operator.index(declared_size)
    except TypeError as exc:
        raise TypeError("input size must be an integer") from exc
    if declared_size != source_count:
        raise ValueError("input size does not match its shape")
    return source_shape, destination_shape, destination_count, dtype


@lru_cache(maxsize=128)
def _body(source_shape, destination_shape, index_type):
    """Return the dialect-neutral copy body for one destination element."""

    source_rows, source_cols = source_shape[-2:]
    destination_rows, destination_cols = destination_shape[-2:]
    # This function is only launched for a non-empty destination.  max(1, ...)
    # keeps source generation well-formed for direct source inspection when a
    # caller asks for an empty shape.
    destination_matrix = max(1, destination_rows * destination_cols)
    destination_width = max(1, destination_cols)
    batch = f"{index_type} batch=i/{destination_matrix}u;"
    remainder = f"{index_type} rem=i%{destination_matrix}u;"
    row = f"{index_type} row=rem/{destination_width}u;"
    column = f"{index_type} col=rem%{destination_width}u;"

    if not source_rows or not source_cols:
        copy = "x1[i]=0u;"
    else:
        source_index = f"((batch*{source_rows}u+row)*{source_cols}u+col)"
        if destination_rows <= source_rows and destination_cols <= source_cols:
            copy = f"x1[i]=x0[{source_index}];"
        else:
            copy = (
                f"if(row<{source_rows}u&&col<{source_cols}u)"
                f" x1[i]=x0[{source_index}];"
                " else x1[i]=0u;"
            )
    return f"{batch} {remainder} {row} {column} {copy}"


@lru_cache(maxsize=128)
def _opencl_source(source_shape, destination_shape):
    body = _body(source_shape, destination_shape, "uint")
    return (
        "__kernel void compute(__global const uint *x0, __global uint *x1) {"
        f"uint i=get_global_id(0); {body} }}"
    )


@lru_cache(maxsize=128)
def _cuda_source(source_shape, destination_shape):
    count = prod(destination_shape)
    body = _body(source_shape, destination_shape, "unsigned int")
    return (
        'extern "C" __global__ void compute(const unsigned int*x0, unsigned int*x1) {'
        f"unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;"
        f"if(i>={count}u) return; {body} }}"
    )


def _prepare(backend, a, shape):
    source_shape, destination_shape, count, dtype = _validate(a, shape)
    out = backend.empty(destination_shape, dtype)
    return source_shape, destination_shape, count, out


def opencl_copy(backend, a, shape):
    """Copy/pad/crop ``a`` into OpenCL storage using raw uint32 words."""

    source_shape, destination_shape, count, out = _prepare(backend, a, shape)
    if count:
        backend._run(_opencl_source(source_shape, destination_shape), count, a._data, out)
    return out


def cuda_copy(backend, a, shape):
    """Copy/pad/crop ``a`` into CUDA storage using raw uint32 words."""

    source_shape, destination_shape, count, out = _prepare(backend, a, shape)
    if count:
        backend._run(
            _cuda_source(source_shape, destination_shape),
            (count + 127) // 128,
            a._data,
            out,
            threads=128,
        )
    return out


def vulkan_copy(backend, a, shape):
    """Copy/pad/crop ``a`` into Vulkan storage using raw uint32 words."""

    source_shape, destination_shape, count, out = _prepare(backend, a, shape)
    if count:
        backend._run(
            _body(source_shape, destination_shape, "uint"),
            count,
            [(a._data, _UINT32), (out, _UINT32)],
        )
    return out


__all__ = ["cuda_copy", "opencl_copy", "vulkan_copy"]
