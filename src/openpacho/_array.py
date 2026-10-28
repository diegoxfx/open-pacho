"""A deliberately bounded NumPy-compatible array API with explicit host transfers."""

import operator
from math import prod

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from ._ops import UFUNCS as _UFUNCS
from ._ops import resolve_elementwise
from .backends import get_backend
from .exceptions import BackendMismatchError

_FUNCTIONS = {}
_UNSET = object()


def _shape(value):
    if isinstance(value, (int, np.integer)):
        value = (value,)
    result = tuple(operator.index(d) for d in value)
    if any(d < 0 for d in result):
        raise ValueError("negative dimensions are not allowed")
    return result


def _strides(shape):
    return tuple(prod(shape[i + 1 :]) for i in range(len(shape)))


class ndarray(NDArrayOperatorsMixin):
    """C-contiguous array owned by one backend. Construct with array()/asarray().

    Slicing and transpose produce independent arrays, unlike NumPy views. Reshape
    shares storage, but arrays do not support item assignment in this release.
    """

    __array_priority__ = 1000

    def __init__(self, *args, **kwargs):
        raise TypeError("Use openpacho.array(), asarray(), zeros(), or ones()")

    @classmethod
    def _wrap(cls, data, shape, dtype, backend):
        if getattr(backend, "_is_tracing", False):
            return backend.wrap(data, shape, dtype)
        obj = object.__new__(cls)
        obj._data, obj._shape, obj._dtype, obj._backend = (
            data,
            tuple(shape),
            np.dtype(dtype),
            backend,
        )
        return obj

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return prod(self.shape)

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def backend(self):
        return self._backend.name

    @property
    def device(self):
        return self._backend.device

    @property
    def T(self):
        return transpose(self)

    def __repr__(self):
        return f"openpacho.ndarray(shape={self.shape}, dtype={self.dtype}, backend={self.backend!r}, device={self.device!r})"

    def __len__(self):
        if not self.ndim:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __bool__(self):
        raise ValueError("Array truth is ambiguous; transfer a scalar explicitly with .item()")

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            "Implicit host conversion is disabled. Use openpacho.asnumpy(a) or a.to_host()"
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        kwargs.pop("out", None)
        if ufunc is np.matmul:
            return matmul(*inputs, **kwargs)
        if ufunc not in _UFUNCS.values():
            return NotImplemented
        if kwargs:
            raise NotImplementedError(f"ufunc options {tuple(kwargs)} are not supported")
        return _elementwise(ufunc, *inputs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _FUNCTIONS or not all(issubclass(t, (ndarray, np.ndarray)) for t in types):
            return NotImplemented
        return _FUNCTIONS[func](*args, **kwargs)

    def to_host(self):
        """Synchronize and return an independent NumPy array."""
        return self._backend.download(self)

    def item(self):
        if self.size != 1:
            raise ValueError("item() requires exactly one element")
        return self.to_host().item()

    def copy(self):
        return self.astype(self.dtype, copy=True)

    def astype(self, dtype, *, copy=True):
        dtype = self._backend.validate_dtype(dtype)
        if not copy and dtype == self.dtype:
            return self
        data = self._backend.elementwise("cast", [self], self.shape, dtype)
        return self._wrap(data, self.shape, dtype, self._backend)

    def reshape(self, *shape, order="C"):
        if len(shape) == 1 and not isinstance(shape[0], (int, np.integer)):
            shape = shape[0]
        return reshape(self, shape, order=order)

    def transpose(self, *axes):
        if len(axes) == 1 and (axes[0] is None or isinstance(axes[0], (tuple, list))):
            axes = axes[0]
        return transpose(self, axes or None)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        return sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        return mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def min(self, axis=None, out=None, keepdims=False, initial=_UNSET, where=_UNSET):
        return amin(self, axis, out, keepdims, initial, where)

    def max(self, axis=None, out=None, keepdims=False, initial=_UNSET, where=_UNSET):
        return amax(self, axis, out, keepdims, initial, where)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if builtins_sum(k is Ellipsis for k in key) > 1:
            raise IndexError("an index can only have a single ellipsis")
        used = builtins_sum(k is not None and k is not Ellipsis for k in key)
        if used > self.ndim:
            raise IndexError("too many indices")
        expanded = []
        for k in key:
            expanded.extend([slice(None)] * (self.ndim - used) if k is Ellipsis else [k])
        if not any(k is Ellipsis for k in key):
            expanded.extend([slice(None)] * (self.ndim - used))
        shape, strides, offset, axis = [], [], 0, 0
        source_strides = _strides(self.shape)
        for k in expanded:
            if k is None:
                shape.append(1)
                strides.append(0)
                continue
            dim, stride = self.shape[axis], source_strides[axis]
            axis += 1
            if isinstance(k, slice):
                start, stop, step = k.indices(dim)
                shape.append(len(range(start, stop, step)))
                strides.append(stride * step)
                offset += start * stride
            elif isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)):
                idx = operator.index(k)
                if idx < -dim or idx >= dim:
                    raise IndexError("index out of bounds")
                offset += (idx % dim) * stride
            else:
                raise NotImplementedError(
                    "Only integer, slice, ellipsis and newaxis indexing are supported"
                )
        data = self._backend.gather(self, tuple(shape), offset, tuple(strides))
        return self._wrap(data, shape, self.dtype, self._backend)


from builtins import sum as builtins_sum


def array(obj, dtype=None, *, copy=True, order="C", backend=None):
    if order != "C":
        raise NotImplementedError("Only C order is supported")
    if isinstance(obj, ndarray):
        target = obj._backend if backend is None else get_backend(backend)
        if target is not obj._backend:
            raise BackendMismatchError("Use to_backend(a, backend) to transfer explicitly")
        return obj.astype(dtype or obj.dtype, copy=copy)
    target = get_backend(backend)
    host = np.asarray(obj, dtype=dtype, order="C")
    target.validate_dtype(host.dtype)
    return ndarray._wrap(target.upload(host), host.shape, host.dtype, target)


def asarray(obj, dtype=None, *, order="C", backend=None):
    return array(obj, dtype=dtype, copy=False, order=order, backend=backend)


def asnumpy(a):
    return a.to_host() if isinstance(a, ndarray) else np.array(a, copy=True)


def to_backend(a, backend, *, device=None):
    target = get_backend(backend, device=device)
    if isinstance(a, ndarray) and a._backend is target:
        return a
    return asarray(asnumpy(a), backend=target)


def _common_backend(args):
    found = [a._backend for a in args if isinstance(a, ndarray)]
    if found and any(b is not found[0] for b in found):
        raise BackendMismatchError("Arrays belong to different backends/devices; use to_backend()")
    return found[0] if found else get_backend()


def _elementwise(ufunc, *inputs):
    from ._fusion import Expression, HostScalar, trace_op

    if any(isinstance(a, Expression) for a in inputs):
        return trace_op(ufunc, *inputs)
    backend = _common_backend(inputs)
    resolved, shape = resolve_elementwise(ufunc, inputs)
    for dtype in resolved:
        backend.validate_dtype(dtype)
    args = [
        HostScalar(a, dt)
        if backend.name in ("opencl", "vulkan", "cuda") and np.isscalar(a)
        else asarray(a, dtype=dt, backend=backend)
        for a, dt in zip(inputs, resolved)
    ]
    shape = np.broadcast_shapes(*(a.shape for a in args))
    data = backend.elementwise(ufunc.__name__, args, shape, resolved[-1])
    return ndarray._wrap(data, shape, resolved[-1], backend)


def _unary_or_binary(ufunc):
    def call(*args):
        return _elementwise(ufunc, *args)

    call.__name__ = ufunc.__name__
    return call


globals().update({name: _unary_or_binary(ufunc) for name, ufunc in _UFUNCS.items()})


_WHERE_MISSING = object()


def where(condition, x=_WHERE_MISSING, y=_WHERE_MISSING):
    """Select broadcasted values on the chosen backend; both branches are evaluated."""
    if x is _WHERE_MISSING and y is _WHERE_MISSING:
        raise NotImplementedError("One-argument where (nonzero indices) is not supported")
    if x is _WHERE_MISSING or y is _WHERE_MISSING:
        raise TypeError("where requires both x and y")
    return _elementwise(np.where, condition, x, y)


def empty(shape, dtype=float, *, order="C", backend=None):
    if order != "C":
        raise NotImplementedError("Only C order is supported")
    shape = _shape(shape)
    target = get_backend(backend)
    dtype = target.validate_dtype(dtype)
    return ndarray._wrap(target.empty(shape, dtype), shape, dtype, target)


def full(shape, fill_value, dtype=None, *, order="C", backend=None):
    if np.ndim(fill_value) != 0:
        raise NotImplementedError("full() supports scalar fill values")
    if order != "C":
        raise NotImplementedError("Only C order is supported")
    shape = _shape(shape)
    target = get_backend(backend)
    dtype = target.validate_dtype(dtype or np.asarray(fill_value).dtype)
    return ndarray._wrap(target.full(shape, fill_value, dtype), shape, dtype, target)


def zeros(shape, dtype=float, *, order="C", backend=None):
    return full(shape, 0, dtype=dtype, order=order, backend=backend)


def ones(shape, dtype=float, *, order="C", backend=None):
    return full(shape, 1, dtype=dtype, order=order, backend=backend)


def zeros_like(a, dtype=None):
    a = asarray(a)
    return zeros(a.shape, dtype=dtype or a.dtype, backend=a._backend)


def ones_like(a, dtype=None):
    a = asarray(a)
    return ones(a.shape, dtype=dtype or a.dtype, backend=a._backend)


def arange(start, stop=None, step=1, dtype=None, *, backend=None):
    # Constructors execute once on host and upload; arithmetic never does.
    return asarray(np.arange(start, stop, step, dtype=dtype), backend=backend)


def linspace(start, stop, num=50, endpoint=True, dtype=None, *, backend=None):
    return asarray(
        np.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype), backend=backend
    )


def eye(N, M=None, k=0, dtype=float, *, backend=None):
    return asarray(np.eye(N, M=M, k=k, dtype=dtype), backend=backend)


def reshape(a, shape, order="C"):
    a = asarray(a)
    if order != "C":
        raise NotImplementedError("Only C order is supported")
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    shape = tuple(operator.index(d) for d in shape)
    if builtins_sum(d == -1 for d in shape) > 1 or any(d < -1 for d in shape):
        raise ValueError("Invalid reshape dimensions")
    if -1 in shape:
        known = prod(d for d in shape if d != -1)
        if known == 0 or a.size % known:
            raise ValueError("Cannot infer reshape dimension")
        shape = tuple(a.size // known if d == -1 else d for d in shape)
    if prod(shape) != a.size:
        raise ValueError("Cannot reshape array to requested shape")
    return ndarray._wrap(a._data, shape, a.dtype, a._backend)


def transpose(a, axes=None):
    a = asarray(a)
    axes = tuple(reversed(range(a.ndim))) if axes is None else tuple(axes)
    if len(axes) != a.ndim:
        raise ValueError("axes don't match array")
    axes = tuple(_axis(i, a.ndim) for i in axes)
    if len(set(axes)) != a.ndim:
        raise ValueError("repeated axis in transpose")
    shape = tuple(a.shape[i] for i in axes)
    strides = tuple(_strides(a.shape)[i] for i in axes)
    data = a._backend.gather(a, shape, 0, strides)
    return ndarray._wrap(data, shape, a.dtype, a._backend)


def _axis(axis, ndim):
    axis = operator.index(axis)
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} out of bounds for array of dimension {ndim}")
    return axis % ndim


def _axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    values = axis if isinstance(axis, tuple) else (axis,)
    # NumPy allows axis=0 and axis=-1 on scalar reductions (but not tuples).
    if ndim == 0 and not isinstance(axis, tuple) and axis in (0, -1):
        return ()
    values = tuple(_axis(i, ndim) for i in values)
    if len(set(values)) != len(values):
        raise ValueError("duplicate reduction axis")
    return tuple(sorted(values))


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if out is not None:
        raise NotImplementedError("out= is not supported")
    a = asarray(a)
    axes = _axes(axis, a.ndim)
    if dtype is None:
        dtype = (
            np.dtype("int64")
            if a.dtype.kind in "bi"
            else np.dtype("uint64")
            if a.dtype.kind == "u"
            else a.dtype
        )
    dtype = a._backend.validate_dtype(dtype)
    shape = (
        tuple(1 if i in axes else d for i, d in enumerate(a.shape))
        if keepdims
        else tuple(d for i, d in enumerate(a.shape) if i not in axes)
    )
    data = a._backend.reduce(a, axes, keepdims, dtype)
    return ndarray._wrap(data, shape, dtype, a._backend)


def _extreme(a, axis, out, keepdims, initial, where, operation):
    if out is not None:
        raise NotImplementedError("out= is not supported")
    if initial is not _UNSET:
        raise NotImplementedError("initial= is not supported")
    if where is not _UNSET:
        raise NotImplementedError("where= is not supported")
    a = asarray(a)
    axes = _axes(axis, a.ndim)
    if prod(a.shape[i] for i in axes) == 0:
        raise ValueError(
            f"zero-size array to reduction operation {operation} which has no identity"
        )
    shape = (
        tuple(1 if i in axes else d for i, d in enumerate(a.shape))
        if keepdims
        else tuple(d for i, d in enumerate(a.shape) if i not in axes)
    )
    data = a._backend.reduce(a, axes, keepdims, a.dtype, operation=operation)
    return ndarray._wrap(data, shape, a.dtype, a._backend)


def amin(a, axis=None, out=None, keepdims=False, initial=_UNSET, where=_UNSET):
    """Minimum preserving dtype, propagating NaNs and preferring negative zero ties."""
    return _extreme(a, axis, out, keepdims, initial, where, "min")


def amax(a, axis=None, out=None, keepdims=False, initial=_UNSET, where=_UNSET):
    """Maximum preserving dtype, propagating NaNs and preferring positive zero ties."""
    return _extreme(a, axis, out, keepdims, initial, where, "max")


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    a = asarray(a)
    # Unlike np.sum, np.mean rejects integer axes on a zero-dimensional input.
    if a.ndim == 0 and axis is not None and not isinstance(axis, tuple):
        raise ValueError("axis out of bounds for scalar mean")
    dtype = np.dtype(dtype or (a.dtype if a.dtype.kind == "f" else np.float64))
    if dtype.kind != "f":
        raise NotImplementedError("mean supports floating point accumulation only")
    axes = _axes(axis, a.ndim)
    count = prod(a.shape[i] for i in axes)
    return sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims) / count


def matmul(a, b, *, padding=False):
    """Matrix product; padding is an opt-in hint for supported custom GPU kernels."""
    if padding is not False:
        from ._padding import prepare_padding, validate_padding

        padding = validate_padding(padding)
    backend = _common_backend((a, b))
    a, b = asarray(a, backend=backend), asarray(b, backend=backend)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("matmul requires arrays with at least one dimension")
    if a.shape[-1] != b.shape[-2 if b.ndim > 1 else 0]:
        raise ValueError("matmul inner dimensions do not match")
    dtype = backend.validate_dtype(np.result_type(a.dtype, b.dtype))
    if dtype.kind == "b":
        raise NotImplementedError("Boolean matmul is not supported")
    a, b = a.astype(dtype, copy=False), b.astype(dtype, copy=False)
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    shape = (
        batch + (() if a.ndim == 1 else (a.shape[-2],)) + (() if b.ndim == 1 else (b.shape[-1],))
    )
    plan = prepare_padding(backend, a.shape, b.shape, shape, dtype) if padding else None
    data = backend.matmul(a, b, shape, dtype) if plan is None else plan.run(a, b)
    return ndarray._wrap(data, shape, dtype, backend)


def dot(a, b):
    if (a.ndim if isinstance(a, ndarray) else np.ndim(a)) == 0 or (
        b.ndim if isinstance(b, ndarray) else np.ndim(b)
    ) == 0:
        return _elementwise(np.multiply, a, b)
    if (a.ndim if isinstance(a, ndarray) else np.ndim(a)) > 2 or (
        b.ndim if isinstance(b, ndarray) else np.ndim(b)
    ) > 2:
        raise NotImplementedError("Higher-rank dot differs from batched matmul; use @ for batches")
    return matmul(a, b)


def diag(a, k=0):
    a = asarray(a)
    k = operator.index(k)
    if a.ndim != 2:
        raise NotImplementedError("diag currently extracts matrix diagonals only")
    rows, cols = a.shape
    start_row, start_col = max(0, -k), max(0, k)
    length = max(0, min(rows - start_row, cols - start_col))
    data = a._backend.gather(a, (length,), start_row * cols + start_col, (cols + 1,))
    return ndarray._wrap(data, (length,), a.dtype, a._backend)


for _name in (
    "sum",
    "amin",
    "amax",
    "mean",
    "reshape",
    "transpose",
    "dot",
    "diag",
    "zeros_like",
    "ones_like",
    "where",
):
    _FUNCTIONS[getattr(np, _name)] = globals()[_name]

_FUNCTIONS[np.min] = amin
_FUNCTIONS[np.max] = amax
