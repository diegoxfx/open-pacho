"""Typed operation capture shared by elementwise fuse and cached compile."""

import operator
from dataclasses import dataclass
from functools import wraps
from types import SimpleNamespace

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from ._array import ndarray
from ._ops import HostScalar, resolve_elementwise
from .exceptions import BackendMismatchError


@dataclass(frozen=True)
class InputRef:
    index: int


class ScalarExpression(NDArrayOperatorsMixin):
    """Host scalar arithmetic stays on host and is rebound on every plan call."""

    __array_priority__ = 1500
    shape = ()

    def __init__(self, evaluate, sample):
        self.evaluate = evaluate
        self.sample = sample
        self.weak_type = type(sample) if type(sample) in (int, float, complex) else None
        self.dtype = np.asarray(sample).dtype

    def __bool__(self):
        raise TypeError("Data-dependent control flow is not captured; declare static_argnames")

    def __index__(self):
        raise TypeError("Shape/axis/loop arguments must be declared in static_argnames")

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Cannot convert a captured scalar to a host array")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            raise NotImplementedError("Only plain scalar ufunc calls can be captured")
        if any(isinstance(a, Expression) for a in inputs):
            return trace_op(ufunc, *inputs)
        return self._combine(ufunc, inputs)

    @staticmethod
    def _combine(function, inputs):
        if any(isinstance(a, ndarray) for a in inputs):
            return NotImplemented
        samples = [a.sample if isinstance(a, ScalarExpression) else a for a in inputs]
        with np.errstate(all="ignore"):
            sample = function(*samples)

        def evaluate(bindings):
            values = [
                a.evaluate(bindings) if isinstance(a, ScalarExpression) else a for a in inputs
            ]
            with np.errstate(all="ignore"):
                return function(*values)

        return ScalarExpression(evaluate, sample)


for _dunder, _operator in (
    ("add", operator.add),
    ("sub", operator.sub),
    ("mul", operator.mul),
    ("truediv", operator.truediv),
):

    def _forward(self, other, fn=_operator):
        return self._combine(fn, (self, other))

    def _reverse(self, other, fn=_operator):
        return self._combine(fn, (other, self))

    setattr(ScalarExpression, f"__{_dunder}__", _forward)
    setattr(ScalarExpression, f"__r{_dunder}__", _reverse)
ScalarExpression.__neg__ = lambda self: self._combine(operator.neg, (self,))
ScalarExpression.__pos__ = lambda self: self._combine(operator.pos, (self,))


class Expression(ndarray):
    __array_priority__ = 2000

    def __init__(
        self, backend, shape, dtype, *, op="input", inputs=(), value=None, resolved=(), attrs=None
    ):
        self._shape, self._dtype, self._backend = tuple(shape), np.dtype(dtype), backend
        self._data = self
        self.op, self.inputs, self.value = op, tuple(inputs), value
        self.resolved, self.attrs = tuple(resolved), attrs or {}

    def __bool__(self):
        raise TypeError("Captured functions do not support data-dependent Python control flow")

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Host conversion is forbidden during capture")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            raise NotImplementedError("Only plain ufunc calls can be captured")
        kwargs.pop("out", None)
        if kwargs:
            raise NotImplementedError("ufunc options are not captured")
        if ufunc is np.matmul:
            from ._array import matmul

            return matmul(*inputs)
        return trace_op(ufunc, *inputs)


class TraceBackend:
    _is_tracing = True

    def __init__(self, target, *, elementwise_only=False):
        self.target = target
        self.name, self.device = target.name, target.device
        self.elementwise_only = elementwise_only

    def validate_dtype(self, dtype):
        return self.target.validate_dtype(dtype)

    def wrap(self, data, shape, dtype):
        if not isinstance(data, Expression):
            raise TypeError("Invalid captured storage")
        if data.shape == tuple(shape) and data.dtype == np.dtype(dtype):
            return data
        self._general("reshape")
        return Expression(self, shape, dtype, op="reshape", inputs=(data,))

    def _general(self, name):
        if self.elementwise_only:
            raise NotImplementedError(
                f"fuse supports elementwise expressions only, not {name}; use compile"
            )

    def upload(self, value):
        value = np.asarray(value)
        if value.ndim:
            raise TypeError("Pass host arrays as explicit OpenPacho arguments before capture")
        return Expression(self, (), value.dtype, op="scalar", value=HostScalar(value, value.dtype))

    def download(self, value):
        raise TypeError(
            "Host observations are forbidden during capture; observe the returned array"
        )

    def synchronize(self):
        raise TypeError("Explicit synchronization is not captured")

    def empty(self, shape, dtype):
        raise NotImplementedError("Uninitialized arrays are not captured")

    def full(self, shape, value, dtype):
        self._general("full")
        return Expression(self, shape, dtype, op="full", attrs={"value": value})

    def elementwise(self, name, args, shape, dtype):
        if name != "cast":
            return trace_op(getattr(np, name), *args)
        return Expression(
            self, shape, dtype, op="cast", inputs=tuple(args), resolved=(args[0].dtype, dtype)
        )

    def gather(self, a, shape, offset, source_strides):
        self._general("gather")
        return Expression(
            self,
            shape,
            a.dtype,
            op="gather",
            inputs=(a,),
            attrs={"offset": offset, "strides": tuple(source_strides)},
        )

    def reduce(self, a, axes, keepdims, dtype, *, operation="sum"):
        self._general("reduce")
        shape = (
            tuple(1 if i in axes else d for i, d in enumerate(a.shape))
            if keepdims
            else tuple(d for i, d in enumerate(a.shape) if i not in axes)
        )
        attrs = {"axes": tuple(axes), "keepdims": keepdims}
        if operation != "sum":
            attrs["operation"] = operation
        return Expression(
            self,
            shape,
            dtype,
            op="reduce",
            inputs=(a,),
            attrs=attrs,
        )

    def matmul(self, a, b, shape, dtype):
        self._general("matmul")
        return Expression(self, shape, dtype, op="matmul", inputs=(a, b))

    def prepare_padded_matmul(self, ashape, bshape, shape, dtype):
        self._general("matmul")
        return SimpleNamespace(
            run=lambda a, b: Expression(
                self, shape, dtype, op="matmul", inputs=(a, b), attrs={"padding": True}
            )
        )

    def norm(self, a, axes, keepdims, dtype):
        self._general("norm")
        shape = (
            tuple(1 if i in axes else d for i, d in enumerate(a.shape))
            if keepdims
            else tuple(d for i, d in enumerate(a.shape) if i not in axes)
        )
        return Expression(
            self,
            shape,
            dtype,
            op="norm",
            inputs=(a,),
            attrs={"axes": tuple(axes), "keepdims": keepdims},
        )


def trace_op(ufunc, *inputs):
    traced = [a for a in inputs if isinstance(a, Expression)]
    backend = traced[0]._backend
    for value in inputs:
        if isinstance(value, ndarray) and value._backend is not backend:
            raise BackendMismatchError(
                "Captured arrays must be explicit arguments on the same backend"
            )
    resolved, shape = resolve_elementwise(ufunc, inputs)
    for dtype in resolved:
        backend.validate_dtype(dtype)
    nodes = []
    for value, dtype in zip(inputs, resolved):
        if isinstance(value, Expression):
            nodes.append(value)
        elif isinstance(value, ScalarExpression):
            nodes.append(Expression(backend, (), dtype, op="scalar", value=value))
        elif np.isscalar(value):
            nodes.append(
                Expression(backend, (), dtype, op="scalar", value=HostScalar(value, dtype))
            )
        else:
            raise TypeError("Captured operands must be arrays or numeric scalars")
    return Expression(
        backend, shape, resolved[-1], op=ufunc.__name__, inputs=nodes, resolved=resolved
    )


def fuse(function):
    """Capture one pure elementwise result; use compile for persistent plans."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        from ._compiler import capture_call

        plan, bindings = capture_call(function, args, kwargs, elementwise_only=True)
        return plan.execute(bindings)

    return wrapped
