"""Cached, bounded function capture and backend-independent execution plans."""

import inspect
from collections import Counter, OrderedDict
from functools import update_wrapper
from threading import RLock

import numpy as np

from ._array import ndarray
from ._fusion import Expression, InputRef, ScalarExpression, TraceBackend
from ._lowering import create_lowering
from ._ops import UFUNCS, HostScalar
from ._padding import prepare_padding, validate_padding
from .backends import use_backend
from .exceptions import BackendMismatchError

ELEMENTWISE = {f.__name__ for f in UFUNCS.values()} | {"cast", "where"}


def _bind(function, args, kwargs, signature=None):
    signature = signature or inspect.signature(function)
    if any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in signature.parameters.values()):
        raise TypeError("compile requires named parameters, without *args or **kwargs")
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def capture_call(
    function, args, kwargs, *, elementwise_only=False, static_argnames=(), padding=False
):
    bound = _bind(function, args, kwargs)
    bindings = tuple(bound.arguments.values())
    arrays = [a for a in bindings if isinstance(a, ndarray)]
    if not arrays:
        raise TypeError("Capture requires at least one OpenPacho array argument")
    target = arrays[0]._backend
    if getattr(target, "_is_tracing", False):
        raise TypeError(
            "Nested compile/fuse calls are not supported; wrap the outer plain function"
        )
    if any(a._backend is not target for a in arrays):
        raise BackendMismatchError("Captured arguments belong to different backends/devices")
    trace = TraceBackend(target, elementwise_only=elementwise_only)
    for index, (name, value) in enumerate(tuple(bound.arguments.items())):
        if name in static_argnames:
            continue
        if isinstance(value, ndarray):
            bound.arguments[name] = Expression(
                trace, value.shape, value.dtype, value=InputRef(index)
            )
        elif isinstance(value, (int, float, bool, np.generic)):
            bound.arguments[name] = ScalarExpression(lambda b, i=index: b[i], value)
        else:
            raise TypeError(
                f"Argument {name!r} must be an OpenPacho array or numeric scalar; "
                "declare shape/axis options in static_argnames"
            )
    with use_backend(trace):
        result = function(*bound.args, **bound.kwargs)
    multiple = isinstance(result, tuple)
    outputs = result if multiple else (result,)
    if not outputs or any(not isinstance(a, Expression) for a in outputs):
        raise TypeError("Captured functions must return array expressions")
    if elementwise_only and multiple:
        raise TypeError("fuse must return one elementwise array expression")
    return ExecutionPlan(target, outputs, multiple, padding=padding), bindings


def _topological(outputs):
    result, seen = [], set()

    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        for child in node.inputs:
            visit(child)
        result.append(node)

    for output in outputs:
        visit(output)
    return result


class ExecutionPlan:
    def __init__(self, backend, outputs, multiple, *, padding=False):
        self.backend, self.outputs, self.multiple = backend, outputs, multiple
        self.padding = validate_padding(padding)
        self.lowering = create_lowering(backend)
        self._execution_scope = getattr(backend, "execution_scope", None)
        self.nodes = _topological(outputs)
        self.steps = []
        # Two references from one consumer (x*x) do not need a stored x.
        # Distinct consumers and public outputs remain materialization boundaries.
        self.uses = Counter(child for node in self.nodes for child in {id(a) for a in node.inputs})
        self.output_ids = {id(a) for a in outputs}
        built = set()
        prepare_layout = getattr(backend, "prepare_layout_matmul", None)
        can_reduce_materialized = getattr(self.lowering, "can_reduce_materialized", None)

        def private_transpose(node):
            if (
                node.op != "gather"
                or len(node.inputs) != 1
                or len(node.shape) != 2
                or self.uses[id(node)] != 1
                or id(node) in self.output_ids
            ):
                return False
            original = node.inputs[0]
            return (
                len(original.shape) == 2
                and node.shape == original.shape[::-1]
                and node.dtype == original.dtype
                and node.attrs.get("offset") == 0
                and node.attrs.get("strides") == (1, original.shape[1])
            )

        def ensure(node):
            if id(node) in built:
                return
            if node.op in ("input", "scalar"):
                built.add(id(node))
                return
            if node.op == "matmul" and (self.padding or node.attrs.get("padding", False)):
                left, right = node.inputs
                padded = prepare_padding(backend, left.shape, right.shape, node.shape, node.dtype)
                if padded is not None:
                    for arg in node.inputs:
                        ensure(arg)
                    self.steps.append((node, "padded_matmul", node.inputs, padded))
                    built.add(id(node))
                    return
            if node.op == "matmul" and prepare_layout is not None:
                transposed = tuple(private_transpose(arg) for arg in node.inputs)
                if any(transposed):
                    args = tuple(
                        arg.inputs[0] if transpose else arg
                        for arg, transpose in zip(node.inputs, transposed)
                    )
                    layout = prepare_layout(
                        args[0].shape,
                        args[1].shape,
                        node.shape,
                        node.dtype,
                        transpose_a=transposed[0],
                        transpose_b=transposed[1],
                    )
                    if layout is not None:
                        for arg in args:
                            ensure(arg)
                        self.steps.append((node, "layout_matmul", args, layout))
                        built.add(id(node))
                        return
            # Fuse only a private GEMV product with its residual epilogue.
            # An explicitly selected BLAS provider remains the owner of matmul.
            if (
                node.op == "subtract"
                and self.lowering is not None
                and node.inputs[1].op == "matmul"
            ):
                rhs, product = node.inputs
                matrix, vector = product.inputs
                if (
                    self.uses[id(product)] == 1
                    and id(product) not in self.output_ids
                    and self.lowering.can_residual_matvec(matrix, vector, rhs, node)
                ):
                    args = (matrix, vector, rhs)
                    for arg in args:
                        ensure(arg)
                    self.steps.append((node, "residual_matvec", args, None))
                    built.add(id(node))
                    return
            map_reduce = (
                self.lowering is not None
                and node.op == "reduce"
                and node.inputs[0].op in ELEMENTWISE
                and self.uses[id(node.inputs[0])] == 1
                and id(node.inputs[0]) not in self.output_ids
            )
            if (self.lowering is not None and node.op in ELEMENTWISE) or map_reduce:
                group, leaves, visited = [], [], set()

                def collect(current, root=False):
                    if id(current) in visited:
                        return
                    visited.add(id(current))
                    inline = current.op in ELEMENTWISE and (
                        root or (id(current) not in self.output_ids and self.uses[id(current)] <= 1)
                    )
                    if not inline:
                        ensure(current)
                        leaves.append(current)
                        return
                    for child in current.inputs:
                        collect(child)
                    group.append(current)

                collect(node.inputs[0] if map_reduce else node, True)
                if map_reduce:
                    arguments, source = self.lowering.prepare_map_reduce(node, group, leaves)
                    self.steps.append((node, "map_reduce", arguments, source))
                else:
                    arguments, source = self.lowering.prepare_fused(node, group, leaves)
                    self.steps.append((node, "fused", arguments, source))
            elif (
                node.op == "reduce"
                and can_reduce_materialized is not None
                and can_reduce_materialized(node)
            ):
                # Shared/public producers stay materialization boundaries. The
                # lowerer can reuse its reduction kernel with a single stored leaf.
                for child in node.inputs:
                    ensure(child)
                arguments, source = self.lowering.prepare_map_reduce(node, (), node.inputs)
                self.steps.append((node, "map_reduce", arguments, source))
            else:
                for child in node.inputs:
                    ensure(child)
                self.steps.append((node, node.op, node.inputs, None))
            built.add(id(node))

        for output in outputs:
            ensure(output)
        # A padded matmul is one logical step, but can submit pack A, pack B,
        # GEMM and crop. Include these launches when amortizing backend scope.
        dispatch_count = sum(
            getattr(source, "dispatch_count", 1) if kind == "padded_matmul" else 1
            for _, kind, _, source in self.steps
        )
        if dispatch_count < getattr(backend, "execution_scope_min_steps", 0):
            self._execution_scope = None
        # Count actual plan edges, including fusion boundaries, for prompt release.
        self.live_uses = Counter(id(a) for _, _, args, _ in self.steps for a in args)
        self.live_uses.update(id(a) for a in outputs)

    def execute(self, bindings):
        batch = self._execution_scope
        if batch is not None:
            with batch():
                return self._execute(bindings)
        return self._execute(bindings)

    def _execute(self, bindings):
        backend = self.backend
        values, remaining = {}, self.live_uses.copy()

        def value(node):
            key = id(node)
            if key not in values:
                if node.op == "input":
                    values[key] = bindings[node.value.index]
                elif node.op == "scalar":
                    data = node.value
                    values[key] = (
                        HostScalar(data.evaluate(bindings), node.dtype)
                        if isinstance(data, ScalarExpression)
                        else data
                    )
                else:
                    raise RuntimeError(f"Missing plan dependency {node.op}")
            return values[key]

        for node, kind, args, source in self.steps:
            operands = [value(a) for a in args]
            if kind not in ELEMENTWISE and kind not in ("fused", "map_reduce"):
                # Scalar constants stay by value in expressions, but array
                # primitives require ordinary backend storage, even at rank 0.
                operands = [
                    ndarray._wrap(backend.full((), a._data, a.dtype), (), a.dtype, backend)
                    if isinstance(a, HostScalar)
                    else a
                    for a in operands
                ]
            if kind == "fused":
                data = self.lowering.run_fused(node, source, args, operands)
            elif kind == "map_reduce":
                data = self.lowering.run_map_reduce(node, source, args, operands)
            elif kind in ELEMENTWISE:
                converted = []
                for a, dt in zip(operands, node.resolved):
                    if isinstance(a, HostScalar):
                        a = HostScalar(a._data, dt)
                    elif a.dtype != dt:
                        a = a.astype(dt, copy=False)
                    converted.append(a)
                data = backend.elementwise(kind, converted, node.shape, node.dtype)
            elif kind == "reduce":
                options = (
                    {"operation": node.attrs["operation"]} if "operation" in node.attrs else {}
                )
                data = backend.reduce(
                    operands[0], node.attrs["axes"], node.attrs["keepdims"], node.dtype, **options
                )
            elif kind == "matmul":
                data = backend.matmul(*operands, node.shape, node.dtype)
            elif kind in ("padded_matmul", "layout_matmul"):
                data = source.run(*operands)
            elif kind == "norm":
                data = backend.norm(
                    operands[0], node.attrs["axes"], node.attrs["keepdims"], node.dtype
                )
            elif kind == "residual_matvec":
                data = backend.residual_matvec(*operands)
            elif kind == "gather":
                data = backend.gather(
                    operands[0], node.shape, node.attrs["offset"], node.attrs["strides"]
                )
            elif kind == "reshape":
                data = operands[0]._data
            elif kind == "full":
                data = backend.full(node.shape, node.attrs["value"], node.dtype)
            else:
                raise NotImplementedError(f"No execution rule for {kind}")
            values[id(node)] = ndarray._wrap(data, node.shape, node.dtype, backend)
            for arg in args:
                remaining[id(arg)] -= 1
                if remaining[id(arg)] == 0:
                    values.pop(id(arg), None)
        result = []
        for node in self.outputs:
            output = value(node)
            if isinstance(output, HostScalar):
                output = ndarray._wrap(
                    backend.full((), output._data, node.dtype), (), node.dtype, backend
                )
            result.append(output)
        return tuple(result) if self.multiple else result[0]

    def explain(self):
        return {
            "backend": self.backend.name,
            "device": self.backend.device,
            "blas": getattr(self.backend, "blas", None),
            "padding": self.padding,
            "outputs": len(self.outputs),
            "steps": [
                {
                    "kind": kind,
                    "shape": node.shape,
                    "dtype": str(node.dtype),
                    **(
                        {"operation": node.attrs.get("operation", "sum")}
                        if kind in ("reduce", "map_reduce")
                        else {}
                    ),
                    **({"padding": source.explain()} if kind == "padded_matmul" else {}),
                    **({"layout": source.explain()} if kind == "layout_matmul" else {}),
                }
                for node, kind, _, source in self.steps
            ],
            "host_observations": 0,
            "python_retrace_on_cache_hit": False,
        }


def _static_key(value):
    if isinstance(value, tuple):
        return (tuple, tuple(_static_key(v) for v in value))
    if value is None or isinstance(value, (str, int, float, bool, np.dtype)):
        return (type(value), value)
    raise TypeError("Static arguments must be immutable scalar values, dtypes or tuples")


class CompiledFunction:
    def __init__(self, function, static_argnames=(), cache_size=32, *, padding=False):
        self._padding = validate_padding(padding)
        if isinstance(static_argnames, str):
            static_argnames = (static_argnames,)
        self.static_argnames = tuple(static_argnames)
        if not set(self.static_argnames) <= set(inspect.signature(function).parameters):
            raise ValueError("static_argnames contains an unknown parameter")
        if not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        self.python_function = function
        self._signature = inspect.signature(function)
        self._parameter_names = tuple(self._signature.parameters)
        self._fast_positional = all(
            parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            for parameter in self._signature.parameters.values()
        )
        self.max_size = cache_size
        self._plans, self._lock = OrderedDict(), RLock()
        self._hits = self._misses = 0
        update_wrapper(self, function)

    @property
    def padding(self):
        return self._padding

    def _plan(self, args, kwargs):
        if self._fast_positional and not kwargs and len(args) == len(self._parameter_names):
            bindings = args
            arguments = zip(self._parameter_names, bindings)
        else:
            bound = _bind(self.python_function, args, kwargs, self._signature)
            bindings = tuple(bound.arguments.values())
            arguments = bound.arguments.items()
        key = []
        for name, value in arguments:
            if name in self.static_argnames:
                key.append((name, "static", _static_key(value)))
            elif isinstance(value, ndarray):
                if getattr(value._backend, "_is_tracing", False):
                    raise TypeError("Nested compiled calls are not supported")
                key.append((name, "array", value.shape, value.dtype.str, value._backend))
            elif isinstance(value, (int, float, bool, np.generic)):
                key.append((name, "scalar", type(value), np.asarray(value).dtype.str))
            else:
                raise TypeError(
                    f"Unsupported dynamic argument {name!r}; use static_argnames for metadata"
                )
        key = tuple(key)
        with self._lock:
            if key in self._plans:
                self._hits += 1
                self._plans.move_to_end(key)
                return self._plans[key], bindings
            plan, bindings = capture_call(
                self.python_function,
                args,
                kwargs,
                static_argnames=self.static_argnames,
                padding=self.padding,
            )
            self._plans[key] = plan
            self._misses += 1
            if len(self._plans) > self.max_size:
                self._plans.popitem(last=False)
            return plan, bindings

    def __call__(self, *args, **kwargs):
        # Native NumPy code remains native when no OpenPacho array is supplied.
        if not any(isinstance(a, ndarray) for a in (*args, *kwargs.values())):
            return self.python_function(*args, **kwargs)
        plan, bindings = self._plan(args, kwargs)
        return plan.execute(bindings)

    def explain(self, *args, **kwargs):
        plan, _ = self._plan(args, kwargs)
        return plan.explain()

    def prepare(self, *args, **kwargs):
        """Specialize once; the returned callable validates each replacement binding.

        Does not retain example arrays, freeze dynamic scalars, or reuse exposed
        output storage. A shape/dtype/backend/static mismatch is an explicit error.
        """
        plan, bindings = self._plan(args, kwargs)
        return PreparedCall(self, plan, bindings)

    def cache_info(self):
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._plans),
                "max_size": self.max_size,
            }

    def clear_cache(self):
        with self._lock:
            self._plans.clear()
            self._hits = self._misses = 0


class PreparedCall:
    def __init__(self, compiled, plan, bindings):
        self._plan = plan
        self._signature = compiled._signature
        self._fast_positional = all(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in self._signature.parameters.values()
        )
        self._specs = []
        for name, value in zip(self._signature.parameters, bindings):
            if name in compiled.static_argnames:
                spec = ("static", _static_key(value))
            elif isinstance(value, ndarray):
                spec = ("array", value.shape, value.dtype, value._backend)
            else:
                spec = ("scalar", type(value), np.asarray(value).dtype)
            self._specs.append((name, spec))
        update_wrapper(self, compiled.python_function)

    def __call__(self, *args, **kwargs):
        if not kwargs and self._fast_positional and len(args) == len(self._specs):
            bindings = args
        else:
            bindings = tuple(
                _bind(self.__wrapped__, args, kwargs, self._signature).arguments.values()
            )
        for value, (name, spec) in zip(bindings, self._specs):
            if spec[0] == "array":
                valid = (
                    isinstance(value, ndarray)
                    and value.shape == spec[1]
                    and value.dtype == spec[2]
                    and value._backend is spec[3]
                )
            elif spec[0] == "static":
                valid = _static_key(value) == spec[1]
            else:
                valid = type(value) is spec[1] and np.asarray(value).dtype == spec[2]
            if not valid:
                raise ValueError(f"Prepared argument {name!r} does not match its specialization")
        return self._plan.execute(bindings)

    def explain(self):
        return dict(self._plan.explain(), prepared=True, binding_validation=True)


def compile(function=None, *, static_argnames=(), cache_size=32, padding=False):
    """Compile a pure array function; supports compile(f), @compile and @compile().

    NumPy inputs execute the original function. OpenPacho inputs select their
    existing backend. Cache hits rebind arrays and dynamic scalars without tracing.
    Shape/axis options must be declared static; no host reads, mutation or dynamic
    Python control flow. No implicit CPU fallback on capture/execution failure.
    padding permits private tile-aligned buffers for supported custom GEMM;
    logical shapes and math policy stay unchanged. Other operations ignore it.
    """
    padding = validate_padding(padding)

    def decorate(fn):
        return CompiledFunction(fn, static_argnames, cache_size, padding=padding)

    return decorate if function is None else decorate(function)
