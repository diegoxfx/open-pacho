"""Materialized reduction planning contracts, without initializing a GPU."""

import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

import openpacho as p
from openpacho._array import ndarray
from openpacho._cuda_lowering import CUDALowering
from openpacho._lowering import OpenCLLowering
from openpacho.backends import NumpyBackend


class Interpreter:
    """An external lowering implementation, deliberately without the new hook."""

    def __init__(self, backend):
        self.backend = backend
        self.requests = []
        self.storage = []
        self.reduction_inputs = []

    def can_residual_matvec(self, *args):
        return False

    def prepare_fused(self, node, group, leaves):
        return tuple(leaves), tuple(group)

    def prepare_map_reduce(self, node, group, leaves):
        self.requests.append((node, tuple(group), tuple(leaves)))
        return tuple(leaves), tuple(group)

    def _map(self, node, group, arguments, operands):
        values = dict(zip(map(id, arguments), operands))
        for expression in group:
            args = []
            for child, dtype in zip(expression.inputs, expression.resolved):
                value = values[id(child)]
                data = np.asarray(value._data, dtype=dtype).reshape(-1)
                args.append(ndarray._wrap(data, value.shape, dtype, self.backend))
            data = self.backend.elementwise(expression.op, args, expression.shape, expression.dtype)
            self.storage.append(weakref.ref(data))
            values[id(expression)] = ndarray._wrap(
                data, expression.shape, expression.dtype, self.backend
            )
        return values[id(node)]

    def run_fused(self, node, payload, arguments, operands):
        return self._map(node, payload, arguments, operands)._data

    def run_map_reduce(self, node, payload, arguments, operands):
        mapped = self._map(node.inputs[0], payload, arguments, operands)
        self.reduction_inputs.append(id(mapped._data))
        return self.backend.reduce(
            mapped,
            node.attrs["axes"],
            node.attrs["keepdims"],
            node.dtype,
            operation=node.attrs.get("operation", "sum"),
        )


def cpu_backend(policy=None, *, name="external"):
    class Lowerer(Interpreter):
        pass

    if policy is not None:
        Lowerer.can_reduce_materialized = policy

    class Backend(NumpyBackend):
        def create_lowering(self):
            return Lowerer(self)

    backend = Backend()
    backend.name = name
    return backend


@pytest.mark.parametrize("name", ("opencl", "cuda", "vulkan", "external"))
@pytest.mark.parametrize("opt_in", (False, True))
def test_external_capability_is_optional_and_independent_of_name(name, opt_in):
    # The extension chooses its own supported dtype. The compiler must not
    # substitute a backend name check or impose a built-in dtype policy.
    policy = (lambda self, node: True) if opt_in else None
    backend = cpu_backend(policy, name=name)
    host = np.arange(15, dtype=np.float64).reshape(3, 5)
    array = p.asarray(host, backend=backend)
    calculate = p.compile(lambda x: p.max(x, axis=1))
    np.testing.assert_array_equal(p.asnumpy(calculate(array)), host.max(axis=1))
    assert [step["kind"] for step in calculate.explain(array)["steps"]] == [
        "map_reduce" if opt_in else "reduce"
    ]


@pytest.mark.parametrize("implementation", (OpenCLLowering, CUDALowering))
@pytest.mark.parametrize(
    "dtype,operation,kind",
    [
        (np.float32, "min", "map_reduce"),
        (np.float32, "max", "map_reduce"),
        (np.float64, "min", "reduce"),
        (np.float64, "max", "reduce"),
        (np.int32, "max", "reduce"),
        (np.bool_, "min", "reduce"),
        (np.float32, "sum", "reduce"),
    ],
)
def test_native_capability_limits_bare_route_and_preserves_fusion(
    implementation, dtype, operation, kind
):
    backend = cpu_backend(implementation.can_reduce_materialized)
    host = np.arange(15).reshape(3, 5).astype(dtype)
    array = p.asarray(host, backend=backend)
    function = getattr(p, operation)
    bare = p.compile(lambda x: function(x, axis=1))
    mapped = p.compile(lambda x: function(p.where(x == 0, x, x), axis=1))
    for calculate, expected_kind in ((bare, kind), (mapped, "map_reduce")):
        np.testing.assert_array_equal(
            p.asnumpy(calculate(array)),
            getattr(np, operation)(
                host, axis=1, **({"dtype": dtype} if operation == "sum" else {})
            ),
        )
        assert [step["kind"] for step in calculate.explain(array)["steps"]] == [expected_kind]
    bare_plan = next(iter(bare._plans.values()))
    if kind == "map_reduce":
        node, group, leaves = bare_plan.lowering.requests[0]
        assert group == () and leaves == node.inputs
    assert next(iter(mapped._plans.values())).lowering.requests[0][1]


@pytest.mark.parametrize("returned", (False, True))
def test_shared_producer_is_materialized_once_and_rebound_without_retention(returned):
    backend = cpu_backend(OpenCLLowering.can_reduce_materialized)
    events, traces = [], []

    @contextmanager
    def scope():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    backend.execution_scope = scope
    backend.execution_scope_min_steps = 4

    @p.compile
    def calculate(x, rate):
        traces.append("trace")
        shared = x + rate
        results = (p.min(shared, axis=0), p.max(shared, axis=1), shared * 2)
        return (shared, *results) if returned else results

    host = np.arange(15, dtype=np.float32).reshape(3, 5)
    first_input = p.asarray(host, backend=backend)
    reference = weakref.ref(first_input)
    prepared = calculate.prepare(first_input, 2.0)
    plan = prepared._plan
    assert [step[1] for step in plan.steps] == ["fused", "map_reduce", "map_reduce", "fused"]
    for node, group, leaves in plan.lowering.requests:
        assert not group and leaves == node.inputs
        assert leaves[0] is plan.steps[0][0]
    backend.execution_scope = lambda: pytest.fail("cached scope must stay bound")

    first = calculate(first_input, 2.0)
    replacement = p.asarray(host * -3, backend=backend)
    second = prepared(x=replacement, rate=4.0)
    for actual, shared in ((first, host + 2), (second, host * -3 + 4)):
        expected = (shared.min(axis=0), shared.max(axis=1), shared * 2)
        if returned:
            expected = (shared, *expected)
        for result, want in zip(actual, expected):
            np.testing.assert_array_equal(p.asnumpy(result), want)
    assert traces == ["trace"]
    assert calculate.cache_info()["misses"] == 1
    assert events == ["enter", "exit"] * 2
    inputs = plan.lowering.reduction_inputs
    assert len(inputs) == 4 and inputs[0] == inputs[1] and inputs[2] == inputs[3]
    # Only add and multiply execute for each call, despite two reduction consumers.
    assert len(plan.lowering.storage) == 4
    del first_input, replacement, first, second, actual, result
    gc.collect()
    assert reference() is None
    assert all(storage() is None for storage in plan.lowering.storage)


def test_non_elementwise_input_keeps_its_materialization_step():
    backend = cpu_backend(CUDALowering.can_reduce_materialized)
    host = np.arange(15, dtype=np.float32).reshape(3, 5)
    array = p.asarray(host, backend=backend)
    calculate = p.compile(lambda x: p.max(x.T, axis=1, keepdims=True))
    np.testing.assert_array_equal(p.asnumpy(calculate(array)), host.T.max(axis=1, keepdims=True))
    plan = next(iter(calculate._plans.values()))
    assert [step[1] for step in plan.steps] == ["gather", "map_reduce"]
    node, group, leaves = plan.lowering.requests[0]
    assert group == () and leaves == node.inputs
    assert leaves[0] is plan.steps[0][0]


@pytest.mark.parametrize("implementation", (OpenCLLowering, CUDALowering))
def test_native_source_accepts_a_materialized_leaf_without_a_mapped_group(implementation):
    backend = cpu_backend(implementation.can_reduce_materialized)
    array = p.asarray(np.zeros((3, 8193), dtype=np.float32), backend=backend)
    prepared = p.compile(lambda x: p.max(x, axis=1)).prepare(array)
    node = prepared._plan.outputs[0]
    lowerer = implementation(SimpleNamespace(work_group_size=128))
    assert lowerer.can_reduce_materialized(node)
    arguments, payload = lowerer.prepare_map_reduce(node, (), node.inputs)
    assert arguments == node.inputs
    assert "x0[" in payload[0] and "v0" in payload[0]
    assert payload[1:3] == (3, 3)
