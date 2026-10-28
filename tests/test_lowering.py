"""Backend extension contracts without a GPU runtime or GPU device name."""

import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

import openpacho as pacho
from openpacho import _lowering
from openpacho._array import ndarray
from openpacho.backends import NumpyBackend


class ExampleBackend(NumpyBackend):
    name = "example-device"


def test_factory_errors_propagate_and_names_do_not_select_lowering():
    assert _lowering.create_lowering(SimpleNamespace(name="opencl")) is None

    def broken_factory():
        raise RuntimeError("lowering preparation failed")

    with pytest.raises(RuntimeError, match="preparation failed"):
        _lowering.create_lowering(SimpleNamespace(create_lowering=broken_factory))


def test_execution_scope_is_bound_once_and_exits_on_failure():
    events = []

    class ScopedBackend(ExampleBackend):
        @contextmanager
        def execution_scope(self):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

    backend = ScopedBackend()
    array = pacho.asarray(np.arange(3, dtype=np.float32), backend=backend)
    operation = pacho.compile(lambda a: a + 1)
    np.testing.assert_array_equal(pacho.asnumpy(operation(array)), [1, 2, 3])
    backend.execution_scope = lambda: pytest.fail("scope must be bound during preparation")
    operation(array)

    def fail(*args):
        raise RuntimeError("execution failed")

    backend.elementwise = fail
    with pytest.raises(RuntimeError, match="execution failed"):
        operation(array)
    assert events == ["enter", "exit"] * 3


class ExampleLowering:
    """CPU interpreter standing in for an independently implemented lowerer."""

    def __init__(self, backend):
        self.backend = backend
        self.results = []
        self.scalar_dtypes = []

    def can_residual_matvec(self, matrix, vector, rhs, node):
        return False

    def prepare_fused(self, node, group, leaves):
        return tuple(leaves), tuple(group)

    prepare_map_reduce = prepare_fused

    def _map(self, group, arguments, operands):
        values = {id(arg): value for arg, value in zip(arguments, operands)}
        for value in operands:
            if getattr(value, "_host_scalar", False):
                self.scalar_dtypes.append(value.dtype)
        for expression in group:
            args = []
            for child, dtype in zip(expression.inputs, expression.resolved):
                value = values[id(child)]
                data = np.asarray(value._data, dtype=dtype).reshape(-1)
                args.append(ndarray._wrap(data, value.shape, dtype, self.backend))
            data = self.backend.elementwise(expression.op, args, expression.shape, expression.dtype)
            values[id(expression)] = ndarray._wrap(
                data, expression.shape, expression.dtype, self.backend
            )
        return values[id(group[-1])]

    def run_fused(self, node, payload, arguments, operands):
        data = self._map(payload, arguments, operands)._data
        self.results.append(weakref.ref(data))
        return data

    def run_map_reduce(self, node, payload, arguments, operands):
        mapped = self._map(payload, arguments, operands)
        return self.backend.reduce(mapped, node.attrs["axes"], node.attrs["keepdims"], node.dtype)


def test_independent_lowerer_rebinds_and_preserves_shared_outputs(monkeypatch):
    monkeypatch.setattr(
        ExampleBackend, "create_lowering", lambda self: ExampleLowering(self), raising=False
    )
    backend = ExampleBackend()

    @pacho.compile
    def operation(a, rate):
        shared = a + rate
        return shared, np.sum(shared * shared), shared - 1

    host = np.arange(12, dtype=np.float32).reshape(3, 4)
    array = pacho.asarray(host, backend=backend)
    array_ref = weakref.ref(array)
    first = operation(array, 2.0)
    plan = next(iter(operation._plans.values()))
    assert [step["kind"] for step in plan.explain()["steps"]] == ["fused", "map_reduce", "fused"]
    second = operation(pacho.asarray(host + 5, backend=backend), 3.0)
    for actual, expected in zip(first, (host + 2, np.sum((host + 2) ** 2), host + 1)):
        np.testing.assert_array_equal(pacho.asnumpy(actual), expected)
    for actual, expected in zip(second, (host + 8, np.sum((host + 8) ** 2), host + 7)):
        np.testing.assert_array_equal(pacho.asnumpy(actual), expected)
    assert operation.cache_info()["misses"] == 1
    assert all(dtype == np.dtype("float32") for dtype in plan.lowering.scalar_dtypes)
    assert plan.lowering.scalar_dtypes
    del array, first, second, actual
    gc.collect()
    assert array_ref() is None
    assert all(result() is None for result in plan.lowering.results)


def test_unregistered_backend_keeps_ordinary_primitives():
    backend = ExampleBackend()
    array = pacho.asarray(np.arange(5, dtype=np.float32), backend=backend)
    operation = pacho.compile(lambda a: np.sum((a + 2) * a))
    np.testing.assert_array_equal(pacho.asnumpy(operation(array)), 50)
    assert next(iter(operation._plans.values())).lowering is None
    assert [step["kind"] for step in operation.explain(array)["steps"]] == [
        "add",
        "multiply",
        "reduce",
    ]
    assert _lowering.create_lowering(NumpyBackend()) is None


@pytest.mark.parametrize("name,expected", [("opencl", True), ("vulkan", False)])
def test_residual_capability_preserves_partitioned_dot_boundary(name, expected):
    backend = SimpleNamespace(name=name, blas="custom")
    implementation = _lowering.OpenCLLowering if name == "opencl" else _lowering.VulkanLowering
    backend.create_lowering = lambda: implementation(backend)
    lowerer = _lowering.create_lowering(backend)
    dtype = np.dtype("float32")
    matrix = SimpleNamespace(ndim=2, shape=(1, 4097), dtype=dtype)
    vector = SimpleNamespace(ndim=1, shape=(4097,), dtype=dtype)
    rhs = SimpleNamespace(ndim=1, shape=(1,), dtype=dtype)
    node = SimpleNamespace(inputs=(rhs, rhs), shape=(1,), dtype=dtype)
    assert lowerer.can_residual_matvec(matrix, vector, rhs, node) is expected
    backend.blas = "clblast"
    assert not lowerer.can_residual_matvec(matrix, vector, rhs, node)


@pytest.mark.parametrize("operation,expected", [("sum", [3, 7]), ("min", [1, 3]), ("max", [2, 4])])
def test_opencl_lowerer_retains_partial_reduction_shape_and_scalar_abi(operation, expected):
    class RecordingBackend(NumpyBackend):
        name = "opencl"

        def create_lowering(self):
            return _lowering.OpenCLLowering(self)

        def _run(self, source, count, *args, local_size=None):
            assert source == "prepared kernel"
            assert count == 256
            assert local_size == 128
            assert args[0].dtype == np.dtype("float32")
            assert args[0] == np.float32(3.25)
            args[-1][:] = [1, 2, 3, 4]

    from openpacho._ops import HostScalar

    backend = RecordingBackend()
    lowerer = _lowering.create_lowering(backend)
    node = SimpleNamespace(
        shape=(2, 1),
        dtype=np.dtype("float32"),
        attrs={} if operation == "sum" else {"operation": operation},
    )
    payload = ("prepared kernel", 2, 2, 256, 128)
    actual = lowerer.run_map_reduce(node, payload, (), [HostScalar(3.25, np.float32)])
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == node.dtype
