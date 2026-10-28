"""Layout preparation and lifetime contracts, exercised without a GPU."""

import gc
import weakref

import numpy as np
import pytest

import openpacho as p
from openpacho._compiler import ExecutionPlan
from openpacho._fusion import Expression, InputRef, TraceBackend
from openpacho._opencl import OpenCLBackend
from openpacho.backends import NumpyBackend


class Provider:
    def __init__(self):
        self.calls = []

    def matmul(self, queue, a, b, out, shape, dtype, *, transpose_a, transpose_b):
        self.calls.append((a.shape, b.shape, transpose_a, transpose_b))
        left, right = a._data.reshape(a.shape), b._data.reshape(b.shape)
        out[:] = ((left.T if transpose_a else left) @ (right.T if transpose_b else right)).ravel()


class LayoutBackend(NumpyBackend):
    # Deliberately not named opencl: compiler routing must use the capability.
    name = "test-layout"
    queue = None
    prepare_layout_matmul = OpenCLBackend.prepare_layout_matmul

    def __init__(self):
        self._blas = Provider()
        self.gathers = 0

    def gather(self, *args):
        self.gathers += 1
        return super().gather(*args)

    def download(self, array):
        raise AssertionError("execution must not transfer to host")


@pytest.mark.parametrize("ta,tb", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_private_transpose_rebinding_and_no_transfers(ta, tb, dtype):
    backend = LayoutBackend()
    rng = np.random.default_rng(42)
    left = rng.normal(size=(7, 5) if ta else (5, 7)).astype(dtype)
    right = rng.normal(size=(9, 7) if tb else (7, 9)).astype(dtype)
    a, b = p.array(left, backend=backend), p.array(right, backend=backend)

    @p.compile
    def product(a, b):
        return (a.T if ta else a) @ (b.T if tb else b)

    prepared = product.prepare(a, b)
    assert [s["kind"] for s in prepared.explain()["steps"]] == ["layout_matmul"]
    layout = prepared.explain()["steps"][0]["layout"]
    assert layout["leading_dimensions"] == (left.shape[1], right.shape[1], 9)
    for scale in (1, 2):
        current = p.array(left * scale, backend=backend)
        result = prepared(current, b)
        expected = ((left * scale).T if ta else left * scale) @ (right.T if tb else right)
        np.testing.assert_allclose(result._data.reshape(result.shape), expected, rtol=1e-6)
    assert backend.gathers == 0
    assert backend._blas.calls == [(left.shape, right.shape, ta, tb)] * 2
    refs = weakref.ref(a), weakref.ref(b)
    del a, b
    gc.collect()
    assert all(ref() is None for ref in refs)
    with pytest.raises(ValueError, match="specialization"):
        prepared(
            current,
            p.array(
                right.astype(np.float64 if dtype == np.float32 else np.float32), backend=backend
            ),
        )


@pytest.mark.parametrize("public", [False, True])
def test_shared_or_public_transpose_remains_materialized(public):
    backend = LayoutBackend()
    host = np.arange(35, dtype=np.float32).reshape(7, 5)
    a, b = p.array(host, backend=backend), p.array(host, backend=backend)

    def source(a, b):
        t = a.T
        return (t @ b, t) if public else (t @ b, t + 1)

    fn = p.compile(source)
    actual = fn(a, b)
    assert [s["kind"] for s in fn.explain(a, b)["steps"]].count("gather") == 1
    assert not backend._blas.calls
    assert backend.gathers == 1
    expected = source(host, host)
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result._data.reshape(result.shape), reference)


@pytest.mark.parametrize(
    "offset,strides,shape", [(1, (1, 5), (5, 7)), (0, (7, 1), (5, 7)), (0, (1, 5), (4, 7))]
)
def test_non_full_transpose_gather_is_not_folded(offset, strides, shape):
    backend = LayoutBackend()
    trace = TraceBackend(backend)
    original = Expression(trace, (7, 5), np.float32, value=InputRef(0))
    gather = trace.gather(original, shape, offset, strides)
    right = Expression(trace, (7, 9), np.float32, value=InputRef(1))
    product = trace.matmul(gather, right, (shape[0], 9), np.float32)
    plan = ExecutionPlan(backend, (product,), False)
    assert [s[1] for s in plan.steps] == ["gather", "matmul"]


@pytest.mark.parametrize("ashape,bshape", [((0, 5), (0, 9)), ((7, 0), (7, 9)), ((7, 5), (7, 0))])
def test_empty_dimensions_keep_safe_fallback(ashape, bshape):
    backend = LayoutBackend()
    a = p.array(np.zeros(ashape, dtype=np.float32), backend=backend)
    b = p.array(np.zeros(bshape, dtype=np.float32), backend=backend)
    fn = p.compile(lambda a, b: a.T @ b)
    result = fn(a, b)
    assert not backend._blas.calls
    np.testing.assert_array_equal(
        result._data.reshape(result.shape), np.zeros(ashape).T @ np.zeros(bshape)
    )


def test_missing_or_declined_capability_preserves_copy():
    for backend in (NumpyBackend(), LayoutBackend()):
        if isinstance(backend, LayoutBackend):
            backend._blas = None
        a = p.array(np.ones((7, 5), dtype=np.float32), backend=backend)
        fn = p.compile(lambda a: a.T @ a)
        assert [s["kind"] for s in fn.explain(a)["steps"]] == ["gather", "matmul"]


def test_provider_error_propagates_without_materialized_fallback():
    backend = LayoutBackend()
    a = p.array(np.ones((7, 5), dtype=np.float32), backend=backend)
    prepared = p.compile(lambda a: a.T @ a).prepare(a)

    def fail(*args, **kwargs):
        raise RuntimeError("CLBlast status code -101")

    backend._blas.matmul = fail
    with pytest.raises(RuntimeError, match="CLBlast status code -101"):
        prepared(a)
    assert backend.gathers == 0


def test_one_private_transpose_used_by_both_operands():
    backend = LayoutBackend()
    host = np.arange(25, dtype=np.float32).reshape(5, 5)
    a = p.array(host, backend=backend)

    def source(a):
        t = a.T
        return t @ t

    result = p.compile(source)(a)
    np.testing.assert_array_equal(result._data.reshape(result.shape), host.T @ host.T)
    assert backend.gathers == 0


def test_mixed_dtype_keeps_required_conversion():
    backend = LayoutBackend()
    a = p.array(np.arange(35, dtype=np.float32).reshape(7, 5), backend=backend)
    b = p.array(np.arange(63, dtype=np.float64).reshape(7, 9), backend=backend)
    fn = p.compile(lambda a, b: a.T @ b)
    result = fn(a, b)
    assert result.dtype == np.dtype("float64")
    np.testing.assert_array_equal(
        result._data.reshape(result.shape), a._data.reshape(a.shape).T @ b._data.reshape(b.shape)
    )
    assert [s["kind"] for s in fn.explain(a, b)["steps"]] == ["gather", "cast", "matmul"]


def test_eager_transpose_still_materializes_independent_storage():
    backend = LayoutBackend()
    a = p.array(np.arange(35, dtype=np.float32).reshape(7, 5), backend=backend)
    t = a.T
    assert backend.gathers == 1
    assert not np.shares_memory(t._data, a._data)


def test_explicit_padding_precedes_layout():
    backend = LayoutBackend()
    calls = []

    class Padded:
        def run(self, a, b):
            calls.append("padding")
            return backend.matmul(a, b, (5, 5), a.dtype)

    backend.prepare_padded_matmul = lambda *args: Padded()
    a = p.array(np.ones((7, 5), dtype=np.float32), backend=backend)
    fn = p.compile(lambda a: a.T @ a, padding=True)
    fn(a)
    assert calls == ["padding"]
    assert backend.gathers == 1
    assert not backend._blas.calls
