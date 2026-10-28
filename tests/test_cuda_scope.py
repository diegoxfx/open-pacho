"""Nested CUDA activation must honor the actual per-thread driver context."""

from concurrent.futures import ThreadPoolExecutor
from threading import RLock, local
from types import SimpleNamespace

import pytest

from openpacho._cuda import CUDABackend


def fake_backend():
    backend = object.__new__(CUDABackend)
    backend._lock = RLock()
    backend._closed = False
    backend._context = 47
    state = local()
    events = []

    def stack():
        if not hasattr(state, "stack"):
            state.stack = [99]
        return state.stack

    def push(context):
        events.append(("push", context))
        stack().append(context)
        return (0,)

    def pop():
        events.append(("pop", stack()[-1]))
        return (0, stack().pop())

    def current():
        events.append(("query", stack()[-1]))
        return (0, stack()[-1])

    backend.driver = SimpleNamespace(
        cuCtxPushCurrent=push, cuCtxPopCurrent=pop, cuCtxGetCurrent=current
    )
    return backend, stack, events


def test_nested_scope_restores_context_on_exception_without_repeated_pushes():
    backend, stack, events = fake_backend()
    try:
        with (
            pytest.raises(ValueError, match="operation failed"),
            backend.execution_scope(),
            backend._activate(),
            backend._activate(),
        ):
            assert stack() == [99, 47]
            raise ValueError("operation failed")
        assert stack() == [99]
        assert backend._activation_depth == 0
        assert [e[0] for e in events] == ["push", "query", "query", "pop"]
        with backend._activate():
            assert stack()[-1] == 47
        assert stack() == [99]
    finally:
        backend._closed = True


def test_nested_foreign_context_is_temporarily_overridden_and_restored():
    backend, stack, events = fake_backend()
    try:
        with backend.execution_scope():
            backend.driver.cuCtxPushCurrent(71)
            with backend._activate():
                assert stack() == [99, 47, 71, 47]
            assert stack() == [99, 47, 71]
            backend.driver.cuCtxPopCurrent()
        assert stack() == [99]
        assert backend._activation_depth == 0
        assert events.count(("query", 71)) == 1
    finally:
        backend._closed = True


@pytest.mark.parametrize("failure", ["push", "query", "pop"])
def test_driver_errors_leave_depth_and_lock_usable(failure):
    backend, stack, _ = fake_backend()
    original = getattr(
        backend.driver,
        "cuCtx" + {"push": "PushCurrent", "query": "GetCurrent", "pop": "PopCurrent"}[failure],
    )
    name = "cuCtx" + {"push": "PushCurrent", "query": "GetCurrent", "pop": "PopCurrent"}[failure]
    try:
        if failure == "query":
            with backend.execution_scope():
                setattr(backend.driver, name, lambda: (1,))
                with pytest.raises(RuntimeError, match="CUDA call failed"), backend._activate():
                    pytest.fail("failed query must not execute body")
                assert backend._activation_depth == 1
                setattr(backend.driver, name, original)
        else:
            setattr(backend.driver, name, lambda *args: (1,))
            with pytest.raises(RuntimeError, match="CUDA call failed"), backend._activate():
                assert failure == "pop"
            setattr(backend.driver, name, original)
            # A failed native pop may leave its frame in the driver. Do not
            # claim it was restored, and clean up the fake before retrying.
            if failure == "pop":
                original()
        assert backend._activation_depth == 0
        with backend._activate():
            assert stack()[-1] == 47
        assert stack() == [99]
    finally:
        backend._closed = True


def test_scope_depth_is_serialized_with_per_thread_contexts():
    backend, stack, _ = fake_backend()
    try:

        def work(_):
            for _ in range(40):
                with backend.execution_scope(), backend._activate():
                    assert stack() == [99, 47]
                assert stack() == [99]

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(work, range(6)))
        assert backend._activation_depth == 0
        backend._closed = True
        with pytest.raises(RuntimeError, match="closed"), backend.execution_scope():
            pass
    finally:
        backend._closed = True
