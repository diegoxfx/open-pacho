"""CPU coverage for the ordinary compiled-call binding fast path."""

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager

import numpy as np
import pytest

import openpacho as pacho
import openpacho._compiler as compiler_module
from openpacho.backends import NumpyBackend


def _array(value, backend="numpy", dtype=None):
    return pacho.asarray(value, dtype=dtype, backend=backend)


def test_full_positional_cache_hit_skips_signature_bind(monkeypatch):
    def affine(a, scale):
        return a * scale + 1

    compiled = pacho.compile(affine)
    value = np.arange(6, dtype=np.float32)
    array = _array(value)
    original_bind = compiler_module._bind
    bind_calls = []

    def track_bind(*args, **kwargs):
        bind_calls.append((args, kwargs))
        return original_bind(*args, **kwargs)

    monkeypatch.setattr(compiler_module, "_bind", track_bind)
    np.testing.assert_allclose(pacho.asnumpy(compiled(array, 2.0)), value * 2 + 1)
    np.testing.assert_allclose(pacho.asnumpy(compiled(array, 3.0)), value * 3 + 1)

    # The miss still binds while capturing the function; the equivalent
    # positional cache hit constructs its bindings directly from args.
    assert len(bind_calls) == 1
    assert compiled.cache_info() == {"hits": 1, "misses": 1, "size": 1, "max_size": 32}


def test_kwargs_and_defaults_reuse_plan_and_rebind_scalars():
    def affine(a, scale=2.0):
        return a * scale + 1

    compiled = pacho.compile(affine)
    value = np.arange(5, dtype=np.float32)
    array = _array(value)

    np.testing.assert_allclose(pacho.asnumpy(compiled(array, scale=2.0)), value * 2 + 1)
    np.testing.assert_allclose(pacho.asnumpy(compiled(array, scale=3.0)), value * 3 + 1)
    np.testing.assert_allclose(pacho.asnumpy(compiled(array)), value * 2 + 1)

    assert compiled.cache_info() == {"hits": 2, "misses": 1, "size": 1, "max_size": 32}


def test_positional_only_binding_and_errors_are_preserved():
    def affine(a, scale=2.0, /):
        return a * scale + 1

    compiled = pacho.compile(affine)
    value = np.arange(4, dtype=np.float32)
    array = _array(value)
    np.testing.assert_allclose(pacho.asnumpy(compiled(array, 3.0)), value * 3 + 1)

    with pytest.raises(TypeError, match="positional-only"):
        compiled(array, scale=3.0)


def test_keyword_only_and_variadic_signature_errors_are_preserved():
    def keyword_only(a, *, scale=2.0):
        return a * scale + 1

    compiled = pacho.compile(keyword_only)
    value = np.arange(4, dtype=np.float32)
    array = _array(value)
    np.testing.assert_allclose(pacho.asnumpy(compiled(array, scale=3.0)), value * 3 + 1)
    np.testing.assert_allclose(pacho.asnumpy(compiled(array)), value * 2 + 1)

    with pytest.raises(TypeError):
        compiled(array, 3.0)

    def variadic(a, *rest):
        return a

    with pytest.raises(TypeError, match="without \\*args or \\*\\*kwargs"):
        pacho.compile(variadic)(array)


def test_static_values_and_shape_dtype_backend_remain_specialization_keys():
    def reshape_to(a, shape):
        return np.reshape(a, shape)

    compiled = pacho.compile(reshape_to, static_argnames="shape")
    value = np.arange(6, dtype=np.float32)
    backend_a = NumpyBackend()
    backend_b = NumpyBackend()
    first = _array(value, backend_a)
    same = _array(value.copy(), backend_a)
    different_dtype = _array(value, backend_a, dtype=np.float64)
    different_backend = _array(value, backend_b)

    np.testing.assert_array_equal(pacho.asnumpy(compiled(first, (2, 3))), value.reshape(2, 3))
    np.testing.assert_array_equal(pacho.asnumpy(compiled(same, (2, 3))), value.reshape(2, 3))
    np.testing.assert_array_equal(pacho.asnumpy(compiled(first, (3, 2))), value.reshape(3, 2))
    np.testing.assert_array_equal(
        pacho.asnumpy(compiled(different_dtype, (2, 3))), value.astype(np.float64).reshape(2, 3)
    )
    np.testing.assert_array_equal(
        pacho.asnumpy(compiled(different_backend, (2, 3))), value.reshape(2, 3)
    )

    assert compiled.cache_info() == {"hits": 1, "misses": 4, "size": 4, "max_size": 32}


def test_compiled_plan_does_not_retain_input_arrays():
    compiled = pacho.compile(lambda a: a * 2 + 1)
    source = _array(np.arange(7, dtype=np.float32))
    source_ref = weakref.ref(source)
    result = compiled(source)
    del source
    gc.collect()

    assert source_ref() is None
    np.testing.assert_allclose(pacho.asnumpy(result), np.arange(7, dtype=np.float32) * 2 + 1)


def test_execution_scope_threshold_is_bound_once_without_backend_name_checks():
    backend = NumpyBackend()
    events = []

    @contextmanager
    def scope():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    backend.execution_scope = scope
    backend.execution_scope_min_steps = 2
    a = _array(np.arange(7, dtype=np.float32), backend)
    for fn in (lambda a: a, lambda a: a + 1):
        compiled = pacho.compile(fn)
        compiled(a)
        assert next(iter(compiled._plans.values()))._execution_scope is None
    assert events == []
    compiled = pacho.compile(lambda a: (a + 1) * 2)
    np.testing.assert_array_equal(pacho.asnumpy(compiled(a)), (np.arange(7) + 1) * 2)
    backend.execution_scope_min_steps = 99
    compiled(a)
    assert events == ["enter", "exit"] * 2
    assert compiled.cache_info()["hits"] == 1
