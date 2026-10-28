"""CPU-only checks for the paired three-actions benchmark setup."""

from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np
import pytest

from benchmarks import three_actions as runner
from openpacho import _compiler
from openpacho.backends import NumpyBackend


def test_controls_select_modes_and_restore_the_compiler_on_cpu():
    current_plan = _compiler.CompiledFunction._plan

    def old_plan(*args, **kwargs):
        raise AssertionError("the historical plan should not execute in this test")

    control = object.__new__(runner.Controls)
    control.backend = NumpyBackend()
    control.current_plan = current_plan
    control.old_plan = old_plan
    control.cuda = False
    control.saved = {}

    try:
        for mode, expected in (
            ("baseline", old_plan),
            ("binding", current_plan),
            ("scope", old_plan),
            ("gemv", old_plan),
            ("all", current_plan),
        ):
            control.activate(mode)
            assert _compiler.CompiledFunction._plan is expected

        with pytest.raises(ValueError, match="unknown"):
            control.activate("unknown")
    finally:
        control.restore()

    assert _compiler.CompiledFunction._plan is current_plan


def test_benchmark_cases_keep_cpu_subset_and_custom_gemv_cases():
    all_cases = runner.cases()
    cpu_cases = runner.benchmark_cases("numpy")
    custom_cases = runner.benchmark_cases("cuda-custom")
    cublas_cases = runner.benchmark_cases("cuda-cublas")

    assert len(cpu_cases) == 12
    assert {case[0] for case in cpu_cases} == {"chain", "row_energy", "loss", "ridge"}
    assert all(case in all_cases for case in cpu_cases)
    assert custom_cases[: len(all_cases)] == all_cases
    assert len(custom_cases) == 42
    assert any(case[0] == "matvec" for case in custom_cases)
    assert len(cublas_cases) == len(all_cases)
    assert all(case[0] != "matvec" for case in cublas_cases)


@pytest.mark.parametrize(
    "operation,shape",
    (("chain", (7,)), ("row_energy", (3, 4)), ("loss", (3, 4)), ("ridge", (8, 3))),
)
def test_build_call_supports_numpy_backend_without_gpu(operation, shape):
    backend = NumpyBackend()
    adapter = runner.PachoAdapter("numpy", backend)
    call, complete, expected, reference = runner.build_call(
        adapter, operation, shape, "float32", seed=19
    )

    actual = adapter.host(call())
    complete_actual = complete()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(complete_actual, expected, rtol=2e-5, atol=2e-6)
    assert reference.dtype == np.dtype("float64")


def test_build_call_uses_separate_plans_with_the_backend_execution_scope(monkeypatch):
    backend = NumpyBackend()
    adapter = runner.PachoAdapter("numpy", backend)
    events = []

    @contextmanager
    def first_scope():
        events.append("first-enter")
        yield
        events.append("first-exit")

    @contextmanager
    def second_scope():
        events.append("second-enter")
        yield
        events.append("second-exit")

    compiled = []
    original_compile = runner.p.compile

    def record_compile(function):
        value = original_compile(function)
        compiled.append(value)
        return value

    monkeypatch.setattr(runner.p, "compile", record_compile)

    backend.execution_scope = first_scope
    first_call, _, first_expected, _ = runner.build_call(adapter, "chain", (7,), "float32", seed=19)
    np.testing.assert_allclose(adapter.host(first_call()), first_expected, rtol=2e-5, atol=2e-6)

    backend.execution_scope = second_scope
    second_call, _, second_expected, _ = runner.build_call(
        adapter, "chain", (7,), "float32", seed=19
    )
    np.testing.assert_allclose(adapter.host(second_call()), second_expected, rtol=2e-5, atol=2e-6)

    assert len(compiled) == 2
    assert compiled[0] is not compiled[1]
    first_plan = next(iter(compiled[0]._plans.values()))
    second_plan = next(iter(compiled[1]._plans.values()))
    assert first_plan is not second_plan
    assert first_plan._execution_scope is first_scope
    assert second_plan._execution_scope is second_scope
    assert events == ["first-enter", "first-exit", "second-enter", "second-exit"]


def _patch_tiny_run(monkeypatch, backend, wheel):
    controls = []

    class NoopControls:
        def __init__(self, selected_backend, wheel):
            assert selected_backend is backend
            assert wheel.exists()
            self.cuda = False
            self.modes = []
            self.restored = False
            controls.append(self)

        def activate(self, mode):
            self.modes.append(mode)

        def restore(self):
            self.restored = True

    monkeypatch.setattr(runner, "Controls", NoopControls)
    monkeypatch.setattr(runner, "engine", lambda name: (np, backend))
    monkeypatch.setattr(runner, "benchmark_cases", lambda name: [("chain", (7,), "float32")])
    monkeypatch.setattr(
        runner,
        "warmup",
        lambda function, xp, selected_backend: {"calls": 0, "seconds": 0.0},
    )

    def fast_measure(function, xp, selected_backend):
        activate = getattr(function, "_activate_variant", None)
        if activate is not None:
            activate()
        return 1e-6, 1e-6, function()

    monkeypatch.setattr(runner, "measure", fast_measure)
    return controls


def test_run_writes_report_with_backend_without_close(monkeypatch, tmp_path):
    backend = NumpyBackend()
    sync_calls = []
    backend.synchronize = lambda: sync_calls.append("synchronize")
    wheel = tmp_path / "before.whl"
    wheel.write_bytes(b"tiny-control-wheel")
    controls = _patch_tiny_run(monkeypatch, backend, wheel)
    output = tmp_path / "result.json"

    report = runner.run(output, ["numpy"], repeats=3, control_wheel=wheel)

    assert report["complete"] is True
    assert report["backend"] == "numpy"
    assert len(report["rows"]) == 2
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["complete"] is True
    assert controls[0].restored
    assert sync_calls == ["synchronize"]
    assert not hasattr(backend, "close")


def test_run_preserves_report_when_backend_cleanup_fails(monkeypatch, tmp_path):
    backend = NumpyBackend()

    def fail_synchronize():
        raise RuntimeError("cleanup failed")

    backend.synchronize = fail_synchronize
    wheel = tmp_path / "before.whl"
    wheel.write_bytes(b"tiny-control-wheel")
    controls = _patch_tiny_run(monkeypatch, backend, wheel)
    output = tmp_path / "result.json"

    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.run(output, ["numpy"], repeats=3, control_wheel=wheel)

    assert controls[0].restored
    assert output.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["complete"] is True
    assert len(report["rows"]) == 2
