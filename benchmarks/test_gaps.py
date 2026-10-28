"""CPU-only checks for the gaps profiler's public contracts."""

from __future__ import annotations

import cProfile

import numpy as np

from benchmarks.gaps import (
    BACKEND_NAMES,
    CALL_NAMES,
    WORKLOADS,
    Adapter,
    CallCounter,
    Case,
    _expected,
    _inputs,
    _profile_top,
    _validate,
    build_parser,
)


def test_cli_defaults_are_reproducible():
    args = build_parser().parse_args(["--output", "gaps.json"])
    assert args.backends == list(BACKEND_NAMES)
    assert args.repeats == 7
    assert args.size == 2048


def test_workloads_are_the_four_diagnostic_primitives():
    assert WORKLOADS == ("gemv", "dot", "compiled_energy", "cg")


def test_call_counter_restores_methods_and_keeps_exact_logical_counts():
    class FakeBackend:
        def empty(self):
            return None

        def _run(self):
            return None

        def synchronize(self):
            return None

    backend = FakeBackend()
    original_empty = backend.empty
    counter = CallCounter.install(backend)
    backend.empty()
    backend._run()
    backend.synchronize()
    counter.uninstall(backend)
    assert set(counter.counts) == set(CALL_NAMES)
    assert counter.counts["empty"] == 1
    assert counter.counts["_run"] == 1
    assert counter.counts["synchronize"] == 1
    assert backend.empty.__func__ is original_empty.__func__


def test_cprofile_reports_own_and_cumulative_without_summing_nested_times():
    profile = cProfile.Profile()
    profile.enable()
    np.sum(np.arange(128, dtype=np.float32))
    profile.disable()
    rows = _profile_top(profile)
    assert rows
    assert all(row["cumulative_host_seconds"] >= row["own_host_seconds"] for row in rows)
    assert all("own_host_seconds" in row and "cumulative_host_seconds" in row for row in rows)


def test_compiled_runner_and_callable_are_reused_with_cpu_validation():
    case = Case("compiled_energy", 16, np.dtype("float32"))
    inputs = _inputs(case)
    adapter = object.__new__(Adapter)
    adapter.name = "numpy-control"
    adapter.engine = None
    adapter.xp = np
    adapter._runner_cache = {}

    first, first_reused = adapter._runner_for(case, inputs)
    second, second_reused = adapter._runner_for(case, inputs)
    assert not first_reused
    assert second_reused
    assert first is second
    assert first.fn is second.fn

    expected, _ = _expected(case, inputs)
    actual, _ = adapter.compute(case, inputs, inputs)
    actual_again, _ = adapter.compute(case, inputs, inputs)
    assert _validate(case, actual, expected, inputs)["valid"]
    np.testing.assert_allclose(actual_again, expected, rtol=8e-4, atol=8e-4)


def test_cpu_control_uses_shared_native_solver_and_reports_tolerance():
    case = Case("cg", 12, np.dtype("float32"))
    inputs = _inputs(case)
    adapter = object.__new__(Adapter)
    adapter.name = "numpy-control"
    adapter.engine = None
    adapter.xp = np
    adapter._runner_cache = {}
    actual, metadata = adapter.compute(case, inputs, inputs)
    expected, _ = _expected(case, inputs)
    validation = _validate(case, actual, expected, inputs)
    assert metadata["solver_implementation"] == "benchmarks.suite.native_solver"
    assert metadata["rtol"] == 1e-4
    assert metadata["maxiter"] == 400
    assert metadata["converged"]
    assert validation["valid"]
