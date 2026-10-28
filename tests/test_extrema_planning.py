"""CPU-only checks for the private benchmark experiment and its evidence."""

import gc
import itertools
import json
import weakref
from collections import Counter

import numpy as np
import pytest

import openpacho as p
from benchmarks import extrema_planning as benchmark


class ArrayLowerer:
    """Exercise real plan rebinding while keeping all storage on NumPy."""

    def __init__(self, backend):
        self.backend = backend
        self.requests = []

    def prepare_map_reduce(self, node, group, leaves):
        self.requests.append((node, group, leaves))
        return tuple(leaves), ()

    def run_map_reduce(self, node, payload, arguments, operands):
        assert payload == ()
        assert len(arguments) == len(operands) == 1
        return self.backend.reduce(
            operands[0],
            node.attrs["axes"],
            node.attrs["keepdims"],
            node.dtype,
            operation=node.attrs["operation"],
        )


@pytest.fixture
def cpu_plans(monkeypatch, numpy_backend):
    """Capture ordinary NumPy plans, then attach a fake native lowerer."""
    compile_function = p.compile
    prepared_calls = []

    def compile_with_lowerer(function):
        compiled = compile_function(function)

        class Compiler:
            def prepare(self, *args):
                prepared = compiled.prepare(*args)
                prepared._plan.lowering = ArrayLowerer(numpy_backend)
                prepared_calls.append(prepared)
                return prepared

        return Compiler()

    monkeypatch.setattr(benchmark.p, "compile", compile_with_lowerer)
    return prepared_calls


def test_private_rewrite_preserves_graph_other_steps_and_tuple_rebinding(numpy_backend):
    def calculate(a):
        return (
            p.max(a, axis=(0, 2), keepdims=True),
            p.min(a, axis=(0, 2)),
            p.sum(a, axis=1),
            a + 1,
        )

    a = p.asarray(np.arange(30, dtype="float32").reshape(2, 3, 5), backend=numpy_backend)
    control = p.compile(calculate).prepare(a)
    candidate = p.compile(calculate).prepare(a)
    plan = candidate._plan
    plan.lowering = ArrayLowerer(numpy_backend)
    original_steps = list(plan.steps)
    original_nodes, original_outputs = plan.nodes, plan.outputs
    original_uses, original_live_uses = plan.uses.copy(), plan.live_uses.copy()
    original_specs = list(candidate._specs)
    assert benchmark.rewrite_extrema(candidate) == 2
    assert plan.nodes is original_nodes and plan.outputs is original_outputs
    assert plan.uses == original_uses and plan.live_uses == original_live_uses
    assert candidate._specs == original_specs
    for original, changed in zip(original_steps, plan.steps):
        if original[0].attrs.get("operation") in ("min", "max"):
            assert changed[0] is original[0] and changed[1] == "map_reduce"
            assert all(a is b for a, b in zip(changed[2], original[0].inputs))
        else:
            assert changed is original
    assert all(
        group == () and leaves == tuple(node.inputs)
        for node, group, leaves in plan.lowering.requests
    )
    assert all(kind != "map_reduce" for _, kind, _, _ in control._plan.steps)
    for replacement in (a, a * -2):
        expected = control(replacement)
        actual = candidate(a=replacement)
        assert isinstance(actual, tuple) and len(actual) == 4
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(p.asnumpy(got), p.asnumpy(want))
    execution_scope = plan._execution_scope
    assert benchmark.rewrite_extrema(candidate, enabled=False) == 2
    assert plan._execution_scope is execution_scope
    assert plan.uses == original_uses and plan.live_uses == original_live_uses
    assert all(kind != "map_reduce" for _, kind, _, _ in plan.steps)
    assert benchmark.rewrite_extrema(candidate, enabled=False) == 0
    for got, want in zip(candidate(a=a * -3), control(a * -3)):
        np.testing.assert_array_equal(p.asnumpy(got), p.asnumpy(want))
    # Plans/lowering hold only expression metadata, not the example array.
    reference = weakref.ref(a)
    del a, replacement
    gc.collect()
    assert reference() is None


def test_rewrite_failure_is_atomic_and_requires_a_candidate(numpy_backend):
    a = p.ones((2, 3), dtype="float32", backend=numpy_backend)
    prepared = p.compile(lambda x: (p.min(x), p.max(x))).prepare(a)
    original = prepared._plan.steps

    class BadLowerer(ArrayLowerer):
        def prepare_map_reduce(self, node, group, leaves):
            if self.requests:
                return (), ()
            return super().prepare_map_reduce(node, group, leaves)

    prepared._plan.lowering = BadLowerer(numpy_backend)
    with pytest.raises(AssertionError, match="dependencies"):
        benchmark.rewrite_extrema(prepared)
    assert prepared._plan.steps is original
    no_extrema = p.compile(lambda x: p.sum(x)).prepare(a)
    no_extrema._plan.lowering = ArrayLowerer(numpy_backend)
    with pytest.raises(AssertionError, match="did not find"):
        benchmark.rewrite_extrema(no_extrema)
    with pytest.raises(AssertionError, match="did not find"):
        benchmark.rewrite_extrema(no_extrema, enabled=False)
    no_extrema._plan.lowering = None
    with pytest.raises(ValueError, match="native"):
        benchmark.rewrite_extrema(no_extrema)


def test_prepare_calls_isolates_three_plans_and_reuses_inputs(cpu_plans, numpy_backend):
    host = np.arange(30, dtype="float32").reshape(2, 3, 5)
    case = benchmark.Case("tuple", host.shape, "min", (0, 2))
    calls, metadata = benchmark.prepare_calls(numpy_backend, host, benchmark.case_function(case))
    assert len({id(prepared._plan) for prepared in cpu_plans}) == 3
    assert [metadata[mode]["transformed_steps"] for mode in benchmark.MODES] == [0, 1, 0]
    assert [prepared._plan.steps[0][1] for prepared in cpu_plans] == [
        "reduce",
        "map_reduce",
        "reduce",
    ]
    assert [prepared._plan.steps[0][0].attrs["axes"] for prepared in cpu_plans] == [(0, 2)] * 3
    for mode in benchmark.MODES:
        for scope in benchmark.SCOPES:
            actual = calls[mode][scope]()
            if scope == "prepared":
                actual = p.asnumpy(actual)
            np.testing.assert_array_equal(actual, np.min(host, axis=(0, 2)))
    assert metadata["off-a"]["plan_before"] == metadata["off-a"]["plan_after"]
    assert metadata["off-b"]["plan_before"] == metadata["off-b"]["plan_after"]
    assert metadata["on"]["plan_before"]["steps"][0]["kind"] == "reduce"
    assert metadata["on"]["plan_after"]["steps"][0]["kind"] == "map_reduce"


def test_private_routes_preserve_genuinely_fused_producers(monkeypatch, numpy_backend):
    monkeypatch.setattr(
        numpy_backend, "create_lowering", lambda: ArrayLowerer(numpy_backend), raising=False
    )
    a = p.ones((2, 3), dtype="float32", backend=numpy_backend)
    prepared = p.compile(lambda x: (p.max(x, axis=1), p.min(x + 1, axis=1))).prepare(a)
    plan = prepared._plan
    assert [kind for _, kind, _, _ in plan.steps] == ["reduce", "map_reduce"]
    fused = plan.steps[1]
    fused_node, _, fused_arguments, _ = fused
    assert all(argument is not fused_node.inputs[0] for argument in fused_arguments)
    assert plan.lowering.requests[0][1]  # Captured producer has a nonempty map group.
    uses, live_uses, scope = plan.uses.copy(), plan.live_uses.copy(), plan._execution_scope
    assert benchmark.rewrite_extrema(prepared, enabled=True) == 1
    assert plan.steps[1] is fused
    assert benchmark.rewrite_extrema(prepared, enabled=False) == 1
    assert plan.steps[1] is fused
    assert plan.uses == uses and plan.live_uses == live_uses
    assert plan._execution_scope is scope


def test_prepare_calls_forces_controls_after_default_promotion(
    monkeypatch, cpu_plans, numpy_backend
):
    compile_function = p.compile

    def promoted_compile(function):
        compiled = compile_function(function)

        class PromotedCompiler:
            def prepare(self, *args):
                prepared = compiled.prepare(*args)
                benchmark.rewrite_extrema(prepared, enabled=True)
                return prepared

        return PromotedCompiler()

    monkeypatch.setattr(benchmark.p, "compile", promoted_compile)
    host = np.arange(30, dtype="float32").reshape(2, 3, 5)
    case = benchmark.Case("tuple", host.shape, "min", (0, 2))
    calls, metadata = benchmark.prepare_calls(numpy_backend, host, benchmark.case_function(case))
    assert len({id(prepared._plan) for prepared in cpu_plans}) == 3
    assert all(metadata[mode]["transformed_steps"] == 1 for mode in benchmark.MODES)
    assert all(
        metadata[mode]["plan_before"]["steps"][0]["kind"] == "map_reduce"
        for mode in benchmark.MODES
    )
    assert [metadata[mode]["forced_route"] for mode in benchmark.MODES] == [
        "reduce",
        "map_reduce",
        "reduce",
    ]
    for mode in benchmark.MODES:
        assert metadata[mode]["plan_after"]["steps"][0]["kind"] == metadata[mode]["forced_route"]
        for scope in benchmark.SCOPES:
            actual = calls[mode][scope]()
            if scope == "prepared":
                actual = p.asnumpy(actual)
            np.testing.assert_array_equal(actual, np.min(host, axis=(0, 2)))


def test_case_manifest_references_and_fixed_acceptance(numpy_backend):
    assert len(benchmark.CASES) == len({case.name for case in benchmark.CASES}) == 11
    assert [(case.operation, case.shape, case.axis) for case in benchmark.CASES] == [
        ("max", (3, 8193), 1),
        ("max", (1024, 1024), 1),
        ("min", (8193, 33), 1),
        ("max", (8193, 33), 0),
        ("max", (1048577,), None),
        ("min", (17, 129, 33), (0, 2)),
        ("max", (3, 17), 1),
        ("softmax", (32, 8193), -1),
        ("softmax", (1024, 1024), -1),
        ("softmax", (8193, 33), -1),
        ("logsumexp", (32, 8193), -1),
    ]
    host = np.array([[-4, 0, 4], [-1, 1, 3]], dtype="float32")
    for operation in ("min", "max", "softmax", "logsumexp"):
        case = benchmark.Case(operation, host.shape, operation, 1)
        expected = benchmark.reference_result(case, host)
        actual = p.asnumpy(benchmark.case_function(case)(p.asarray(host, backend=numpy_backend)))
        validation = benchmark.validate_result(actual, expected, case)
        assert validation["reference_close"]
        assert (validation["rtol"], validation["atol"]) == (
            (0, 0) if operation in ("min", "max") else (2e-5, 2e-6)
        )
        damaged = actual.copy()
        damaged.flat[0] += 1
        assert not benchmark.validate_result(damaged, expected, case)["reference_close"]


@pytest.mark.parametrize("repeats", (6, 30))
def test_mode_orders_are_balanced_and_reproducible(repeats):
    orders = benchmark.mode_orders(repeats, 41)
    assert orders == benchmark.mode_orders(repeats, 41)
    assert Counter(orders) == Counter(
        {order: repeats // 6 for order in itertools.permutations(benchmark.MODES)}
    )


def _mock_worker(monkeypatch, numpy_backend):
    monkeypatch.setattr(benchmark, "CASES", (benchmark.Case("tiny", (3, 17), "max", 1),))
    monkeypatch.setattr(benchmark, "engine", lambda name: (p, numpy_backend))
    monkeypatch.setattr(benchmark, "nvidia_snapshot", lambda: {"cpu_test": True})

    def warmup(function, xp, backend, *, seconds):
        assert seconds == 0.1
        for _ in range(3):
            function()
        return {"calls": 3, "seconds": seconds}

    monkeypatch.setattr(benchmark, "warmup", warmup)


def test_cpu_worker_records_protocol_and_refuses_overwrite(
    tmp_path, monkeypatch, numpy_backend, cpu_plans
):
    _mock_worker(monkeypatch, numpy_backend)
    output = tmp_path / "report.json"
    report = benchmark.run(output, "opencl", repeats=6, seed=42)
    assert report["complete"] and report["all_reference_close"]
    assert report["all_bitwise_control_equal"] and len(report["rows"]) == 3
    assert json.loads(output.read_text())["complete"]
    assert report["runtime_sources"] and report["runner_sha256"]
    assert len({row["input_sha256"] for row in report["rows"]}) == 1
    for row in report["rows"]:
        for scope in benchmark.SCOPES:
            assert len(row[scope]["samples_seconds"]) == 6
            assert row[scope]["samples_seconds"] == row[scope + "_samples_seconds"]
            assert len(set(map(tuple, row[scope + "_orders"]))) == 6
            assert row["first_call"][scope]["seconds"] >= 0
            for stage in ("before", "after"):
                assert row["validation"][stage][scope]["bitwise_control_equal"]
                assert row["error"][stage][scope]["max_abs_error"] == 0
    original = output.read_bytes()
    with pytest.raises(ValueError, match="new output"):
        benchmark.run(output, "opencl", repeats=6)
    assert output.read_bytes() == original


def test_cpu_worker_persists_original_failure_even_if_cleanup_fails(
    tmp_path, monkeypatch, numpy_backend, cpu_plans
):
    _mock_worker(monkeypatch, numpy_backend)

    def fail(*args):
        raise RuntimeError("prepare failed")

    def fail_cleanup():
        raise ValueError("cleanup failed")

    monkeypatch.setattr(benchmark, "prepare_calls", fail)
    monkeypatch.setattr(numpy_backend, "close", fail_cleanup, raising=False)
    output = tmp_path / "incomplete.json"
    with pytest.raises(RuntimeError, match="prepare failed"):
        benchmark.run(output, "vulkan", repeats=6)
    report = json.loads(output.read_text())
    assert not report["complete"]
    assert report["execution_error"]["type"] == "RuntimeError"
    assert report["cleanup_error"]["message"] == "cleanup failed"


def test_cpu_worker_completion_preserves_finite_reference_failure(
    tmp_path, monkeypatch, numpy_backend, cpu_plans
):
    _mock_worker(monkeypatch, numpy_backend)
    reference = benchmark.reference_result
    monkeypatch.setattr(benchmark, "reference_result", lambda case, host: reference(case, host) + 1)
    report = benchmark.run(tmp_path / "finite-failure.json", "opencl", repeats=6)
    assert report["complete"] and not report["all_reference_close"]
    assert report["all_bitwise_control_equal"]
    assert all(row["validation"]["after"]["complete"]["violations"] == 3 for row in report["rows"])


def test_cpu_worker_detects_post_timing_bit_changes(
    tmp_path, monkeypatch, numpy_backend, cpu_plans
):
    _mock_worker(monkeypatch, numpy_backend)
    prepare = benchmark.prepare_calls

    def prepare_changed(*args):
        calls, metadata = prepare(*args)
        original = calls["on"]["complete"]
        count = 0

        def changed():
            nonlocal count
            count += 1
            actual = original()
            # First call + three warmups + six samples precede the replay.
            if count == 11:
                actual.flat[0] += 1
            return actual

        calls["on"]["complete"] = changed
        return calls, metadata

    monkeypatch.setattr(benchmark, "prepare_calls", prepare_changed)
    output = tmp_path / "changed-bits.json"
    with pytest.raises(AssertionError, match="Replay output"):
        benchmark.run(output, "opencl", repeats=6)
    report = json.loads(output.read_text())
    assert not report["complete"] and not report["all_bitwise_control_equal"]
    candidate = next(row for row in report["rows"] if row["mode"] == "on")
    assert len(candidate["complete"]["samples_seconds"]) == 6
    assert not candidate["validation"]["after"]["complete"]["bitwise_control_equal"]


@pytest.mark.parametrize(
    "backend,repeats,seed",
    [
        ("numpy", 6, 1),
        ("cuda", 6, 1),
        ("opencl", 5, 1),
        ("opencl", 7, 1),
        ("opencl", True, 1),
        ("opencl", 6, -1),
    ],
)
def test_invalid_requests_never_instantiate_backend(tmp_path, monkeypatch, backend, repeats, seed):
    monkeypatch.setattr(benchmark, "engine", lambda name: pytest.fail("Unexpected backend"))
    with pytest.raises(ValueError):
        benchmark.run(tmp_path / "invalid.json", backend, repeats, seed)
    assert not (tmp_path / "invalid.json").exists()
