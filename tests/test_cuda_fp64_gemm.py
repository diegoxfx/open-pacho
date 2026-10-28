"""CPU protocol checks; these tests never initialize CUDA or launch a GPU."""

import hashlib
import itertools
import json
from collections import Counter
from dataclasses import asdict

import numpy as np
import pytest

import openpacho as p
from benchmarks import cuda_fp64_gemm as benchmark
from openpacho._cuda import CUDABackend


class CaptureBackend:
    """Use the authoritative CUDA source generator with host-only allocations."""

    matmul = CUDABackend.matmul
    _element_kernel = CUDABackend._element_kernel
    name = "cuda"
    blas = "custom"
    _register_gemm = False

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def _run(self, *args, **kwargs):
        raise AssertionError("Capture must not execute a launch")


def test_capture_uses_runtime_generator_preserves_inherited_run_and_pointer_abi(numpy_backend):
    backend = CaptureBackend()
    a = p.asarray(np.ones((2, 1, 3, 7)), backend=numpy_backend)
    b = p.asarray(np.ones((1, 4, 7, 5)), backend=numpy_backend)
    before = vars(backend).copy()
    kernel = benchmark.capture_current(backend, backend.matmul, a, b, (2, 4, 3, 5), a.dtype)
    assert vars(backend) == before
    assert kernel.blocks == 1 and kernel.threads == 128 and kernel.shared_bytes == 0
    assert kernel.shape == (2, 4, 3, 5)
    assert "long long batch=i/15LL" in kernel.source
    assert "j<7LL" in kernel.source and "__shared__" not in kernel.source
    with pytest.raises(AssertionError, match="Capture must not execute"):
        backend._run(kernel.source, kernel.blocks)


def test_capture_rejects_multiple_launches_and_restores_existing_override(numpy_backend):
    backend = CaptureBackend()
    previous = object()
    backend._run = previous
    a = p.asarray(np.ones((3, 7)), backend=numpy_backend)
    b = p.asarray(np.ones((7, 5)), backend=numpy_backend)

    def two_launches(a, b, shape, dtype):
        out = backend.empty(shape, dtype)
        backend._run("source", 1, a._data, b._data, out)
        backend._run("source", 1, a._data, b._data, out)
        return out

    with pytest.raises(AssertionError, match="got 2"):
        benchmark.capture_current(backend, two_launches, a, b, (3, 5), a.dtype)
    assert backend._run is previous

    def broken(*args):
        raise RuntimeError("deliberate failure")

    with pytest.raises(RuntimeError, match="deliberate failure"):
        benchmark.capture_current(backend, broken, a, b, (3, 5), a.dtype)
    assert backend._run is previous


def test_frozen_sources_can_replay_after_current_route_changes(tmp_path, numpy_backend):
    backend = CaptureBackend()
    a = p.asarray(np.ones((3, 7)), backend=numpy_backend)
    b = p.asarray(np.ones((7, 5)), backend=numpy_backend)
    kernel = benchmark.capture_current(backend, backend.matmul, a, b, (3, 5), a.dtype)
    key = benchmark.route_key(a.shape, b.shape)
    report = {
        "schema": "openpacho.cuda-fp64-gemm.v1",
        "captured_controls": {
            key: dict(asdict(kernel), sha256=hashlib.sha256(kernel.source.encode()).hexdigest())
        },
    }
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    controls = benchmark.load_controls(path)

    def promoted(*args):
        raise AssertionError("Historical controls must bypass current matmul generation")

    backend.matmul = promoted
    routes = benchmark.Routes(
        backend, {"grid_x": 2**31 - 1, "threads": 1024, "shared_bytes": 49152}, controls
    )
    control, candidate = routes.kernels(a, b, (3, 5), a.dtype)
    assert control == kernel and candidate.shared_bytes == 4096
    routes.frozen = True
    changed = p.asarray(np.ones((4, 7)), backend=numpy_backend)
    with pytest.raises(AssertionError, match="Uncaptured control"):
        routes.kernels(changed, b, (4, 5), a.dtype)
    report["captured_controls"][key]["source"] += "corruption"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        benchmark.load_controls(path)


def test_strict_validation_detects_signed_zero_drift_and_special_classes():
    reference = np.array([0.0, -0.0, 1.0, np.inf, -np.inf, np.nan])
    assert benchmark.compare(reference.copy(), reference)["accepted"]
    changed = reference.copy()
    changed[0] = -0.0
    result = benchmark.compare(changed, reference)
    assert not result["accepted"] and not result["finite_bitwise_equal"]
    changed = reference.copy()
    changed[3] = -np.inf
    assert not benchmark.compare(changed, reference)["classifications_equal"]
    changed = reference.copy()
    changed[2] = np.nextafter(changed[2], np.inf)
    assert not benchmark.compare(changed, reference)["accepted"]
    assert not benchmark.compare(reference.astype("float32"), reference)["accepted"]


def test_balanced_orders_and_capability_rejection():
    orders = benchmark.mode_orders(24, 17)
    expected = set(itertools.permutations(benchmark.MODES))
    assert set(orders) == expected
    assert set(Counter(orders).values()) == {4}
    assert orders == benchmark.mode_orders(24, 17)
    for invalid in (True, 0, 5, 7, 12.0):
        with pytest.raises(ValueError):
            benchmark.mode_orders(invalid, 17)
    kernel = benchmark.source_for((17, 19), (19, 33))
    for limits in (
        {"grid_x": 5, "threads": 1024, "shared_bytes": 49152},
        {"grid_x": 100, "threads": 128, "shared_bytes": 49152},
        {"grid_x": 100, "threads": 1024, "shared_bytes": 2048},
    ):
        with pytest.raises(ValueError):
            benchmark.check_limits(kernel, limits)


def test_application_protocol_uses_independent_plans_original_inputs_and_complete_scopes(
    monkeypatch,
    numpy_backend,
):
    # Exercise the real compiler, dense network, transfers, and sample loop on
    # CPU. Only route selection is held at the same real CPU backend method;
    # GPU numerical acceptance remains exclusively a hardware-run obligation.
    original = numpy_backend.matmul

    class CPURoutes:
        frozen = False

        def matmul(self, mode):
            return original

    compile_function = p.compile
    plans = []

    def record_compile(function):
        compiled = compile_function(function)

        class Compiler:
            def prepare(self, *args):
                plan = compiled.prepare(*args)
                plans.append(plan)
                return plan

        return Compiler()

    monkeypatch.setattr(benchmark.p, "compile", record_compile)
    row = {"execution_complete": False, "numerically_accepted": False}
    case = benchmark.dense.Case("sgd", 7, 3, 5, 2, "float64")
    saves = []
    benchmark.application_row(
        case, numpy_backend, CPURoutes(), 6, 17, lambda: saves.append(True), row
    )
    assert len(plans) == 3 and len({id(plan._plan) for plan in plans}) == 3
    assert row["execution_complete"] and row["numerically_accepted"]
    assert row["original_parameters_every_call"]
    assert set(row["scopes"]) == {"resident_prepared_application", "complete"}
    assert len(saves) >= 12
    for scope in row["scopes"].values():
        assert len(scope["orders"]) == 6
        for state in scope["modes"].values():
            assert len(state["samples_seconds"]) == 6
            assert len(state["validation"]) == 8
            assert all(v["accepted"] for v in state["validation"])
            assert state["warmup"]["calls"] >= 3
    # Inherited method remains inherited following all scoped sample calls.
    assert "matmul" not in vars(numpy_backend)


def test_unavailable_backend_retains_honest_report_and_exclusive_output(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("CUDA intentionally unavailable")

    monkeypatch.setattr(benchmark.p, "get_backend", unavailable)
    path = tmp_path / "report.json"
    report = benchmark.run(path, profile="quick", repeats=6)
    assert not report["attempts_complete"]
    assert not report["execution_complete"]
    assert not report["all_numerically_accepted"]
    assert "CUDA intentionally unavailable" in report["error"]["message"]
    assert json.loads(path.read_text(encoding="utf-8")) == report
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.run(path, profile="quick", repeats=6)
    assert path.read_bytes() == before
    assert (
        benchmark.main(
            ["--output", str(tmp_path / "cli.json"), "--profile", "quick", "--repeats", "6"]
        )
        == 1
    )
