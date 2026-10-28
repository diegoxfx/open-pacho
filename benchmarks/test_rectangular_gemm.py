"""CPU-only verification of paired runtime selection and numerical reporting."""

import itertools
import json
from collections import Counter

import numpy as np
import pytest

from benchmarks import backend_sharing
from benchmarks import rectangular_gemm as bench
from openpacho.backends import NumpyBackend


class Backend(NumpyBackend):
    _small_row_gemm_config = ((16, 32, 16), 16)

    def __init__(self, *, candidate_offset=0, fail_close=False):
        self.uploads = 0
        self.closed = False
        self.candidate_offset = candidate_offset
        self.fail_close = fail_close
        self.configurations = []

    def upload(self, value):
        self.uploads += 1
        return super().upload(value)

    def matmul(self, *args):
        self.configurations.append(self._small_row_gemm_config)
        out = super().matmul(*args)
        return out + self.candidate_offset if self._small_row_gemm_config else out

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close failed")


def setup(monkeypatch, **kwargs):
    backend = Backend(**kwargs)
    monkeypatch.setattr(bench, "CASES", (((33, 17), (17, 257)),))
    monkeypatch.setattr(bench, "engine", lambda _: (bench.p, backend))
    monkeypatch.setattr(bench, "warmup", lambda *args: {"calls": 1, "seconds": 0})
    monkeypatch.setattr(bench, "nvidia_snapshot", lambda: {"available": False})
    names, events = {}, []
    original = bench.prepare_calls

    def prepare(*args):
        calls, explanations = original(*args)
        names.update(
            (function, (mode, scope))
            for mode, scopes in calls.items()
            for scope, function in scopes.items()
        )
        return calls, explanations

    def measure(function, *args):
        function._activate_variant()
        events.append((names[function], backend._small_row_gemm_config))
        return 1e-6, 1e-6, function()

    monkeypatch.setattr(bench, "prepare_calls", prepare)
    monkeypatch.setattr(bench, "measure", measure)
    return backend, events


@pytest.mark.parametrize("name", ["opencl", "cuda-custom"])
def test_balanced_runtime_selection_and_real_complete_uploads(monkeypatch, tmp_path, name):
    backend, events = setup(monkeypatch)
    output = tmp_path / "paired.json"
    report = bench.run(output, [name], repeats=6)
    assert report["complete"] and report["all_reference_close"] and backend.closed
    assert "_small_row_gemm_config" not in vars(backend)
    assert backend.uploads == 2 + 2 * 3 * (1 + 6)
    assert backend.configurations[:3] == [None, Backend._small_row_gemm_config, None]
    for (mode, _), configuration in events:
        assert configuration == (Backend._small_row_gemm_config if mode == "on" else None)
    for scope in bench.SCOPES:
        modes = [mode for ((mode, event_scope), _) in events if event_scope == scope]
        assert modes[:3] == list(bench.MODES)
        samples = modes[3:]
        assert Counter(tuple(samples[i : i + 3]) for i in range(0, len(samples), 3)) == Counter(
            itertools.permutations(bench.MODES)
        )
    for row in report["rows"]:
        assert row["explain"]["selected_tile"] == ((16, 32, 16) if row["mode"] == "on" else None)
        assert row["bitwise_control_equal"]
        for scope in bench.SCOPES:
            assert row[scope + "_replay_bitwise_equal"]
            assert len(row[scope]["samples_seconds"]) == 6
    assert json.loads(output.read_text()) == json.loads(json.dumps(report))


def test_fallback_and_batched_cases_are_recorded(monkeypatch, tmp_path):
    backend, _ = setup(monkeypatch)
    monkeypatch.setattr(bench, "CASES", (((32, 16), (16, 256)), ((2, 33, 17), (17, 257))))
    report = bench.run(tmp_path / "fallback.json", ["opencl"], repeats=6)
    assert report["complete"] and report["all_reference_close"]
    assert all(row["explain"]["selected_tile"] is None for row in report["rows"])
    assert all(row["explain"]["selected_route"] == "direct-fallback" for row in report["rows"])
    assert backend.closed


def test_candidate_precision_failure_is_visible_without_relaxation(monkeypatch, tmp_path, capsys):
    backend, _ = setup(monkeypatch, candidate_offset=1)
    original = ((16, 16, 32), 16)
    backend._small_row_gemm_config = original
    report = bench.run(tmp_path / "precision.json", ["cuda-custom"], repeats=6)
    assert report["complete"] and not report["all_reference_close"]
    assert backend._small_row_gemm_config is original
    assert [row["validation"]["reference_close"] for row in report["rows"]] == [True, False, True]
    assert not report["rows"][1]["bitwise_control_equal"]
    assert all(
        row["validation"]["rtol"] == row["validation"]["atol"] == 5e-5 for row in report["rows"]
    )
    assert "Numerical validation failed" in capsys.readouterr().out


@pytest.mark.parametrize("instance_override", [False, True])
def test_nonfinite_failure_restores_exact_override(monkeypatch, tmp_path, instance_override):
    backend, _ = setup(monkeypatch, candidate_offset=np.nan)
    original = ((16, 16, 32), 16)
    if instance_override:
        backend._small_row_gemm_config = original
    output = tmp_path / "failure.json"
    with pytest.raises(ValueError, match="Nonfinite"):
        bench.run(output, ["opencl"], repeats=6)
    report = json.loads(output.read_text())
    assert backend.closed and not report["complete"] and not report["all_reference_close"]
    if instance_override:
        assert backend._small_row_gemm_config is original
    else:
        assert "_small_row_gemm_config" not in vars(backend)
    assert report["execution_error"]["type"] == "ValueError"


def test_cleanup_failure_is_recorded(monkeypatch, tmp_path):
    backend, _ = setup(monkeypatch, fail_close=True)
    output = tmp_path / "cleanup.json"
    with pytest.raises(RuntimeError, match="close failed"):
        bench.run(output, ["opencl"], repeats=6)
    report = json.loads(output.read_text())
    assert not report["complete"] and report["cleanup_error"]["type"] == "RuntimeError"
    assert backend.closed and "_small_row_gemm_config" not in vars(backend)


def test_real_measure_activates_before_clock_and_synchronization(monkeypatch):
    events = []

    def function():
        events.append("execute")
        return 1

    function._activate_variant = lambda: events.append("activate")
    monkeypatch.setattr(backend_sharing, "sync", lambda *args: events.append("sync"))
    monkeypatch.setattr(
        backend_sharing.time, "perf_counter_ns", lambda: events.append("clock") or 0
    )
    backend_sharing.measure(function, None, None)
    assert events == ["activate", "sync", "clock", "execute", "clock", "sync", "clock"]


def test_independent_prepared_plans_share_resident_inputs_and_upload_replacements(monkeypatch):
    backend = Backend()
    host = (np.ones((33, 17), dtype="float32"), np.ones((17, 257), dtype="float32"))
    identities = []
    original = bench.p.compile

    def compile(function):
        result = original(function)
        identities.append(result)
        return result

    monkeypatch.setattr(bench.p, "compile", compile)
    calls, _ = bench.prepare_calls(backend, host, backend._small_row_gemm_config)
    assert len(identities) == 3 and len({id(compiled) for compiled in identities}) == 3
    assert backend.uploads == 2
    for scopes in calls.values():
        scopes["complete"]._activate_variant()
        scopes["complete"]()
    assert backend.uploads == 2 + 3 * 2


@pytest.mark.parametrize("repeats", [0, 5, 7, True, 6.0])
def test_invalid_repeat_counts(tmp_path, repeats):
    with pytest.raises(ValueError, match="multiple of six"):
        bench.run(tmp_path / "bad.json", ["opencl"], repeats=repeats)


@pytest.mark.parametrize("backends", [[], ["opencl-clblast"], ["opencl", "cuda-custom"]])
def test_requires_one_custom_backend(tmp_path, backends):
    with pytest.raises(ValueError, match="Select one"):
        bench.run(tmp_path / "bad.json", backends)
