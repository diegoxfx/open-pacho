"""CPU validation of vector A/B/A controls, timing scopes and failure reports."""

import itertools
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import blas_vectors as runner
from openpacho.backends import NumpyBackend


class RecordingBackend(NumpyBackend):
    def __init__(self, offset=0, fail_close=False):
        self.closed, self.uploads = False, 0
        self.offset, self.fail_close = offset, fail_close

    def upload(self, value):
        self.uploads += 1
        return super().upload(value)

    def matmul(self, a, b, shape, dtype):
        return super().matmul(a, b, shape, dtype) + np.dtype(dtype).type(self.offset)

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close failed")


def setup(monkeypatch, tmp_path, *, offset=0, fail_close=False):
    backend = RecordingBackend(offset, fail_close)
    original = backend.matmul
    wheel = tmp_path / "control.whl"
    wheel.write_bytes(b"fake wheel; extracted methods are patched")
    monkeypatch.setattr(runner, "CASES", (((2, 3), (3,)),))
    monkeypatch.setattr(runner, "engine", lambda name: (runner.p, backend))
    monkeypatch.setattr(runner, "warmup", lambda *args: {"calls": 0, "seconds": 0})
    monkeypatch.setattr(runner, "nvidia_snapshot", lambda: {"available": False})
    monkeypatch.setattr(runner, "old_methods", lambda *args: {"matmul": original.__func__})
    events, call_names = [], {}
    prepare = runner.prepare_calls

    def prepare_calls(*args):
        calls = prepare(*args)
        call_names.update(
            (function, (mode, scope))
            for mode, scopes in calls.items()
            for scope, function in scopes.items()
        )
        return calls

    def measure(function, *args):
        function._activate_variant()
        events.append(call_names[function])
        return 1e-6, 1e-6, function()

    monkeypatch.setattr(runner, "prepare_calls", prepare_calls)
    monkeypatch.setattr(runner, "measure", measure)
    return backend, original, wheel, events


def test_report_balances_six_orders_includes_uploads_and_restores_control(tmp_path, monkeypatch):
    backend, original, wheel, events = setup(monkeypatch, tmp_path)
    output = tmp_path / "report.json"
    report = runner.run(output, ["cuda-cublas"], repeats=6, control_wheel=wheel)
    assert report["complete"] is report["all_reference_close"] is True
    assert report["old_controls_bitwise_equal"] is True
    assert backend.closed and backend.matmul == original
    assert len(report["rows"]) == len(runner.DTYPES) * 3
    per_dtype = len(events) // len(runner.DTYPES)
    for start in range(0, len(events), per_dtype):
        group = events[start : start + per_dtype]
        for scope in runner.SCOPES:
            measured = [mode for mode, event_scope in group if event_scope == scope]
            # Resident also has three initial accuracy calls; both have replay.
            validation_calls = 6 if scope == "resident" else 3
            samples = measured[validation_calls:]
            assert Counter(tuple(samples[i : i + 3]) for i in range(0, len(samples), 3)) == Counter(
                itertools.permutations(runner.MODES)
            )
    assert backend.uploads == len(runner.DTYPES) * (2 + 2 * 3 * 7)
    assert all(
        row[scope + "_replay_bitwise_equal"] and len(row[scope]["samples_seconds"]) == 6
        for row in report["rows"]
        for scope in runner.SCOPES
    )
    assert json.loads(output.read_text()) == json.loads(json.dumps(report))


def test_independent_plans_share_resident_arrays_but_total_reuploads(monkeypatch):
    backend = RecordingBackend()
    modes, plans = [], []
    controls = SimpleNamespace(activate=lambda mode: modes.append(mode))
    hosts = (np.ones((2, 3), np.float32), np.ones(3, np.float32))

    def compile_probe(function):
        assert function is runner.product
        identity = len(plans)
        plans.append(identity)
        return SimpleNamespace(prepare=lambda *args: lambda *values: (identity, values))

    monkeypatch.setattr(runner.p, "compile", compile_probe)
    monkeypatch.setattr(runner.p, "asnumpy", lambda value: value)
    calls = runner.prepare_calls(backend, controls, hosts)
    assert modes == list(runner.MODES) and len(plans) == 3
    first = None
    for index, (mode, scopes) in enumerate(calls.items()):
        for function in scopes.values():
            function._activate_variant()
            assert modes[-1] == mode
        identity, resident = scopes["resident"]()
        other_identity, uploaded = scopes["complete"]()
        assert identity == other_identity == index
        assert all(a is not b for a, b in zip(resident, uploaded))
        if first is None:
            first = resident
        assert all(a is b for a, b in zip(resident, first))


def test_numerical_failure_is_visible_and_does_not_stop_other_modes(tmp_path, monkeypatch, capsys):
    backend, original, wheel, _ = setup(monkeypatch, tmp_path, offset=1)
    report = runner.run(tmp_path / "error.json", ["cuda-cublas"], repeats=6, control_wheel=wheel)
    assert backend.closed and backend.matmul == original
    assert report["complete"] is True and report["all_reference_close"] is False
    assert len(report["rows"]) == 6
    assert all(row["validation"]["violations"] == 2 for row in report["rows"])
    assert "Numerical validation failed" in capsys.readouterr().out
    float32 = next(row for row in report["rows"] if row["dtype"] == "float32")
    float64 = next(row for row in report["rows"] if row["dtype"] == "float64")
    assert (float32["validation"]["rtol"], float32["validation"]["atol"]) == (5e-5, 5e-6)
    assert (float64["validation"]["rtol"], float64["validation"]["atol"]) == (1e-11, 1e-12)


@pytest.mark.parametrize("failure", ["execution", "cleanup"])
def test_failures_restore_runtime_close_and_save_report(tmp_path, monkeypatch, failure):
    backend, original, wheel, _ = setup(monkeypatch, tmp_path, fail_close=failure == "cleanup")
    if failure == "execution":

        def fail(*args):
            raise RuntimeError("execution failed")

        monkeypatch.setattr(runner, "measure", fail)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError):
        runner.run(output, ["cuda-cublas"], repeats=6, control_wheel=wheel)
    saved = json.loads(output.read_text())
    assert saved["complete"] is False
    assert failure + "_error" in saved
    assert backend.closed and backend.matmul == original


def test_controls_reject_unknown_modes_without_mutating_backend(monkeypatch):
    backend = RecordingBackend()
    original = backend.matmul
    monkeypatch.setattr(runner, "old_methods", lambda *args: {"matmul": original.__func__})
    controls = runner.Controls(backend, None)
    with pytest.raises(ValueError, match="Unknown"):
        controls.activate("bad")
    assert backend.matmul == original


@pytest.mark.parametrize(
    "options",
    [
        {"backends": []},
        {"backends": ["cuda-custom"]},
        {"repeats": 3},
        {"repeats": 7},
        {"repeats": True},
        {"seed": -1},
        {"control_component": "gemm"},
        {"control_wheel": None},
    ],
)
def test_invalid_configuration_does_not_initialize_gpu(tmp_path, monkeypatch, options):
    monkeypatch.setattr(runner, "engine", lambda *args: pytest.fail("GPU initialized"))
    args = {
        "backends": ["cuda-cublas"],
        "repeats": 6,
        "control_wheel": tmp_path / "x.whl",
        **options,
    }
    with pytest.raises(ValueError):
        runner.run(tmp_path / "invalid.json", **args)


def test_cli_success_means_completion_not_accuracy(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda *args: observed.append(args) or {"complete": True, "all_reference_close": False},
    )
    output, wheel = tmp_path / "output.json", tmp_path / "before.whl"
    assert (
        runner.main(
            [
                "--output",
                str(output),
                "--control-wheel",
                str(wheel),
                "--repeats",
                "12",
                "--seed",
                "19",
            ]
        )
        == 0
    )
    assert observed == [(output, ["cuda-cublas"], 12, 19, wheel, "methods")]
