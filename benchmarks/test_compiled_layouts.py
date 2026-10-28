"""CPU checks of independent controls, balanced orders, and failure reporting."""

import itertools
import json
from collections import Counter

import numpy as np
import pytest

from benchmarks import compiled_layouts as bench
from openpacho._opencl import OpenCLBackend
from openpacho.backends import NumpyBackend


class Backend(NumpyBackend):
    queue = None
    prepare_layout_matmul = OpenCLBackend.prepare_layout_matmul

    def __init__(self, offset=0):
        self._blas = self
        self.uploads = 0
        self.closed = False
        self.offset = offset

    def upload(self, array):
        self.uploads += 1
        return super().upload(array)

    def matmul(self, *args, **kwargs):
        if not kwargs:
            return super().matmul(*args) + self.offset
        _, a, b, out, _, _ = args
        left, right = a._data.reshape(a.shape), b._data.reshape(b.shape)
        out[:] = (
            (left.T if kwargs["transpose_a"] else left)
            @ (right.T if kwargs["transpose_b"] else right)
        ).ravel() + self.offset

    def close(self):
        self.closed = True


def setup(monkeypatch, *, offset=0):
    backend = Backend(offset)
    monkeypatch.setattr(
        bench, "CASES", ({"name": "small", "dimensions": (5, 7, 9), "side": "both"},)
    )
    monkeypatch.setattr(bench, "engine", lambda _: (bench.p, backend))
    monkeypatch.setattr(bench, "warmup", lambda *args: {"calls": 1, "seconds": 0})
    monkeypatch.setattr(bench, "nvidia_snapshot", lambda: {"available": False})
    events, names = [], {}
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
        events.append(names[function])
        return 1e-6, 1e-6, function()

    monkeypatch.setattr(bench, "prepare_calls", prepare)
    monkeypatch.setattr(bench, "measure", measure)
    return backend, events


def test_real_controls_and_six_balanced_permutations(monkeypatch, tmp_path):
    backend, events = setup(monkeypatch)
    output = tmp_path / "result.json"
    report = bench.run(output, ["opencl-clblast"], repeats=6)
    assert report["complete"] and report["all_reference_close"]
    assert backend.closed
    assert "prepare_layout_matmul" not in vars(backend)
    assert backend.uploads == 2 + 2 * 3 * (1 + 6)
    assert len(report["rows"]) == 3
    for row in report["rows"]:
        assert [step["kind"] for step in row["explain"]["steps"]] == (
            ["layout_matmul"] if row["mode"] == "on" else ["gather", "gather", "matmul"]
        )
        for scope in bench.SCOPES:
            assert len(row[scope]["samples_seconds"]) == 6
            assert row[scope + "_replay_bitwise_equal"]
    for scope in bench.SCOPES:
        modes = [mode for mode, current in events if current == scope]
        assert modes[:3] == list(bench.MODES)
        samples = modes[3:]
        assert Counter(tuple(samples[i : i + 3]) for i in range(0, len(samples), 3)) == Counter(
            itertools.permutations(bench.MODES)
        )
    assert json.loads(output.read_text()) == json.loads(json.dumps(report))


@pytest.mark.parametrize("override", [False, True])
def test_control_restores_hook_on_preparation_error(monkeypatch, override):
    backend = Backend()
    original = lambda *args: None
    if override:
        backend.prepare_layout_matmul = original
    with pytest.raises(RuntimeError, match="prepare failed"), bench.layout_enabled(backend, False):
        assert backend.prepare_layout_matmul is None
        raise RuntimeError("prepare failed")
    if override:
        assert backend.prepare_layout_matmul is original
    else:
        assert "prepare_layout_matmul" not in vars(backend)
        assert callable(backend.prepare_layout_matmul)


def test_covariance_keeps_shared_centered_input_and_folds_private_transpose(monkeypatch, tmp_path):
    backend, _ = setup(monkeypatch)
    monkeypatch.setattr(bench, "CASES", ({"name": "covariance", "covariance": (11, 5)},))
    report = bench.run(tmp_path / "covariance.json", ["opencl-clblast"], repeats=6)
    assert report["complete"] and report["all_reference_close"]
    optimized = report["rows"][1]["explain"]["steps"]
    assert "layout_matmul" in [step["kind"] for step in optimized]
    assert "gather" not in [step["kind"] for step in optimized]
    assert backend.closed


def test_finite_failures_remain_visible_and_are_timed(monkeypatch, tmp_path, capsys):
    backend, _ = setup(monkeypatch, offset=1)
    report = bench.run(tmp_path / "finite.json", ["opencl-clblast"], repeats=6)
    assert report["complete"] and not report["all_reference_close"]
    assert all(row["validation"]["violations"] == 45 for row in report["rows"])
    assert all("complete" in row for row in report["rows"])
    assert "Numerical validation failed" in capsys.readouterr().out
    assert backend.closed


def test_nonfinite_failure_saves_partial_and_closes(monkeypatch, tmp_path):
    backend, _ = setup(monkeypatch, offset=np.nan)
    output = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="Nonfinite"):
        bench.run(output, ["opencl-clblast"], repeats=6)
    report = json.loads(output.read_text())
    assert not report["complete"] and not report["all_reference_close"]
    assert report["execution_error"]["type"] == "ValueError"
    assert not report["rows"][0]["validation"]["finite"]
    assert backend.closed


@pytest.mark.parametrize("repeats", [0, 5, 7, True, 6.0])
def test_bad_repeat_count_rejected(tmp_path, repeats):
    with pytest.raises(ValueError, match="multiple of six"):
        bench.run(tmp_path / "bad.json", ["opencl-clblast"], repeats=repeats)


@pytest.mark.parametrize("backends", [[], ["opencl"], ["opencl-clblast", "opencl-clblast"]])
def test_requires_exactly_one_clblast_backend(tmp_path, backends):
    with pytest.raises(ValueError, match="exactly opencl-clblast"):
        bench.run(tmp_path / "bad.json", backends)
