"""Worker completion precedes the next gate; failures never launch a new worker."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import isolated


def test_sequential_gates_and_reversed_second_round(tmp_path, monkeypatch):
    events = []

    def gate(*args, **kwargs):
        events.append("gate")
        return {"ready": True, "baseline_temperature": 40, "status": "observed-idle"}

    def worker(command, **kwargs):
        backend = command[command.index("--backends") + 1]
        events.append(backend)
        Path(command[command.index("--output") + 1]).write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(isolated, "wait_ready", gate)
    monkeypatch.setattr(isolated, "nvidia_snapshot", lambda *a: {"available": True})
    monkeypatch.setattr(isolated.subprocess, "run", worker)
    report = isolated.run(tmp_path / "new", ["opencl", "vulkan"], rounds=2)
    assert report["complete"]
    assert events[::2] == ["gate"] * 4
    assert events[1::2][:2] == events[1::2][2:][::-1]


def test_failed_readiness_preserves_report_and_never_starts_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        isolated, "wait_ready", lambda *a, **kw: {"ready": False, "status": "timeout"}
    )
    monkeypatch.setattr(
        isolated.subprocess,
        "run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("worker started")),
    )
    report = isolated.run(tmp_path / "new", ["opencl"])
    assert not report["complete"]
    assert (
        json.loads((tmp_path / "new/session.json").read_text())["runs"][0]["status"]
        == "not-run-readiness"
    )


def test_failed_worker_prevents_next_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(
        isolated, "wait_ready", lambda *a, **kw: {"ready": True, "baseline_temperature": 40}
    )
    monkeypatch.setattr(isolated, "nvidia_snapshot", lambda *a: {"available": False})
    monkeypatch.setattr(isolated.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    report = isolated.run(tmp_path / "new", ["opencl", "vulkan"])
    assert not report["complete"] and len(report["runs"]) == 1
    assert report["runs"][0]["status"] == "worker-failed"


@pytest.mark.parametrize("backends,repeats", [(["opencl"], 21), (["cuda-cublas"], 30)])
def test_padding_decision_rejects_invalid_experiment_before_creating_output(
    tmp_path, backends, repeats
):
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="Paired decision"):
        isolated.run(output, backends, repeats=repeats, worker="padding_decision")
    assert not output.exists()


def test_padding_decision_uses_isolated_worker_with_balanced_repeat_count(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(
        isolated, "wait_ready", lambda *a, **kw: {"ready": True, "baseline_temperature": 40}
    )
    monkeypatch.setattr(isolated, "nvidia_snapshot", lambda *a: {"available": True})

    def worker(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--output") + 1]).write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(isolated.subprocess, "run", worker)
    report = isolated.run(
        tmp_path / "new", ["opencl"], rounds=1, repeats=30, worker="padding_decision"
    )
    assert report["complete"]
    assert commands[0][2] == "benchmarks.padding_decision"
    assert commands[0][commands[0].index("--repeats") + 1] == "30"
