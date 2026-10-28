"""Warmup and temporary control overrides cannot leak into subsequent runs."""

import pytest

from benchmarks import backend_sharing as runner


def test_control_override_restored_even_if_experiment_raises(monkeypatch, tmp_path):
    import openpacho._gemm as gemm

    original = gemm.register_gemm_data

    def fail(*args):
        restore = args[-1]
        restore.append((gemm, "register_gemm_data", original))
        gemm.register_gemm_data = None
        raise RuntimeError("experiment failed")

    monkeypatch.setattr(runner, "_run", fail)
    with pytest.raises(RuntimeError, match="experiment failed"):
        runner.run(tmp_path / "result.json", ["opencl"])
    assert gemm.register_gemm_data is original


def test_warmup_has_minimum_count_and_duration(monkeypatch):
    state = {"elapsed": 0.0, "calls": 0}
    monkeypatch.setattr(runner.time, "monotonic", lambda: state["elapsed"])

    def measure(*args):
        state["elapsed"] += 0.1
        state["calls"] += 1
        return 0.1, 0.1, None

    monkeypatch.setattr(runner, "measure", measure)
    result = runner.warmup(None, None, None, seconds=0.45)
    assert result["calls"] == 5 and result["seconds"] == pytest.approx(0.5)
