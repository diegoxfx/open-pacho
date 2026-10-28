"""Readiness is evidence, not a timer-based claim that the GPU finished."""

import pytest

from benchmarks.stability import Policy, wait_ready


class Clock:
    value = 0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def snapshot(*, temperature=40, utilization=0, memory=0, uuid="gpu"):
    return {
        "available": True,
        "uuid": uuid,
        "temperature.gpu": temperature,
        "utilization.gpu": utilization,
        "utilization.memory": memory,
    }


def gate(samples, **kwargs):
    samples = iter(samples)
    last = None

    def probe():
        nonlocal last
        last = next(samples, last)
        return last

    clock = Clock()
    return wait_ready(
        probe,
        Policy(min_idle_seconds=4, max_wait_seconds=10),
        clock=clock.now,
        sleep=clock.sleep,
        **kwargs,
    )


def test_minimum_pause_and_consecutive_readings():
    result = gate([snapshot()])
    assert result["ready"] and result["elapsed_seconds"] == 4
    assert len(result["readings"]) == 3


@pytest.mark.parametrize(
    "busy", [snapshot(utilization=20), snapshot(memory=20), snapshot(temperature=50)]
)
def test_hot_or_busy_never_passes_on_timeout(busy):
    result = gate([busy], baseline_temperature=40, allow_unverified=True)
    assert not result["ready"] and result["status"] == "timeout"


def test_interrupted_quiet_window_starts_again():
    result = gate([snapshot(), snapshot(utilization=90), snapshot(), snapshot(), snapshot()])
    assert result["ready"] and result["elapsed_seconds"] == 8


def test_temperature_must_return_to_initial_baseline():
    result = gate([snapshot(temperature=t) for t in (49, 46, 42, 41, 41)], baseline_temperature=40)
    assert result["ready"] and result["elapsed_seconds"] == 8


def test_missing_telemetry_is_not_reported_as_stable():
    result = gate([{"available": False}])
    assert not result["ready"] and result["status"] == "telemetry-unavailable"
    result = gate([{"available": False}], allow_unverified=True)
    assert result["ready"] and result["status"] == "unverified-pause"
    assert result["elapsed_seconds"] == 4


def test_probe_loss_does_not_hide_previous_busy_reading():
    result = gate([snapshot(utilization=90), {"available": False}], allow_unverified=True)
    assert not result["ready"] and result["status"] == "timeout"


def test_identity_switch_is_rejected():
    result = gate([snapshot(uuid="one"), snapshot(uuid="two")])
    assert not result["ready"] and result["status"] == "device-identity-changed"


def test_thermal_slowdown_prevents_readiness_even_with_idle_utilization():
    result = gate([{**snapshot(), "thermal_slowdown": True}])
    assert not result["ready"] and result["status"] == "timeout"


def test_worker_telemetry_respects_parent_gpu_selection(monkeypatch):
    from types import SimpleNamespace

    from benchmarks import stability

    commands = []

    def command(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(
            stdout="gpu,name,0000:01:00.0,40,0,0,100,100,10,P8,0,Not Active,Not Active"
        )

    monkeypatch.setenv("PACHO_BENCHMARK_GPU_ID", "GPU-selected")
    monkeypatch.setattr(stability.subprocess, "run", command)
    result = stability.nvidia_snapshot()
    assert result["available"] and result["thermal_slowdown"] is False
    assert "--id=GPU-selected" in commands[-1]
    stability.nvidia_snapshot("explicit")
    assert "--id=explicit" in commands[-1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_idle_seconds": 121},
        {"interval_seconds": 0},
        {"consecutive": 1},
        {"max_wait_seconds": float("nan")},
    ],
)
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        Policy(**kwargs)
