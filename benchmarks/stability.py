"""Benchmark-only readiness checks; never a replacement for GPU synchronization."""

import csv
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Policy:
    min_idle_seconds: float = 10
    max_wait_seconds: float = 120
    interval_seconds: float = 2
    consecutive: int = 3
    max_utilization: float = 5
    temperature_tolerance: float = 2
    max_temperature: float | None = None

    def __post_init__(self):
        values = (
            self.min_idle_seconds,
            self.max_wait_seconds,
            self.interval_seconds,
            self.max_utilization,
            self.temperature_tolerance,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Readiness policy must be finite")
        if not 0 <= self.min_idle_seconds <= self.max_wait_seconds:
            raise ValueError("Require 0 <= minimum idle <= maximum wait")
        if not 0 < self.interval_seconds <= 60 or self.consecutive < 2:
            raise ValueError("Require 0 < interval <= 60 and at least two readings")
        if not 0 <= self.max_utilization <= 100 or self.temperature_tolerance < 0:
            raise ValueError("Invalid utilization/temperature tolerance")
        if self.max_temperature is not None and not math.isfinite(self.max_temperature):
            raise ValueError("Maximum temperature must be finite")


FIELDS = (
    "uuid",
    "name",
    "pci.bus_id",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "pstate",
    "clocks_event_reasons.active",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
)


def nvidia_snapshot(device=None):
    """One explicitly selected GPU. Unsupported counters remain unavailable."""
    device = os.environ.get("PACHO_BENCHMARK_GPU_ID", "0") if device is None else device
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                f"--query-gpu={','.join(FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        rows = list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != len(FIELDS):
            raise ValueError("Telemetry must identify exactly one GPU")
        data = dict(zip(FIELDS, (v.strip() for v in rows[0])))
        for key in FIELDS[3:9]:
            try:
                data[key] = float(data[key])
                if not math.isfinite(data[key]):
                    data[key] = None
            except ValueError:
                data[key] = None
        thermal = [data[k].lower() for k in FIELDS[-2:]]
        data["thermal_slowdown"] = (
            True
            if "active" in thermal
            else False
            if all(v == "not active" for v in thermal)
            else None
        )
        return {"available": True, **data}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"available": False, "error": str(exc)}


def wait_ready(
    probe,
    policy,
    *,
    baseline_temperature=None,
    allow_unverified=False,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Wait after worker exit; sustained measurement warmup belongs to the worker.

    Missing telemetry can authorize only an explicitly requested unverified pause.
    Busy/hot telemetry never turns into success just because the timeout elapsed.
    """
    start = clock()
    readings, window = [], []
    identity = None
    report = {
        "policy": asdict(policy),
        "baseline_temperature": baseline_temperature,
        "readings": readings,
        "ready": False,
    }
    while True:
        reading = dict(probe())
        elapsed = clock() - start
        reading["elapsed_seconds"] = elapsed
        readings.append(reading)
        required = ("temperature.gpu", "utilization.gpu", "utilization.memory")
        available = reading.get("available") and all(
            isinstance(reading.get(key), (float, int)) and math.isfinite(reading[key])
            for key in required
        )
        if available:
            uuid = reading.get("uuid")
            if not uuid or identity is not None and uuid != identity:
                report.update(status="device-identity-changed", elapsed_seconds=elapsed)
                return report
            identity = uuid
            ceiling = policy.max_temperature
            if baseline_temperature is not None:
                relative = baseline_temperature + policy.temperature_tolerance
                ceiling = relative if ceiling is None else min(ceiling, relative)
            good = (
                max(reading[k] for k in required[1:]) <= policy.max_utilization
                and (ceiling is None or reading["temperature.gpu"] <= ceiling)
                and reading.get("thermal_slowdown") is not True
            )
            window = (window + [reading])[-policy.consecutive :] if good else []
            temps = [r["temperature.gpu"] for r in window]
            if (
                elapsed >= policy.min_idle_seconds
                and len(window) == policy.consecutive
                and max(temps) - min(temps) <= policy.temperature_tolerance
            ):
                report.update(
                    ready=True,
                    status="observed-idle",
                    uuid=identity,
                    baseline_temperature=sum(temps) / len(temps),
                    elapsed_seconds=elapsed,
                )
                return report
        else:
            window = []
            # Do not conceal a hot/busy reading earlier in this gate.
            if (
                allow_unverified
                and elapsed >= policy.min_idle_seconds
                and all(not r.get("available") for r in readings)
            ):
                report.update(ready=True, status="unverified-pause", elapsed_seconds=elapsed)
                return report
            if not allow_unverified:
                report.update(status="telemetry-unavailable", elapsed_seconds=elapsed)
                return report
        if elapsed >= policy.max_wait_seconds:
            report.update(status="timeout", elapsed_seconds=elapsed)
            return report
        sleep(min(policy.interval_seconds, policy.max_wait_seconds - elapsed))
