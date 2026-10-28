"""CPU-only adapter preparation control with dummy allocation and launch."""

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.gaps import _sample_stats
from benchmarks.routing import historical_rng
from openpacho import _random_cuda, _random_opencl, _random_vulkan


def run(output, control_wheel, repeats=11, calls=2000, seed=941):
    if output.exists() or repeats < 3 or calls < 1:
        raise ValueError("New output, >=3 samples and positive call count required")
    legacy, digest = historical_rng(control_wheel)
    rows = []
    order = ["baseline", "adapter"]
    random.Random(seed).shuffle(order)
    for index, (name, module) in enumerate(
        (
            ("opencl", _random_opencl),
            ("vulkan", _random_vulkan),
            ("cuda", _random_cuda),
        )
    ):
        backend = SimpleNamespace(name=name, empty=lambda *args: None, _run=lambda *args: None)
        for distribution in ("random", "uniform", "normal", "exponential"):
            arguments = (
                backend,
                (33,),
                np.dtype("float32"),
                distribution,
                47,
                53,
                np.float32(0),
                np.float32(1),
            )
            for mode in order if index % 2 == 0 else list(reversed(order)):
                function = legacy.random_data if mode == "baseline" else module.random_data
                for _ in range(100):
                    function(*arguments)
                samples = []
                for _ in range(repeats):
                    start = time.perf_counter_ns()
                    for _ in range(calls):
                        function(*arguments)
                    samples.append((time.perf_counter_ns() - start) * 1e-9 / calls)
                rows.append(
                    {
                        "backend": name,
                        "distribution": distribution,
                        "mode": mode,
                        **_sample_stats(samples),
                    }
                )
    report = {
        "rows": rows,
        "control_source_sha256": digest,
        "calls_per_sample": calls,
        "scope": "adapter preparation only, dummy empty and _run; not GPU dispatch or public API latency",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-wheel", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.control_wheel)
