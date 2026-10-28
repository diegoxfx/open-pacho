"""Blocked vs interleaved backend timing: a scheduling control, not an attribution.

The process warms every backend before either mode. Differences can reflect
frequency, driver/context state and measurement order, not only kernel changes.
"""

import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.execution import make_inputs
from benchmarks.gaps import Adapter, Case


def run(output):
    adapters = [Adapter(name) for name in ("opencl", "opencl-clblast", "vulkan", "cupy")]
    rows = []
    with threadpool_limits(limits=1):
        for size in (2048, 32768):
            inputs = make_inputs("energy", size, "float32")
            y = 2 * (inputs[0] * inputs[1] + inputs[2])
            expected = np.sum(y * y, axis=-1)
            case = Case("compiled_energy", size, np.dtype("float32"))
            residents = [tuple(a.upload(x) for x in inputs) for a in adapters]

            def measure(index, case=case, residents=residents, inputs=inputs, expected=expected):
                adapter = adapters[index]
                adapter.sync()
                start = time.perf_counter()
                value, _ = adapter.compute(case, residents[index], inputs)
                adapter.sync()
                elapsed = time.perf_counter() - start
                np.testing.assert_allclose(adapter.host(value), expected, rtol=8e-4, atol=8e-4)
                return elapsed

            for i in range(len(adapters)):
                measure(i)
            # Return to blocked timing after interleaving to expose ordering effects.
            for mode in ("blocked_before", "interleaved", "blocked_after"):
                samples = [[] for _ in adapters]
                if mode.startswith("blocked"):
                    for i in range(len(adapters)):
                        measure(i)
                        for _ in range(21):
                            samples[i].append(measure(i))
                else:
                    order, indices = random.Random(47), list(range(len(adapters)))
                    for _ in range(21):
                        order.shuffle(indices)
                        for i in indices:
                            samples[i].append(measure(i))
                for adapter, times in zip(adapters, samples):
                    rows.append(
                        {
                            "size": size,
                            "implementation": adapter.name,
                            "mode": mode,
                            "seconds": times,
                            "median_seconds": statistics.median(times),
                        }
                    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows, "repeats": 21}, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            row["size"],
            row["implementation"],
            row["mode"],
            round(row["median_seconds"] * 1e6, 1),
            "us",
            flush=True,
        )


if __name__ == "__main__":
    run(Path(".artifacts/benchmarks/gaps-final/scheduling.json"))
