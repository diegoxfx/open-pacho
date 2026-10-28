"""Device event diagnostics, separate from ordinary uninstrumented benchmarks.

Records actual OpenCL kernel/copy event durations on an in-order profiling queue.
Host wall minus kernel time is NOT reported as Python time: gaps include driver,
queue starvation and waits. Does not intercept opaque native CLBlast commands.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from pyopencl.tools import ImmediateAllocator, MemoryPool

import openpacho as p
from examples import solvers


def run(output, size=2048):
    backend = p.get_backend("opencl")
    cl = backend.cl
    backend.queue = cl.CommandQueue(
        backend.context, properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    backend._pool = MemoryPool(ImmediateAllocator(backend.queue))
    rng = np.random.default_rng(49)
    a = rng.uniform(-0.1, 0.1, (size, size)).astype(np.float32)
    a = (a + a.T) / 2
    a[np.diag_indices(size)] = np.sum(np.abs(a), axis=1) + 1
    x = rng.uniform(-1, 1, size).astype(np.float32)
    da, dx = (p.asarray(v, backend=backend) for v in (a, x))
    scalar = p.asarray(np.asarray(1, dtype=np.float32), backend=backend)
    energy = p.compile(lambda a, x: np.sum((a * x + 1) * (a * x + 1), axis=-1))
    cases = {
        "scalar_download": (lambda: scalar.item(), np.asarray(1, dtype=np.float32)),
        "gemv": (lambda: da @ dx, a @ x),
        "dot": (lambda: dx @ dx, x @ x),
        "energy": (lambda: energy(da, dx), np.sum((a * x + 1) * (a * x + 1), axis=-1)),
        "cg": (lambda: solvers.conjugate_gradient(da, dx, rtol=1e-4), np.linalg.solve(a, x)),
    }
    rows = []
    for name, (function, expected) in cases.items():
        function()
        backend.synchronize()
        originals = dict(backend._kernels)
        events = []

        class RecordedKernel:
            def __init__(self, kernel, records):
                self.kernel = kernel
                self.records = records

            def __call__(self, *args, **kwargs):
                event = self.kernel(*args, **kwargs)
                self.records.append(("kernel", event))
                return event

        for source, kernel in originals.items():
            backend._kernels[source] = RecordedKernel(kernel, events)
        original_copy = cl.enqueue_copy

        def copy(*args, _original=original_copy, _events=events, **kwargs):
            event = _original(*args, **kwargs)
            _events.append(("copy", event))
            return event

        cl.enqueue_copy = copy
        try:
            times, records = [], []
            for _ in range(7):
                events.clear()
                start = time.perf_counter()
                actual = function()
                backend.synchronize()
                times.append(time.perf_counter() - start)
                records.append(
                    [
                        {
                            "kind": kind,
                            "queued_ns": event.profile.queued,
                            "submit_ns": event.profile.submit,
                            "start_ns": event.profile.start,
                            "end_ns": event.profile.end,
                        }
                        for kind, event in events
                    ]
                )
            iterations = actual.iterations if name == "cg" else None
            if name == "cg":
                assert actual.converged
                actual = actual.x
            host = (
                p.asnumpy(actual)
                if isinstance(actual, p.ndarray)
                else np.asarray(actual, dtype=np.float32)
            )
            np.testing.assert_allclose(host, expected, rtol=8e-4, atol=1e-4)
        finally:
            cl.enqueue_copy = original_copy
            for source, kernel in originals.items():
                if source in backend._kernels:
                    backend._kernels[source] = kernel
        row = {
            "case": name,
            "iterations": iterations,
            "profiled_wall_seconds": times,
            "events": records,
        }
        for kind in ("kernel", "copy"):
            row[f"{kind}_count"] = [sum(e["kind"] == kind for e in sample) for sample in records]
            row[f"{kind}_median_seconds"] = statistics.median(
                sum(e["end_ns"] - e["start_ns"] for e in sample if e["kind"] == kind) * 1e-9
                for sample in records
            )
        rows.append(row)
        print(
            name,
            "wall ms",
            statistics.median(times) * 1000,
            "kernel ms",
            row["kernel_median_seconds"] * 1000,
            flush=True,
        )
    result = {
        "runtime": p.__file__,
        "device": backend.device,
        "size": size,
        "queue_profiling_enabled": True,
        "kernel_coverage": "custom kernels only; excludes native BLAS",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=2048)
    args = parser.parse_args()
    run(args.output, args.size)
