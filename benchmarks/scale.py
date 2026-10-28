"""Cache-exceeding GEMV/DOT and batched matmul controls.

Seven shuffled synchronized rounds; resident and upload/compute/download modes.
NumPy uses existing host inputs, with independent 1/8/16 BLAS thread limits.
Optional backend failures are preserved and produce a nonzero exit status.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import openpacho as p
from benchmarks.gaps import Adapter
from benchmarks.numerical_metrics import precision_metadata, record_error


class CPU:
    def __init__(self, threads):
        self.name = f"numpy-{threads}t"
        self.threads = threads

    def upload(self, value):
        return value

    def host(self, value):
        return value

    def sync(self):
        pass


def run(output, repeats=7):
    adapters = [CPU(t) for t in (1, 8, 16)]
    errors, rows = {}, []
    for name in ("opencl", "opencl-clblast", "vulkan", "cupy"):
        try:
            adapter = Adapter(name)
            adapter.threads = 1
            adapters.append(adapter)
        except Exception as exc:  # noqa: BLE001 - report unavailable requested devices
            errors[name] = str(exc)
    cases = [("gemv", (n, n), (n,), dt) for n in (512, 2048, 8192) for dt in ("float32", "float64")]
    cases += [("dot", (n,), (n,), "float32") for n in (2048, 1048576, 16777216)]
    cases += [
        ("batch", a, b, "float32")
        for a, b in (
            ((64, 32, 32), (64, 32, 32)),
            ((16, 128, 128), (16, 128, 128)),
            ((8, 1, 64, 64), (1, 4, 64, 64)),
        )
    ]
    report = {
        "origin": str(Path(p.__file__).resolve()),
        "numerical_precision": precision_metadata(),
        "diagnostics_timed": False,
        "repeats": repeats,
        "threads": threadpool_info(),
        "unavailable": errors,
        "rows": rows,
        "note": "warm synchronized wall; transfer mode uses fresh GPU inputs; CPU inputs reused; no DRAM-bandwidth claim for cache-fitting cases",
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def measure(adapter, fn):
        adapter.sync()
        start = time.perf_counter()
        value = fn()
        adapter.sync()
        return time.perf_counter() - start, value

    for case, ashape, bshape, dtype in cases:
        rng = np.random.default_rng(471)
        inputs = tuple(rng.uniform(-1, 1, shape).astype(dtype) for shape in (ashape, bshape))
        with threadpool_limits(limits=1):
            expected = inputs[0] @ inputs[1]
            diagnostic_reference = inputs[0].astype(np.float64) @ inputs[1].astype(np.float64)
        tolerance = 4e-3 if dtype == "float32" else 1e-9

        def check(actual, expected=expected, tolerance=tolerance):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)

        active = []
        for adapter in adapters:
            row = {
                "case": case,
                "left_shape": ashape,
                "right_shape": bshape,
                "dtype": dtype,
                "input_bytes": sum(a.nbytes for a in inputs),
                "implementation": adapter.name,
                "resident_seconds": [],
                "end_to_end_seconds": [],
            }
            rows.append(row)
            try:
                with threadpool_limits(limits=adapter.threads):
                    resident = tuple(adapter.upload(a) for a in inputs)
                    adapter.sync()
                    _, first = measure(adapter, lambda resident=resident: resident[0] @ resident[1])
                    first_host = adapter.host(first)
                    record_error(row, first_host, diagnostic_reference)
                    check(first_host)
                    del first
                    active.append((adapter, resident, row))
            except Exception as exc:  # noqa: BLE001 - preserve failed configurations
                row["error"] = str(exc)
        order = random.Random(47)
        for _ in range(repeats):
            order.shuffle(active)
            for adapter, resident, row in active:
                try:
                    with threadpool_limits(limits=adapter.threads):
                        elapsed, actual = measure(
                            adapter, lambda resident=resident: resident[0] @ resident[1]
                        )
                        actual_host = adapter.host(actual)
                        record_error(row, actual_host, diagnostic_reference)
                        check(actual_host)
                        row["resident_seconds"].append(elapsed)

                        def end_to_end(adapter=adapter, inputs=inputs):
                            a, b = (adapter.upload(value) for value in inputs)
                            return adapter.host(a @ b)

                        elapsed, actual = measure(adapter, end_to_end)
                        record_error(row, actual, diagnostic_reference)
                        check(actual)
                        row["end_to_end_seconds"].append(elapsed)
                except Exception as exc:  # noqa: BLE001
                    row["error"] = str(exc)
        for _, _, row in active:
            for mode in ("resident", "end_to_end"):
                samples = row[f"{mode}_seconds"]
                row[f"{mode}_median_seconds"] = statistics.median(samples) if samples else None
        save()
        print(f"{case} {ashape} {dtype}: {len(active)}/{len(adapters)}", flush=True)
        active.clear()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    report = run(args.output, args.repeats)
    raise SystemExit(bool(report["unavailable"]) or any("error" in r for r in report["rows"]))
