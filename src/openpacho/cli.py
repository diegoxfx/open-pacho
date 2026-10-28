"""Device diagnostics and synchronized, reproducible CPU/GPU benchmarks."""

import argparse
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import _array as xp
from .backends import devices, get_backend
from .exceptions import BackendUnavailableError, UnsupportedDtypeError


def benchmark(*, size=128, repeat=5, dtype="float32", backend="opencl", device=None, seed=47):
    """Measure construction, cold dispatch, warm resident execution, and download separately.

    Timings are synchronized wall-clock seconds, including Python dispatch/allocation.
    Every timed case is checked against NumPy. No speedup claim is made implicitly.
    """
    if size < 1 or repeat < 1:
        raise ValueError("size and repeat must be positive")
    if dtype not in ("float32", "float64"):
        raise ValueError("benchmark dtype must be float32 or float64")
    engine = get_backend(backend, device=device)
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (size, size)).astype(dtype)
    b = rng.uniform(-1, 1, (size, size)).astype(dtype)
    engine.synchronize()
    start = time.perf_counter()
    da, db = xp.asarray(a, backend=engine), xp.asarray(b, backend=engine)
    engine.synchronize()
    upload = time.perf_counter() - start
    cases = []
    for name, device_fn, cpu_fn in (
        ("add", lambda: da + db, lambda: a + b),
        ("matmul", lambda: da @ db, lambda: a @ b),
        ("sum", lambda: da.sum(), lambda: a.sum()),
    ):
        engine.synchronize()
        start = time.perf_counter()
        actual = device_fn()
        engine.synchronize()
        cold = time.perf_counter() - start
        start = time.perf_counter()
        host = xp.asnumpy(actual)
        download = time.perf_counter() - start
        reference = cpu_fn()
        tolerance = 5e-4 if dtype == "float32" else 1e-10
        valid = bool(np.allclose(host, reference, rtol=tolerance, atol=tolerance))
        warm, cpu = [], []
        for _ in range(repeat):
            engine.synchronize()
            start = time.perf_counter()
            actual = device_fn()
            engine.synchronize()
            warm.append(time.perf_counter() - start)
            start = time.perf_counter()
            cpu_fn()
            cpu.append(time.perf_counter() - start)
        cases.append(
            {
                "operation": name,
                "valid": valid,
                "max_abs_error": float(np.max(np.abs(host - reference))),
                "cold_seconds": cold,
                "download_seconds": download,
                "resident_seconds": warm,
                "resident_median_seconds": statistics.median(warm),
                "numpy_seconds": cpu,
                "numpy_median_seconds": statistics.median(cpu),
            }
        )
    versions = {"numpy": np.__version__, "python": platform.python_version()}
    if engine.name in ("opencl", "vulkan", "cuda"):
        package = {"opencl": "pyopencl", "vulkan": "vulkan", "cuda": "cuda-bindings"}[engine.name]
        versions[package] = importlib.metadata.version(package)
    return {
        "schema_version": 1,
        "backend": engine.name,
        "device": engine.device,
        "os": platform.platform(),
        "versions": versions,
        "dtype": dtype,
        "size": size,
        "seed": seed,
        "repeat": repeat,
        "upload_seconds": upload,
        "cases": cases,
        "all_valid": all(c["valid"] for c in cases),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pacho", description="OpenPacho device diagnostics and benchmarks"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("devices", help="List GPU devices; no interactive selection")
    discover.add_argument("--json", action="store_true")
    discover.add_argument("--backend", choices=("opencl", "vulkan", "cuda"), default="opencl")
    bench = sub.add_parser("benchmark", help="Compare synchronized operations with NumPy")
    bench.add_argument("--backend", choices=("numpy", "opencl", "vulkan", "cuda"), default="opencl")
    bench.add_argument("--device", help="Device ID from 'pacho devices'")
    bench.add_argument("--size", type=int, default=128)
    bench.add_argument("--repeat", type=int, default=5)
    bench.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    bench.add_argument("--seed", type=int, default=47)
    bench.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "devices":
            found = devices(args.backend)
            if args.json:
                print(json.dumps([asdict(d) for d in found], indent=2))
            else:
                for d in found:
                    print(f"{d.id}  {d.kind}  {d.name}  ({d.vendor})  float64={d.fp64}")
                if not found:
                    print(f"No {args.backend} devices found.")
            return 0
        report = benchmark(
            size=args.size,
            repeat=args.repeat,
            dtype=args.dtype,
            backend=args.backend,
            device=args.device,
            seed=args.seed,
        )
        output = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        print(output)
        return 0 if report["all_valid"] else 1
    except (BackendUnavailableError, UnsupportedDtypeError, ValueError) as exc:
        print(f"pacho: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
