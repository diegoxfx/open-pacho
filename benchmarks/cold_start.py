"""Fresh-process measurements, explicitly separated from steady-state throughput."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def worker(backend, dtype, size):
    start = time.perf_counter()
    import numpy as np
    from threadpoolctl import threadpool_limits

    threads = int(backend.split("-")[1]) if backend.startswith("numpy-") else 8
    if backend.startswith("numpy-"):
        upload, download = np.asarray, np.asarray

        def sync():
            pass
    elif backend == "opencl":
        import openpacho as pacho

        engine = pacho.get_backend("opencl")

        def upload(value):
            return pacho.asarray(value, backend=engine)

        download, sync = pacho.asnumpy, engine.synchronize
    else:
        import cupy as cp

        cp.cuda.Device(0).use()
        upload, download, sync = cp.asarray, cp.asnumpy, cp.cuda.get_current_stream().synchronize
    sync()
    initialization = time.perf_counter() - start
    rng = np.random.default_rng(47)
    inputs = tuple(rng.uniform(-0.5, 0.5, (size, size)).astype(dtype) for _ in range(2))
    with threadpool_limits(limits=threads, user_api="blas"):
        start = time.perf_counter()
        values = tuple(upload(x) for x in inputs)
        output = values[0] @ values[1]
        host = download(output)
        sync()
        first = time.perf_counter() - start
        expected = inputs[0] @ inputs[1]
        valid = bool(np.allclose(host, expected, rtol=3e-4, atol=3e-5))
    return {
        "backend": backend,
        "dtype": dtype,
        "size": size,
        "initialization_s": initialization,
        "first_end_to_end_s": first,
        "startup_plus_first_s": initialization + first,
        "valid": valid,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.dtype, args.size)))
        return
    if args.output is None:
        parser.error("--output is required")
    rows = []
    for dtype in ("float32", "float64"):
        for backend in ("numpy-1", "numpy-8", "numpy-16", "opencl", "cuda"):
            for i in range(args.rounds):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.cold_start",
                        "--worker",
                        backend,
                        "--dtype",
                        dtype,
                        "--size",
                        str(args.size),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=os.environ.copy(),
                )
                row = json.loads(result.stdout)
                row["round"] = i
                rows.append(row)
                print(f"{backend} {dtype}: {row['startup_plus_first_s']:.3f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"cache_policy": "fresh processes; existing disk caches retained", "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
