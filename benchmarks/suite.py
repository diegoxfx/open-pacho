"""Compare direct NumPy, unchanged OpenPacho/OpenCL, and native CuPy/CUDA.

Run from the repository root: python -m benchmarks.suite --output <directory>
No library kernels are modified or tuned by this experiment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import openpacho as pacho
from examples import solvers

# Do not gain float32 matmul speed by silently permitting TF32 arithmetic.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")


@dataclass(frozen=True)
class Case:
    operation: str
    shape: tuple[int, ...]
    dtype: str
    steps: int = 1

    @property
    def id(self):
        return f"{self.operation}-{'x'.join(map(str, self.shape))}-{self.dtype}-k{self.steps}"


def cases(profile="full"):
    result = []
    for dtype in ("float32", "float64"):
        for n in (1024, 16384, 262144, 1048576, 4194304, 16777216):
            for op in ("add", "sum", "transcendental", "distance"):
                result.append(Case(op, (n,), dtype))
        for n in (16384, 262144, 4194304):
            for k in (1, 10, 100):
                result.append(Case("reuse", (n,), dtype, k))
        for n in (16, 64, 256, 512, 1024, 2048):
            result.append(Case("matmul", (n, n, n), dtype))
        for shape in ((64, 1024, 64), (1024, 64, 1024), (256, 1024, 128), (1024, 1024, 64)):
            result.append(Case("matmul", shape, dtype))
        for n in (64, 256, 1024, 2048, 4096):
            result.append(Case("matvec", (n, n), dtype))
        for n in (64, 256, 1024, 4096):
            result.append(Case("transpose", (n, n), dtype))
        for n in (64, 256, 1024):
            for method in ("cg", "jacobi", "sd"):
                result.append(Case(method, (n, n), dtype))
    if profile == "smoke":
        return [
            Case(op, shape, dtype, steps)
            for dtype in ("float32", "float64")
            for op, shape, steps in (
                ("add", (1024,), 1),
                ("sum", (1024,), 1),
                ("transcendental", (1024,), 1),
                ("distance", (1024,), 1),
                ("reuse", (1024,), 10),
                ("matmul", (16, 17, 19), 1),
                ("matvec", (32, 32), 1),
                ("transpose", (32, 32), 1),
                ("cg", (16, 16), 1),
                ("jacobi", (16, 16), 1),
                ("sd", (16, 16), 1),
            )
        ]
    return result


def make_inputs(case, seed):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(case.dtype)

    def uniform(shape):
        return rng.uniform(-0.5, 0.5, shape).astype(dtype)

    if case.operation == "matmul":
        m, k, n = case.shape
        return (uniform((m, k)), uniform((k, n))), None
    if case.operation in ("cg", "jacobi", "sd"):
        n = case.shape[0]
        # Dense storage of a known SPD, diagonally dominant system; condition <= 9.
        A = np.diag(np.full(n, 2.5, dtype=dtype))
        A += np.diag(np.full(n - 1, -1, dtype=dtype), 1)
        A += np.diag(np.full(n - 1, -1, dtype=dtype), -1)
        exact = uniform((n,))
        return (A, A @ exact), exact
    if case.operation == "matvec":
        return (uniform(case.shape), uniform((case.shape[1],))), None
    if case.operation in ("sum", "transpose"):
        return (uniform(case.shape),), None
    return (uniform(case.shape), uniform(case.shape)), None


def native_solver(xp, method, A, b, rtol, maxiter=400, diagnostics=False):
    """Same normalized recurrence and true-residual stopping as the example solvers.

    Direct NumPy/CuPy operations, without the OpenPacho wrapper's overhead.
    This is not a comparison of iterative solving against a direct factorization.
    """
    x = xp.zeros_like(b)
    for value in (A, b, x):
        if xp.sum(xp.isfinite(value)).item() != value.size:
            raise ValueError("non-finite system")
    diagonal = xp.diag(A) if method == "jacobi" else None
    if diagonal is not None and xp.sum(diagonal == 0).item():
        raise ValueError("zero diagonal")

    def norm(v):
        if xp is np:
            scale = float(np.max(np.abs(v))) if v.size else 0.0
            if not scale or not math.isfinite(scale):
                return scale
            with np.errstate(over="ignore"):
                return float(
                    np.asarray(scale * float(np.sqrt(np.sum((v / scale) ** 2))), dtype=v.dtype)
                )
        # Strong CUDA reference: use cuBLAS NRM2's safe scaling, rather than a
        # fragile high-level sum(v*v). Host result replaces the usual item read.
        cublas = xp.cuda.cublas
        handle = xp.cuda.device.get_cublas_handle()
        result = np.empty((), dtype=v.dtype)
        previous = cublas.getPointerMode(handle)
        try:
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            routine = cublas.snrm2 if v.dtype == np.dtype("float32") else cublas.dnrm2
            routine(handle, v.size, v.data.ptr, 1, result.ctypes.data)
        finally:
            cublas.setPointerMode(handle, previous)
        return float(result)

    r = b - A @ x
    residual = norm(r)
    tolerance = rtol * norm(b)
    if not math.isfinite(residual) or not math.isfinite(tolerance):
        raise ValueError("nonfinite residual")
    if residual <= tolerance:
        return (x, 0) if diagnostics else x
    direction = r / residual if method == "cg" else None
    for iteration in range(1, maxiter + 1):
        if method == "jacobi":
            x = x + r / diagonal
        else:
            q = direction if method == "cg" else r / residual
            Ar = A @ q
            curvature = xp.dot(q, Ar).item()
            if not math.isfinite(curvature) or curvature <= 0:
                raise ValueError("nonpositive curvature")
            x = x + (residual / curvature) * q
        r = b - A @ x
        new_residual = norm(r)
        if not math.isfinite(new_residual):
            raise ValueError("nonfinite residual")
        if new_residual <= tolerance:
            return (x, iteration) if diagnostics else x
        if method == "cg":
            direction = r / new_residual + (new_residual / residual) * direction
        residual = new_residual
    raise ValueError("native solver did not converge")


def operation(case, xp, args, diagnostics=False):
    a = args[0]
    op = case.operation
    if op == "add":
        return a + args[1]
    if op == "sum":
        return xp.sum(a)
    if op == "transcendental":
        return xp.sin(a) * xp.exp(-(a * a)) + xp.cos(args[1])
    if op == "distance":
        diff = a - args[1]
        return xp.sum(diff * diff)
    if op == "reuse":
        for _ in range(case.steps):
            a = 0.99 * a + 0.01 * args[1]
        return a
    if op in ("matmul", "matvec"):
        return a @ args[1]
    if op == "transpose":
        # Equal work: materialize C-contiguous output, not NumPy's zero-copy view.
        return a.T if xp is pacho else a.T.copy(order="C")
    rtol = 1e-4 if case.dtype == "float32" else 1e-7
    if xp is pacho:
        solver = {
            "cg": solvers.conjugate_gradient,
            "jacobi": solvers.jacobi,
            "sd": solvers.steepest_descent,
        }[op]
        result = solver(*args, rtol=rtol, maxiter=400)
        if not result.converged:
            raise ValueError(f"OpenPacho solver failed: {result.reason}")
        return (result.x, result.iterations) if diagnostics else result.x
    return native_solver(xp, op, *args, rtol=rtol, diagnostics=diagnostics)


class Adapter:
    def __init__(self, name):
        self.name = name
        self.cpu_threads = None
        if name.startswith("numpy"):
            self.xp = np
            self.cpu_threads = int(name.split("-")[1])
        elif name in ("opencl", "opencl-fused", "opencl-clblast"):
            self.xp = pacho
            self.engine = pacho.get_backend(
                "opencl", blas="clblast" if name == "opencl-clblast" else "custom"
            )
        elif name == "cuda":
            import cupy

            self.xp = cupy
            cupy.cuda.Device(0).use()
            cupy.cuda.runtime.deviceSynchronize()
        else:
            raise ValueError(f"Unknown backend {name}")

    def sync(self):
        if self.name.startswith("opencl"):
            self.engine.synchronize()
        elif self.name == "cuda":
            self.xp.cuda.get_current_stream().synchronize()

    def upload(self, args):
        if self.name.startswith("opencl"):
            return tuple(pacho.asarray(x, backend=self.engine) for x in args)
        return tuple(self.xp.asarray(x) for x in args)

    def host(self, value):
        if self.name.startswith("opencl"):
            return pacho.asnumpy(value)
        return self.xp.asnumpy(value) if self.name == "cuda" else np.asarray(value)

    def compute(self, case, args, diagnostics=False):
        if self.name == "opencl-fused":
            if case.operation == "transcendental":
                return _fused_transcendental(*args)
            if case.operation == "reuse":
                a, b = args
                for _ in range(case.steps):
                    a = _fused_update(a, b)
                return a
        return operation(case, self.xp, args, diagnostics=diagnostics)

    def cleanup(self):
        self.sync()
        if self.name == "cuda":
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.get_default_pinned_memory_pool().free_all_blocks()


@pacho.fuse
def _fused_transcendental(a, b):
    return pacho.sin(a) * pacho.exp(-(a * a)) + pacho.cos(b)


@pacho.fuse
def _fused_update(a, b):
    return 0.99 * a + 0.01 * b


def validate(case, actual, reference, inputs, exact):
    expected_shape = reference.shape
    if actual.shape != expected_shape or actual.dtype != reference.dtype:
        return {"valid": False, "reason": "shape_or_dtype"}
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    max_error = float(np.max(np.abs(delta)))
    scale = max(float(np.max(np.abs(reference))), 1e-30)
    normalized_error = max_error / scale
    rtol, atol = (3e-4, 3e-5) if case.dtype == "float32" else (1e-9, 1e-10)
    valid = bool(
        np.all(np.isfinite(actual)) and np.allclose(actual, reference, rtol=rtol, atol=atol)
    )
    result = {
        "valid": valid,
        "max_abs_error": max_error,
        "normalized_max_error": normalized_error,
        "rtol": rtol,
        "atol": atol,
    }
    if exact is not None:
        A, b = inputs
        residual = float(
            np.linalg.norm(b.astype(np.float64) - A.astype(np.float64) @ actual.astype(np.float64))
        )
        relative = residual / float(np.linalg.norm(b.astype(np.float64)))
        tolerance = 1e-4 if case.dtype == "float32" else 1e-7
        result.update(
            relative_residual=relative,
            solution_relative_error=float(np.linalg.norm(actual - exact) / np.linalg.norm(exact)),
        )
        # Different backends may stop on adjacent iterations; accept the requested residual.
        result["valid"] = bool(np.all(np.isfinite(actual)) and relative <= tolerance * 1.05)
    return result


def timed(adapter, fn, batch=1):
    adapter.sync()
    start = time.perf_counter_ns()
    for _ in range(batch):
        output = fn()
        adapter.sync()  # Includes all queued GPU work in EVERY invocation.
        del output
    return (time.perf_counter_ns() - start) * 1e-9 / batch


def stats(samples):
    return {
        "median_s": statistics.median(samples),
        "p10_s": float(np.percentile(samples, 10)),
        "p90_s": float(np.percentile(samples, 90)),
        "min_s": min(samples),
        "max_s": max(samples),
    }


def run_case(case, adapters, rounds, seed, target_s, max_batch):
    inputs, exact = make_inputs(case, seed)
    with threadpool_limits(limits=8, user_api="blas"):
        reference = np.asarray(operation(case, np, inputs))
    trials = []
    owned = []
    for adapter in adapters:
        if adapter.cpu_threads is not None:
            trials.append(
                {
                    "adapter": adapter,
                    "mode": "cpu",
                    "fn": lambda ad=adapter: ad.compute(case, inputs),
                }
            )
            continue
        resident = adapter.upload(inputs)
        adapter.sync()
        first_s = timed(adapter, lambda ad=adapter, ar=resident: ad.compute(case, ar))
        value = adapter.compute(case, resident)
        adapter.sync()
        actual = adapter.host(value)
        check = validate(case, actual, reference, inputs, exact)
        if exact is not None:
            _, check["iterations"] = adapter.compute(case, resident, diagnostics=True)
            _, check["numpy_iterations"] = operation(case, np, inputs, diagnostics=True)
        if not check["valid"]:
            raise AssertionError(f"{adapter.name} {case.id}: {check}")
        owned.extend((resident, value))

        def end_to_end(ad=adapter):
            uploaded = ad.upload(inputs)
            result = ad.compute(case, uploaded)
            return ad.host(result)

        # Validate the timed end-to-end path separately, outside the timer.
        e2e_check = validate(case, end_to_end(), reference, inputs, exact)
        if not e2e_check["valid"]:
            raise AssertionError(f"{adapter.name} end-to-end {case.id}: {e2e_check}")
        for mode, fn in (
            ("resident", lambda ad=adapter, ar=resident: ad.compute(case, ar)),
            ("end_to_end", end_to_end),
            ("upload", lambda ad=adapter: ad.upload(inputs)),
            ("download", lambda ad=adapter, v=value: ad.host(v)),
        ):
            trials.append(
                {
                    "adapter": adapter,
                    "mode": mode,
                    "fn": fn,
                    "validation": check
                    if mode == "resident"
                    else e2e_check
                    if mode == "end_to_end"
                    else {
                        "valid": None,
                        "scope": "transfer-only; payload checked separately in transport suite",
                    },
                    "first_call_s": first_s if mode == "resident" else None,
                }
            )
    for trial in trials:
        adapter = trial["adapter"]
        with threadpool_limits(limits=adapter.cpu_threads or 8, user_api="blas"):
            trial["fn"]()
            estimate = timed(adapter, trial["fn"])
        trial["batch"] = max(1, min(max_batch, math.ceil(target_s / max(estimate, 1e-9))))
        trial["samples_s"] = []
    ordering = random.Random(seed + 98)
    for _ in range(rounds):
        ordering.shuffle(trials)
        for trial in trials:
            adapter = trial["adapter"]
            with threadpool_limits(limits=adapter.cpu_threads or 8, user_api="blas"):
                trial["samples_s"].append(timed(adapter, trial["fn"], trial["batch"]))
    rows = []
    for trial in trials:
        row = {
            "case_id": case.id,
            **asdict(case),
            "backend": trial["adapter"].name,
            "mode": trial["mode"],
            "batch": trial["batch"],
            "samples_s": trial["samples_s"],
            "input_bytes": sum(a.nbytes for a in inputs),
            "output_bytes": reference.nbytes,
            "validation": trial.get("validation", {"valid": True, "reference": "numpy"}),
            "first_call_s": trial.get("first_call_s"),
            **stats(trial["samples_s"]),
        }
        rows.append(row)
    del trials, owned
    gc.collect()
    for adapter in adapters:
        adapter.cleanup()
    return rows


def transport(adapters, rounds):
    """Pageable host memory, existing device/host buffers: no array allocation in timing."""
    result = []
    for size in (4096, 1048576, 16777216, 67108864, 268435456):
        host = np.ones(size // 4, dtype=np.float32)
        dest = np.empty_like(host)
        for ad in adapters:
            if ad.cpu_threads:
                continue
            device = ad.upload((host,))[0]
            if ad.name == "opencl":

                def h2d(ad=ad, device=device, host=host):
                    return ad.engine.cl.enqueue_copy(
                        ad.engine.queue, device._data, host, is_blocking=True
                    )

                def d2h(ad=ad, device=device, dest=dest):
                    return ad.engine.cl.enqueue_copy(
                        ad.engine.queue, dest, device._data, is_blocking=True
                    )
            else:

                def h2d(device=device, host=host):
                    return device.set(host)

                def d2h(device=device, dest=dest):
                    return device.get(out=dest)

            for mode, fn in (("h2d", h2d), ("d2h", d2h)):
                fn()
                ad.sync()
                samples = [timed(ad, fn) for _ in range(rounds)]
                if mode == "d2h":
                    np.testing.assert_array_equal(dest, host)
                result.append(
                    {
                        "backend": ad.name,
                        "mode": mode,
                        "bytes": size,
                        "samples_s": samples,
                        **stats(samples),
                        "GB_per_s": size / statistics.median(samples) / 1e9,
                    }
                )
            del device
            ad.cleanup()
    return result


def source_hashes():
    paths = list(Path("src/openpacho").glob("*.py")) + [Path(solvers.__file__), Path(__file__)]
    return {str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def telemetry():
    query = "name,temperature.gpu,utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw,memory.used,pstate"
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--target-ms", type=float, default=2)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=470962)
    parser.add_argument("--cpu-threads", default="1,8,16")
    parser.add_argument("--backends", default="opencl,cuda")
    parser.add_argument("--only", help="comma-separated operation names")
    parser.add_argument(
        "--solver-sizes", help="additional solver-only square sizes, comma separated"
    )
    parser.add_argument("--skip-transport", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds < 3 or args.target_ms <= 0 or args.max_batch < 1:
        parser.error("rounds>=3, target-ms>0 and max-batch>=1 required")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "raw.json").exists():
        parser.error("output already contains raw.json; use a new directory")
    thread_counts = [int(x) for x in args.cpu_threads.split(",")]
    if min(thread_counts) < 1:
        parser.error("CPU thread counts must be positive")
    names = [f"numpy-{n}" for n in thread_counts] + [x for x in args.backends.split(",") if x]
    adapters, init_s = [], {}
    for name in names:
        start = time.perf_counter()
        adapters.append(Adapter(name))
        init_s[name] = time.perf_counter() - start
    selected = cases(args.profile)
    if args.solver_sizes:
        sizes = [int(n) for n in args.solver_sizes.split(",")]
        if min(sizes) < 1:
            parser.error("solver sizes must be positive")
        selected = [
            Case(method, (n, n), dtype)
            for dtype in ("float32", "float64")
            for n in sizes
            for method in ("cg", "jacobi", "sd")
        ]
    if args.only:
        selected = [c for c in selected if c.operation in args.only.split(",")]
    random.Random(args.seed).shuffle(selected)
    meta = {
        "schema_version": 1,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "threadpoolctl")},
        "python": platform.python_version(),
        "blas": threadpool_info(),
        "source_sha256": source_hashes(),
        "initialization_s": init_s,
        "arguments": {**vars(args), "output": str(args.output)},
        "opencl_devices": [asdict(d) for d in pacho.devices()] if "opencl" in names else [],
        "case_order": [c.id for c in selected],
        "telemetry_start": telemetry(),
    }
    if "cuda" in names:
        import cupy as cp

        meta["versions"]["cupy"] = cp.__version__
        meta["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        meta["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
        meta["cuda_accelerators"] = os.environ.get("CUPY_ACCELERATORS", "default")
        meta["cuda_tf32_override"] = os.environ.get("NVIDIA_TF32_OVERRIDE", "unset")
        properties = cp.cuda.runtime.getDeviceProperties(0)
        meta["cuda_device"] = {
            "index": 0,
            "name": properties["name"].decode(),
            "compute_capability": [properties["major"], properties["minor"]],
        }
    if "opencl" in names:
        meta["versions"]["pyopencl"] = importlib.metadata.version("pyopencl")
        meta["selected_opencl_device"] = next(
            ad.engine.device for ad in adapters if ad.name == "opencl"
        )
    meta["timing_policy"] = {
        "cpu_mode": "direct NumPy, host inputs and host output (zero-transfer end-to-end baseline)",
        "resident": "large arrays resident; includes dispatch, allocation, synchronization, output release and scalar uploads",
        "end_to_end": "host inputs through upload, compute, download and temporary release; initialized runtime",
        "allocation": "native defaults: OpenCL MemoryPool; CuPy default device/pinned pools; NumPy allocator",
        "transport": "fixed ascending byte sizes, OpenCL then CUDA; existing buffers and pageable host memory",
    }
    report = {"metadata": meta, "rows": [], "transport": [], "telemetry": [], "errors": []}
    start = time.perf_counter()
    for i, case in enumerate(selected):
        print(f"[{i + 1}/{len(selected)}] {case.id}", flush=True)
        try:
            report["rows"].extend(
                run_case(
                    case,
                    adapters,
                    args.rounds,
                    args.seed + i,
                    args.target_ms / 1000,
                    args.max_batch,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- retain every failed case and exit nonzero
            report["errors"].append(
                {"case_id": case.id, "type": type(exc).__name__, "message": str(exc)}
            )
            print(f"ERROR {case.id}: {exc}", flush=True)
        if i % 10 == 0:
            report["telemetry"].append({"after_case": i + 1, "snapshot": telemetry()})
        (args.output / "checkpoint.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    report["transport"] = [] if args.skip_transport else transport(adapters, args.rounds)
    meta["elapsed_s"] = time.perf_counter() - start
    meta["telemetry_end"] = telemetry()
    meta["source_unchanged"] = source_hashes() == meta["source_sha256"]
    (args.output / "raw.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"Done: {len(report['rows'])} rows, {len(report['errors'])} errors, {meta['elapsed_s']:.1f}s",
        flush=True,
    )
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
