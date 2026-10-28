"""Diagnose Python, backend dispatch and resident GPU execution costs.

The wall samples in this report are deliberately uninstrumented: every sample
synchronizes immediately before and after one resident operation.  cProfile
and backend method counters run in a separate invocation after warmup, so their
overhead cannot contaminate the timing samples.  The four workloads use the
existing :mod:`benchmarks.execution` definitions where they apply:
GEMV, DOT, compiled energy, and the conjugate-gradient application in examples/solvers.py.

Example::

    python -m benchmarks.gaps --output .artifacts/gaps.json \
        --backends opencl,opencl-clblast,vulkan,cupy --repeats 7 --size 2048

Requested optional backends are never silently skipped.  An unavailable
backend is recorded in ``backend_errors`` and the command exits non-zero after
writing the report.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import importlib.metadata
import json
import os
import platform
import pstats
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import openpacho as pacho
from benchmarks.execution import Runner
from benchmarks.execution import energy as execution_energy
from benchmarks.suite import native_solver

CALL_NAMES = (
    "empty",
    "upload",
    "download",
    "elementwise",
    "reduce",
    "matmul",
    "_run",
    "synchronize",
)
BACKEND_NAMES = ("opencl", "opencl-clblast", "vulkan", "cupy")
WORKLOADS = ("gemv", "dot", "compiled_energy", "cg")


def _ensure_cupy_cache() -> None:
    """Keep optional CuPy compilation artifacts in this task's workspace."""

    if os.environ.get("CUPY_CACHE_DIR"):
        return
    cache = Path.cwd() / ".artifacts" / "gaps" / "cupy-cache"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["CUPY_CACHE_DIR"] = str(cache)
    except OSError:
        # CuPy will report a clear driver/cache error if no writable fallback
        # exists; do not make importing CPU-only benchmark helpers fail.
        return


@dataclass(frozen=True)
class Case:
    name: str
    size: int
    dtype: np.dtype


@dataclass
class CallCounter:
    """Exact logical OpenPacho backend calls made by one profiled invocation."""

    counts: dict[str, int]
    originals: dict[str, Any]

    @classmethod
    def install(cls, backend: Any) -> CallCounter | None:
        if backend is None:
            return None
        counter = cls({name: 0 for name in CALL_NAMES}, {})
        for name in CALL_NAMES:
            original = getattr(backend, name, None)
            if not callable(original):
                continue
            counter.originals[name] = original

            def counted(*args, __name=name, __original=original, **kwargs):
                counter.counts[__name] += 1
                return __original(*args, **kwargs)

            setattr(backend, name, counted)
        return counter

    def uninstall(self, backend: Any) -> None:
        for name, original in self.originals.items():
            setattr(backend, name, original)


class Adapter:
    """Common upload/host/sync surface for the selected backend."""

    def __init__(self, name: str):
        self.name = name
        self.engine = None
        self.xp = None
        self._runner_cache: dict[tuple[str, tuple[int, ...]], Runner] = {}
        if name in ("opencl", "opencl-clblast"):
            self.xp = pacho
            options = {"blas": "clblast" if name == "opencl-clblast" else "custom"}
            if name == "opencl-clblast":
                library = os.environ.get("OPENPACHO_CLBLAST_LIBRARY")
                if library:
                    options["clblast_library"] = library
            self.engine = pacho.get_backend("opencl", **options)
        elif name == "vulkan":
            self.xp = pacho
            self.engine = pacho.get_backend("vulkan")
        elif name == "cupy":
            _ensure_cupy_cache()
            import cupy as cp

            self.xp = cp
            cp.cuda.Device(0).use()
            cp.cuda.get_current_stream().synchronize()
        else:
            raise ValueError(f"Unknown backend {name!r}")

    @property
    def is_pacho(self) -> bool:
        return self.engine is not None

    def sync(self) -> None:
        if self.is_pacho:
            self.engine.synchronize()
        else:
            self.xp.cuda.get_current_stream().synchronize()

    def upload(self, value: np.ndarray):
        return (
            pacho.asarray(value, backend=self.engine) if self.is_pacho else self.xp.asarray(value)
        )

    def host(self, value):
        return pacho.asnumpy(value) if self.is_pacho else self.xp.asnumpy(value)

    def _runner_for(self, case: Case, host_inputs: tuple[np.ndarray, ...]) -> tuple[Runner, bool]:
        """Reuse one Runner and compiled callable for one workload/input set."""

        key = (case.name, tuple(id(value) for value in host_inputs))
        runner = self._runner_cache.get(key)
        if runner is not None:
            return runner, True
        if case.name == "compiled_energy":
            runner = Runner("gaps-energy", "energy", host_inputs, engine=self.engine, compiled=True)
        elif case.name == "cg":
            runner = Runner("gaps-cg", "cg_solve", host_inputs, engine=self.engine)
        else:
            raise ValueError(f"Runner cache is only used for compiled workloads: {case.name}")
        self._runner_cache[key] = runner
        return runner, False

    def compute(self, case: Case, resident: tuple[Any, ...], host_inputs: tuple[np.ndarray, ...]):
        if case.name == "gemv":
            return resident[0] @ resident[1], {}
        if case.name == "dot":
            value = pacho.dot(*resident) if self.is_pacho else self.xp.dot(*resident)
            return value, {}
        if case.name == "compiled_energy":
            if self.is_pacho:
                # Runner owns the same energy function used by the execution
                # benchmark and applies pacho.compile to it.
                runner, reused = self._runner_for(case, host_inputs)
                value = runner.compute(resident)
            else:
                a, b, c = resident
                x = a * b + c
                y = 2 * x
                value = self.xp.sum(y * y, axis=-1)
                reused = False
            return value, {
                "compiled": self.is_pacho,
                "runner_reused": reused,
                "runner_cache": self.is_pacho,
            }
        if case.name == "cg":
            if self.is_pacho:
                runner, reused = self._runner_for(case, host_inputs)
                value = runner.compute(resident)
                return value, {
                    "converged": True,
                    "iterations": int(runner.iterations),
                    "public_solver": False,
                    "solver_implementation": "examples.solvers.conjugate_gradient",
                    "runner_reused": reused,
                }
            rtol = 1e-4 if case.dtype == np.dtype("float32") else 1e-7
            value, iterations = native_solver(
                self.xp,
                "cg",
                resident[0],
                resident[1],
                rtol=rtol,
                maxiter=400,
                diagnostics=True,
            )
            return value, {
                "converged": True,
                "iterations": int(iterations),
                "public_solver": False,
                "solver_implementation": "benchmarks.suite.native_solver",
                "rtol": rtol,
                "maxiter": 400,
            }
        raise ValueError(case.name)


def _solver_inputs(size: int, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a well-conditioned dense SPD system that converges quickly."""

    rng = np.random.default_rng(47 + size)
    a = np.eye(size, dtype=dtype) * dtype.type(2.0)
    if size > 1:
        indices = np.arange(size - 1)
        a[indices, indices + 1] = dtype.type(-0.01)
        a[indices + 1, indices] = dtype.type(-0.01)
    exact = rng.uniform(-0.5, 0.5, size).astype(dtype)
    b = (a @ exact).astype(dtype, copy=False)
    return a, b, exact


def _inputs(case: Case) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(47 + case.size * 13)
    if case.name == "gemv":
        return (
            rng.uniform(-1, 1, (case.size, case.size)).astype(case.dtype),
            rng.uniform(-1, 1, case.size).astype(case.dtype),
        )
    if case.name == "dot":
        return (
            rng.uniform(-1, 1, case.size).astype(case.dtype),
            rng.uniform(-1, 1, case.size).astype(case.dtype),
        )
    if case.name == "compiled_energy":
        # This shape is the existing execution benchmark's energy workload.
        return tuple(rng.uniform(-1, 1, (case.size, 32)).astype(case.dtype) for _ in range(3))
    if case.name == "cg":
        return _solver_inputs(case.size, case.dtype)[:2]
    raise ValueError(case.name)


def _expected(case: Case, inputs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    if case.name == "gemv":
        return inputs[0] @ inputs[1], {}
    if case.name == "dot":
        return np.asarray(np.dot(*inputs), dtype=case.dtype), {}
    if case.name == "compiled_energy":
        return execution_energy(*inputs), {"axis": -1}
    if case.name == "cg":
        return _solver_inputs(case.size, case.dtype)[2], {"known_solution": True}
    raise ValueError(case.name)


def _validate(
    case: Case, actual: np.ndarray, expected: np.ndarray, inputs: tuple[np.ndarray, ...]
) -> dict[str, Any]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    rtol, atol = (8e-4, 8e-4) if case.dtype == np.dtype("float32") else (2e-10, 2e-10)
    result: dict[str, Any] = {
        "valid": False,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "rtol": rtol,
        "atol": atol,
    }
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        result["reason"] = "shape_or_dtype"
        return result
    if case.name == "cg":
        a, b = inputs
        residual = float(
            np.linalg.norm(b.astype(np.float64) - a.astype(np.float64) @ actual.astype(np.float64))
        )
        relative = residual / max(float(np.linalg.norm(b.astype(np.float64))), 1e-30)
        limit = 1e-4 if case.dtype == np.dtype("float32") else 1e-7
        result.update(
            residual_norm=residual,
            relative_residual=relative,
            residual_limit=limit,
            valid=bool(np.all(np.isfinite(actual)) and relative <= limit * 1.05),
        )
    else:
        result.update(
            max_abs_error=float(
                np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
            ),
            valid=bool(
                np.all(np.isfinite(actual)) and np.allclose(actual, expected, rtol=rtol, atol=atol)
            ),
        )
    return result


def _sample_stats(samples: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    return {
        "samples_seconds": values,
        "median_seconds": float(statistics.median(values)) if values else None,
        "p10_seconds": float(np.percentile(values, 10)) if values else None,
        "p90_seconds": float(np.percentile(values, 90)) if values else None,
        "min_seconds": min(values) if values else None,
        "max_seconds": max(values) if values else None,
    }


def _measure_resident(adapter: Adapter, fn: Callable[[], Any], repeats: int) -> dict[str, Any]:
    warm = fn()
    adapter.sync()
    del warm
    samples = []
    for _ in range(repeats):
        adapter.sync()
        started = time.perf_counter_ns()
        value = fn()
        adapter.sync()
        samples.append((time.perf_counter_ns() - started) * 1e-9)
        del value
    return _sample_stats(samples)


def _measure_transfers(
    adapter: Adapter, inputs: tuple[np.ndarray, ...], repeats: int
) -> dict[str, Any]:
    resident = tuple(adapter.upload(value) for value in inputs)
    adapter.sync()
    uploads, downloads = [], []
    for _ in range(repeats):
        adapter.sync()
        started = time.perf_counter_ns()
        uploaded = tuple(adapter.upload(value) for value in inputs)
        adapter.sync()
        uploads.append((time.perf_counter_ns() - started) * 1e-9)
        del uploaded
        started = time.perf_counter_ns()
        adapter.host(resident[0])
        downloads.append((time.perf_counter_ns() - started) * 1e-9)
    del resident
    return {"upload": _sample_stats(uploads), "download": _sample_stats(downloads)}


def _profile_top(profile: cProfile.Profile, limit: int = 15) -> list[dict[str, Any]]:
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _callers) in pstats.Stats(
        profile
    ).stats.items():
        rows.append(
            {
                "file": filename,
                "line": line,
                "function": function,
                "calls": int(calls),
                "primitive_calls": int(primitive),
                "own_host_seconds": float(own),
                "cumulative_host_seconds": float(cumulative),
            }
        )
    rows.sort(
        key=lambda row: (row["cumulative_host_seconds"], row["own_host_seconds"]), reverse=True
    )
    return rows[:limit]


def _profile_invocation(adapter: Adapter, fn: Callable[[], Any]) -> dict[str, Any]:
    """Profile exactly one already-warmed call, separately from wall samples."""

    counter = CallCounter.install(adapter.engine)
    profile = cProfile.Profile()
    try:
        profile.enable()
        value = fn()
        adapter.sync()
        profile.disable()
    finally:
        if counter is not None:
            counter.uninstall(adapter.engine)
    del value
    if counter is None:
        counts = {name: 0 for name in CALL_NAMES}
        scope = "CuPy control has no OpenPacho backend methods"
    else:
        counts = dict(counter.counts)
        scope = "exact OpenPacho backend method calls during this profiled invocation"
    return {
        "profiling_overhead_separate": True,
        "cprofile": {
            "top": _profile_top(profile),
            "time_definition": "own_host_seconds=pstats tt; cumulative_host_seconds=pstats ct; default cProfile timer is wall elapsed and includes blocking waits, not CPU utilization",
            "inclusive_times_are_not_additive": True,
        },
        "logical_backend_calls": counts,
        "logical_backend_call_scope": scope,
    }


def _device_properties(adapter: Adapter) -> dict[str, Any]:
    if adapter.name.startswith("opencl"):
        device = adapter.engine.context.devices[0]

        def read(name: str, default=None):
            try:
                value = getattr(device, name)
                return (
                    list(value)
                    if isinstance(value, tuple)
                    else getattr(value, "item", lambda: value)()
                )
            except Exception:  # noqa: BLE001 - device property support varies by driver
                return default

        return {
            "api": "opencl",
            "name": read("name", adapter.engine.device),
            "vendor": read("vendor"),
            "driver": read("driver_version"),
            "global_memory_bytes": read("global_mem_size"),
            "max_work_group_size": read("max_work_group_size"),
            "max_work_item_sizes": read("max_work_item_sizes"),
            "local_memory_bytes": read("local_mem_size"),
            "fp64_supported": bool(getattr(adapter.engine, "fp64", False)),
            "pci_link": None,
        }
    if adapter.name == "vulkan":
        limits = getattr(getattr(adapter.engine, "_properties", None), "limits", None)
        read = lambda name: getattr(limits, name, None) if limits is not None else None
        return {
            "api": "vulkan",
            "push_descriptor_limit": getattr(adapter.engine, "_push_descriptor_limit", 0),
            "name": adapter.engine.device,
            "fp64_supported": bool(getattr(adapter.engine, "fp64", False)),
            "max_work_group_size": read("maxComputeWorkGroupSize"),
            "max_work_group_counts": read("maxComputeWorkGroupCount"),
            "local_memory_bytes": read("maxComputeSharedMemorySize"),
            "pci_link": None,
        }
    try:
        props = adapter.xp.cuda.runtime.getDeviceProperties(adapter.xp.cuda.Device().id)
        clean = {
            key: value.decode(errors="replace").rstrip("\0") if isinstance(value, bytes) else value
            for key, value in props.items()
        }
        return {
            "api": "cuda",
            "name": clean.get("name"),
            "runtime_properties": clean,
            "max_work_group_size": clean.get("maxThreadsPerBlock"),
            "local_memory_bytes": clean.get("sharedMemPerBlock"),
            "fp64_supported": True,
            "pci_link": None,
        }
    except Exception as exc:  # noqa: BLE001 - optional CUDA runtime
        return {"api": "cuda", "properties_available": False, "reason": str(exc), "pci_link": None}


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("open-pacho", "pyopencl", "vulkan", "cupy-cuda13x"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _parse_backends(value: str) -> list[str]:
    selected = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not selected or any(part not in BACKEND_NAMES for part in selected):
        raise argparse.ArgumentTypeError(
            "backends must be a comma-separated subset of opencl,opencl-clblast,vulkan,cupy"
        )
    if len(selected) != len(set(selected)):
        raise argparse.ArgumentTypeError("--backends contains duplicates")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", type=_parse_backends, default=list(BACKEND_NAMES))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--size", type=int, default=2048)
    return parser


def run(output: Path, backends: list[str], repeats: int, size: int) -> tuple[dict[str, Any], int]:
    dtype = np.dtype("float32")
    cases = [Case(name, size, dtype) for name in WORKLOADS]
    report: dict[str, Any] = {
        "schema": "openpacho.gaps.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_origin": str(Path(__file__).resolve()),
        "openpacho_origin": str(Path(pacho.__file__).resolve()),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "versions": _versions(),
        "configuration": {
            "backends": backends,
            "repeats": repeats,
            "size": size,
            "dtype": str(dtype),
            "resident_timing": "synchronized wall samples with warmup outside the clock",
            "profiling": "separate one-call cProfile and logical method-count run",
            "theoretical_flops": None,
        },
        "backend_errors": [],
        "backends": [],
    }
    exit_code = 0
    for backend_name in backends:
        try:
            adapter = Adapter(backend_name)
            backend_report = {
                "backend": backend_name,
                "device": _device_properties(adapter),
                "rows": [],
            }
            for case in cases:
                inputs = _inputs(case)
                expected, expected_metadata = _expected(case, inputs)
                resident = tuple(adapter.upload(value) for value in inputs)
                adapter.sync()
                checked, metadata = adapter.compute(case, resident, inputs)
                adapter.sync()
                validation = _validate(case, adapter.host(checked), expected, inputs)
                checked_iterations = (
                    metadata.get("iterations") if isinstance(metadata, dict) else None
                )
                del checked

                resident = tuple(adapter.upload(value) for value in inputs)
                adapter.sync()

                def timed_call(
                    current_resident=resident,
                    current_adapter=adapter,
                    current_case=case,
                    current_inputs=inputs,
                ):
                    return current_adapter.compute(current_case, current_resident, current_inputs)[
                        0
                    ]

                wall = _measure_resident(adapter, timed_call, repeats)
                prepared_wall = None
                prepared_public_wall = None
                if case.name == "compiled_energy" and adapter.is_pacho:
                    # Diagnostic lower-overhead control only: fixed known-valid
                    # bindings bypass the public signature/cache lookup. This is
                    # not a supported public API or a native executor.
                    runner, _ = adapter._runner_for(case, inputs)
                    plan, bindings = runner.fn._plan(resident, {})
                    prepared_wall = _measure_resident(
                        adapter,
                        lambda plan=plan, bindings=bindings: plan.execute(bindings),
                        repeats,
                    )
                    if hasattr(runner.fn, "prepare"):
                        prepared = runner.fn.prepare(*resident)
                        prepared_public_wall = _measure_resident(
                            adapter,
                            lambda prepared=prepared, resident=resident: prepared(*resident),
                            repeats,
                        )
                transfers = _measure_transfers(adapter, inputs, repeats)
                del resident
                adapter.sync()

                # Warmup again before cProfile; the call counter/profile data
                # then describe cached execution rather than plan construction.
                profile_resident = tuple(adapter.upload(value) for value in inputs)
                adapter.sync()
                profile_warm = adapter.compute(case, profile_resident, inputs)[0]
                adapter.sync()
                del profile_warm

                def profile_call(
                    current_resident=profile_resident,
                    current_adapter=adapter,
                    current_case=case,
                    current_inputs=inputs,
                ):
                    return current_adapter.compute(current_case, current_resident, current_inputs)[
                        0
                    ]

                profile = _profile_invocation(adapter, profile_call)
                del profile_resident
                adapter.sync()
                gc.collect()
                backend_report["rows"].append(
                    {
                        "case": case.name,
                        "size": case.size,
                        "dtype": str(case.dtype),
                        "implementation_origin": str(Path(pacho.__file__).resolve()),
                        "resident_wall": wall,
                        "prepared_plan_wall": prepared_wall,
                        "prepared_public_wall": prepared_public_wall,
                        "synchronized_resident_wall_samples_seconds": wall["samples_seconds"],
                        "transfer_wall": transfers,
                        "validation": validation,
                        "reference": expected_metadata,
                        "result": metadata,
                        "iterations": checked_iterations,
                        "profile": profile,
                    }
                )
            report["backends"].append(backend_report)
        except Exception as exc:  # noqa: BLE001 - preserve explicit optional-backend failure
            exit_code = 1
            report["backend_errors"].append(
                {"backend": backend_name, "type": type(exc).__name__, "message": str(exc)}
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.size < 1:
        parser.error("--size must be positive")
    _report, exit_code = run(args.output, args.backends, args.repeats, args.size)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
