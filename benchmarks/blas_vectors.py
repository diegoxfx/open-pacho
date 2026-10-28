"""Paired cuBLAS vector primitives versus GEMM emulation on the same runtime.

Run one isolated worker with readiness gates between independent processes.
Completion and numerical validity are separate report fields; a successful CLI
exit means the experiment finished, not that every result met its tolerance.
"""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
from types import MethodType

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import engine, measure, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.stability import nvidia_snapshot
from benchmarks.suite import source_hashes
from benchmarks.three_actions import old_methods
from openpacho import _cuda

CASES = (
    ((257,), (257,)),
    ((8191,), (8191,)),
    ((1048573,), (1048573,)),
    ((65, 8191), (8191,)),
    ((8191,), (8191, 65)),
    ((1025, 1023), (1023,)),
    ((1023,), (1023, 1025)),
    ((4097, 511), (511,)),
    ((511,), (511, 4097)),
    ((4, 257, 511), (511,)),
    ((511,), (4, 511, 257)),
    ((65, 63), (63, 97)),  # Unchanged GEMM is a negative control.
    ((1, 257, 511), (511,)),  # Explicit singleton batches keep historical indices above.
    ((511,), (1, 511, 257)),
)
DTYPES = ("float32", "float64")
MODES = ("old-a", "native", "old-b")
SCOPES = ("resident", "complete")


class Controls:
    """Replace only one backend instance's matmul, never global runtime classes."""

    def __init__(self, backend, wheel):
        self.backend, self.current = backend, backend.matmul
        previous = old_methods(wheel, _cuda, _cuda.CUDABackend, ("matmul",))["matmul"]
        self.previous = MethodType(previous, backend)

    def activate(self, mode):
        if mode not in MODES:
            raise ValueError(f"Unknown vector control: {mode}")
        self.backend.matmul = self.current if mode == "native" else self.previous

    def restore(self):
        self.backend.matmul = self.current


def product(a, b):
    return a @ b


def prepare_calls(backend, controls, hosts):
    """Share inputs; keep a distinct compilation and prepared plan for each mode."""
    resident = tuple(p.asarray(value, backend=backend) for value in hosts)
    calls = {}
    for mode in MODES:
        controls.activate(mode)
        prepared = p.compile(product).prepare(*resident)
        scopes = {
            "resident": lambda prepared=prepared: prepared(*resident),
            "complete": lambda prepared=prepared: p.asnumpy(
                prepared(*(p.asarray(value, backend=backend) for value in hosts))
            ),
        }
        for function in scopes.values():
            # measure/warmup run this hook before synchronization and clocks.
            function._activate_variant = lambda mode=mode: controls.activate(mode)
        calls[mode] = scopes
    return calls


def validation(actual, reference, dtype):
    actual = np.asarray(actual)
    if actual.shape != reference.shape or actual.dtype != np.dtype(dtype):
        raise ValueError("Vector benchmark result has the wrong shape or dtype")
    # Same tolerances as the existing three-actions vector benchmark.
    rtol, atol = (5e-5, 5e-6) if dtype == "float32" else (1e-11, 1e-12)
    finite = np.isfinite(actual) & np.isfinite(reference)
    with np.errstate(invalid="ignore", over="ignore"):
        difference = np.abs(actual.astype(np.float64) - reference)
        violations = ~finite | (difference > (atol + rtol * np.abs(reference)))
    return {
        "rtol": rtol,
        "atol": atol,
        "finite": bool(np.all(finite)),
        "reference_close": not bool(np.any(violations)),
        "violations": int(np.count_nonzero(violations)),
    }


def _bits(value):
    return np.asarray(value).reshape(-1).view(np.uint8)


def run(output, backends, repeats=30, seed=2041, control_wheel=None, control_component="methods"):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output file")
    if list(backends) != ["cuda-cublas"]:
        raise ValueError("Select only cuda-cublas")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 6 or repeats % 6:
        raise ValueError("Repeats must be an integer multiple of six, at least six")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if control_wheel is None or control_component != "methods":
        raise ValueError("An explicit historical wheel and methods control are required")
    control_wheel = Path(control_wheel)
    wheel_hash = hashlib.sha256(control_wheel.read_bytes()).hexdigest()
    indices = list(range(len(CASES)))
    random.Random(seed).shuffle(indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "openpacho.blas-vectors.v1",
        "backend": backends[0],
        "seed": seed,
        "repeats": repeats,
        "case_order": indices,
        "source_sha256": source_hashes(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "control_wheel": str(control_wheel.resolve()),
        "control_wheel_sha256": wheel_hash,
        "control_component": "CUDABackend.matmul only; all other components current",
        "scope": (
            "same backend/context/inputs; two independently prepared old controls; "
            "selection outside clocks; all six mode orders equally frequent; "
            "resident includes public prepared execution/allocation/synchronization; "
            "complete additionally uploads both inputs and downloads the output; "
            "compilation/warmup/validation/telemetry excluded from sample clocks"
        ),
        "reference": "float64 from quantized inputs; equal precision for float64 cases",
        "rows": [],
        "complete": False,
        "all_reference_close": None,
        "old_controls_bitwise_equal": None,
    }
    backend = controls = None

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        _, backend = engine(backends[0])
        controls = Controls(backend, control_wheel)
        report["device"] = backend.device
        with threadpool_limits(limits=1):
            for index in indices:
                for dtype in DTYPES:
                    rng = np.random.default_rng(seed * 101 + index)
                    hosts = tuple(
                        rng.uniform(-0.5, 0.5, shape).astype(dtype) for shape in CASES[index]
                    )
                    reference = hosts[0].astype(np.float64) @ hosts[1].astype(np.float64)
                    calls = prepare_calls(backend, controls, hosts)
                    rows, values = {}, {}
                    for mode in MODES:
                        _, _, result = measure(calls[mode]["resident"], p, backend)
                        actual = p.asnumpy(result)
                        del result
                        checked = validation(actual, reference, dtype)
                        row = {
                            "case_index": index,
                            "shapes": CASES[index],
                            "dtype": dtype,
                            "mode": mode,
                            "validation": checked,
                            "error": error_metrics(actual, reference),
                        }
                        rows[mode] = row
                        report["rows"].append(row)
                        report["all_reference_close"] = (
                            report["all_reference_close"] is not False
                            and checked["reference_close"]
                        )
                        if not checked["reference_close"]:
                            print(
                                f"Numerical validation failed: {index}/{dtype}/{mode}: {checked}",
                                flush=True,
                            )
                        if not checked["finite"]:
                            raise ValueError("Vector benchmark produced nonfinite output/reference")
                        values[mode] = _bits(actual)
                    equal = bool(np.array_equal(values["old-a"], values["old-b"]))
                    report["old_controls_bitwise_equal"] = (
                        report["old_controls_bitwise_equal"] is not False and equal
                    )
                    if not equal:
                        raise AssertionError("Independent old controls changed output bits")
                    for mode in MODES:
                        rows[mode]["bitwise_old_equality"] = bool(
                            np.array_equal(values[mode], values["old-a"])
                        )
                    rows["native"]["before"] = nvidia_snapshot()
                    for scope in SCOPES:
                        samples = {mode: [] for mode in MODES}
                        for mode in MODES:
                            rows[mode][scope + "_warmup"] = warmup(calls[mode][scope], p, backend)
                            _, _, result = measure(calls[mode][scope], p, backend)
                            actual = p.asnumpy(result) if scope == "resident" else result
                            replay = bool(np.array_equal(_bits(actual), values[mode]))
                            rows[mode][scope + "_replay_bitwise_equal"] = replay
                            if not replay:
                                raise AssertionError(f"Replay changed output bits: {mode}/{scope}")
                            del actual, result
                        orders = list(itertools.permutations(MODES)) * (repeats // 6)
                        random.Random(seed + index + (100 if scope == "complete" else 0)).shuffle(
                            orders
                        )
                        for order in orders:
                            for mode in order:
                                seconds, _, result = measure(calls[mode][scope], p, backend)
                                samples[mode].append(seconds)
                                del result
                        for mode in MODES:
                            rows[mode][scope] = _sample_stats(samples[mode])
                    rows["native"]["after"] = nvidia_snapshot()
                    save()
                    print(
                        index,
                        dtype,
                        {
                            mode: round(rows[mode]["resident"]["median_seconds"] * 1e6, 2)
                            for mode in MODES
                        },
                        flush=True,
                    )
                    del calls, rows, values, hosts, reference
        report["complete"] = True
    except Exception as exc:
        report["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            if controls is not None:
                controls.restore()
        finally:
            try:
                if backend is not None:
                    close = getattr(backend, "close", backend.synchronize)
                    close()
            except Exception as exc:
                report["complete"] = False
                report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="cuda-cublas")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--control-wheel", type=Path, required=True)
    parser.add_argument("--control-component", choices=("methods",), default="methods")
    args = parser.parse_args(argv)
    try:
        report = run(
            args.output,
            args.backends.split(","),
            args.repeats,
            args.seed,
            args.control_wheel,
            args.control_component,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
