"""Bounded CUDA FP64 experiment; never changes runtime dispatch or math policy.

Run after an external exclusive-device readiness check::

    python -m benchmarks.cuda_fp64_gemm --profile all --output NEW.json

The default 24 repetitions balance every permutation of off-a/on/off-b. Controls
replay the current direct kernel captured without launching it. Save the JSON
and use --control-report NEW.json to retain that exact control after promotion.
Quick selects small/tail/broadcast correctness and small application cases;
all additionally selects ten network-sized GEMMs and four full applications.
No GPU work occurs on import. This module never substitutes a CPU backend.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import random
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks import dense_network as dense
from benchmarks.cuda_fp64_tile import Kernel, source_for

MODES = ("off-a", "on", "off-b")
SEED = 1049
NETWORK_SHAPES = (
    (4096, 128, 256),
    (4096, 256, 32),
    (256, 4096, 32),
    (4096, 32, 256),
    (128, 4096, 256),
    (4095, 127, 255),
    (4095, 255, 31),
    (255, 4095, 31),
    (4095, 31, 255),
    (127, 4095, 255),
)


@dataclass(frozen=True)
class GemmCase:
    name: str
    ashape: tuple
    bshape: tuple
    data: str = "finite"


QUICK_CASES = (
    GemmCase("tiny", (1, 1), (1, 1)),
    GemmCase("tile", (16, 16), (16, 16)),
    GemmCase("tails", (17, 19), (19, 33)),
    GemmCase("rectangle-small", (3, 17), (17, 5)),
    GemmCase("rectangle-tails", (17, 31), (31, 33)),
    GemmCase("rectangle-tile-k", (33, 16), (16, 17)),
    GemmCase("tiny-network-layer1", (31, 7), (7, 11)),
    GemmCase("tiny-network-layer2", (31, 11), (11, 3)),
    GemmCase("tiny-network-dw2", (11, 31), (31, 3)),
    GemmCase("tiny-network-dhidden", (31, 3), (3, 11)),
    GemmCase("tiny-network-dw1", (7, 31), (31, 11)),
    GemmCase("batched", (3, 17, 19), (3, 19, 33)),
    GemmCase("broadcast", (2, 1, 17, 19), (1, 3, 19, 33)),
    GemmCase("adversarial", (17, 19), (19, 33), "adversarial"),
)
ALL_CASES = QUICK_CASES + tuple(
    GemmCase(f"network-{m}x{k}x{n}", (m, k), (k, n)) for m, k, n in NETWORK_SHAPES
)


def mode_orders(repeats, seed):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 6 or repeats % 6:
        raise ValueError("repeats must be a positive multiple of six, at least six")
    orders = list(itertools.permutations(MODES)) * (repeats // 6)
    random.Random(seed).shuffle(orders)
    return orders


@contextmanager
def override(obj, name, value):
    """Restore the exact instance dictionary, including inherited attributes."""
    missing = object()
    previous = vars(obj).get(name, missing)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if previous is missing:
            delattr(obj, name)
        else:
            setattr(obj, name, previous)


def capture_current(backend, original, a, b, shape, dtype):
    """Capture exactly one public backend matmul launch, without executing it.

    The original bound matmul still allocates its output. Recorded argument
    identities are checked, then discarded; caller preallocates separate outputs.
    """
    launches = []

    def record(source, blocks, *args, threads=128):
        launches.append((source, blocks, args, threads))

    with override(backend, "_run", record):
        out = original(a, b, shape, dtype)
    if len(launches) != 1:
        raise AssertionError(f"Expected one direct matmul launch, got {len(launches)}")
    source, blocks, arguments, threads = launches[0]
    if len(arguments) != 3 or not all(x is y for x, y in zip(arguments, (a._data, b._data, out))):
        raise AssertionError("Captured matmul changed its pointer ABI")
    # Require the authoritative sequential direct kernel rather than silently
    # treating a promoted runtime candidate as a historical baseline.
    if "__shared__" in source or "acc+=" not in source or "for(" not in source:
        raise AssertionError("Current kernel is not direct; provide a pre-promotion control report")
    return Kernel(source, int(blocks), int(threads), 0, tuple(shape))


def route_key(ashape, bshape):
    return json.dumps([list(ashape), list(bshape)], separators=(",", ":"))


class Routes:
    """Frozen captured controls plus benchmark-only candidate matmul override."""

    def __init__(self, backend, limits, controls=None):
        if backend.name != "cuda" or backend.blas != "custom":
            raise ValueError("This experiment requires the explicit CUDA custom backend")
        self.backend, self.original, self.limits = backend, backend.matmul, limits
        self.controls = dict(controls or {})
        self.candidates = {}
        self.frozen = False
        self.calls = {mode: {} for mode in MODES}

    def kernels(self, a, b, shape, dtype):
        key = route_key(a.shape, b.shape)
        if key not in self.controls:
            if self.frozen:
                raise AssertionError(f"Uncaptured control shape during timing: {key}")
            self.controls[key] = capture_current(self.backend, self.original, a, b, shape, dtype)
        if key not in self.candidates:
            if self.frozen:
                raise AssertionError(f"Unprepared candidate shape during timing: {key}")
            self.candidates[key] = source_for(a.shape, b.shape)
        control, candidate = self.controls[key], self.candidates[key]
        if tuple(shape) != control.shape or tuple(shape) != candidate.shape:
            raise AssertionError("Control/candidate output shape mismatch")
        check_limits(control, self.limits)
        check_limits(candidate, self.limits)
        return control, candidate

    def matmul(self, mode):
        def calculate(a, b, shape, dtype):
            if np.dtype(dtype) != np.dtype("float64"):
                raise AssertionError("The FP64 experiment encountered another dtype")
            if min(a.ndim, b.ndim) < 2 or not prod(a.shape) or not prod(b.shape):
                # Only explicitly ineligible vectors/empty/K0 retain original semantics.
                return self.original(a, b, shape, dtype)
            control, candidate = self.kernels(a, b, shape, dtype)
            kernel = candidate if mode == "on" else control
            key = route_key(a.shape, b.shape)
            counts = self.calls[mode]
            counts[key] = counts.get(key, 0) + 1
            out = self.backend.empty(shape, dtype)
            self.backend._run(
                kernel.source, kernel.blocks, a._data, b._data, out, threads=kernel.threads
            )
            return out

        return calculate


def device_limits(backend):
    from openpacho._cuda import _check

    names = {
        "threads": "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
        "shared_bytes": "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK",
        "grid_x": "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X",
    }
    result = {
        key: int(
            _check(
                backend.driver.cuDeviceGetAttribute(
                    getattr(backend.driver.CUdevice_attribute, name), backend._device
                )
            )
        )
        for key, name in names.items()
    }
    if result["threads"] < 256 or result["shared_bytes"] < 4096:
        raise ValueError("Candidate requires 256 threads/block and 4096 static shared bytes")
    return result


def check_limits(kernel, limits):
    if not (0 < kernel.blocks <= limits["grid_x"]):
        raise ValueError("Kernel block count exceeds CUDA grid-x capability")
    if not (0 < kernel.threads <= limits["threads"]):
        raise ValueError("Kernel threads exceed CUDA block capability")
    if not (0 <= kernel.shared_bytes <= limits["shared_bytes"]):
        raise ValueError("Kernel static shared storage exceeds CUDA capability")


def compare(actual, expected):
    """Finite outputs require identical bits, including the sign of zero.

    Special values require identical NaN and signed-infinity classifications;
    NaN payload/sign are intentionally unspecified. Finite locations retain the
    exact bit contract even in an otherwise nonfinite output.
    """
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype != np.dtype("float64") or b.dtype != a.dtype:
        return {"accepted": False, "metadata_equal": False}
    finite = np.isfinite(a) & np.isfinite(b)
    classes = all(
        np.array_equal(f(a), f(b)) for f in (np.isfinite, np.isnan, np.isposinf, np.isneginf)
    )
    bits = bool(np.array_equal(a.view(np.uint64)[finite], b.view(np.uint64)[finite]))
    return {
        "accepted": bool(classes and bits),
        "metadata_equal": True,
        "finite_bitwise_equal": bits,
        "classifications_equal": classes,
        "all_finite": bool(np.all(np.isfinite(a))),
        "finite_values": int(np.count_nonzero(finite)),
    }


def snapshot(value):
    return tuple(np.array(p.asnumpy(a), copy=True) for a in dense.outputs(value))


def measure(backend, function):
    backend.synchronize()
    start = time.perf_counter_ns()
    value = function()
    backend.synchronize()
    return (time.perf_counter_ns() - start) * 1e-9, value


def inputs_for(case, seed):
    rng = np.random.default_rng(seed)
    values = tuple(
        rng.uniform(-1, 1, shape).astype("float64") for shape in (case.ashape, case.bshape)
    )
    if case.data == "adversarial":
        a, b = values
        a[0, :] = 0.0
        a[1, :] = -0.0
        a[2, 0] = np.inf
        a[3, 0] = -np.inf
        a[4, 0] = np.nan
        a[5, :4] = (1e308, 1e-308, -1e308, -1e-308)
        b[0, :4] = (0.0, -0.0, np.inf, -np.inf)
        b[1, 4] = np.nan
    return values


def digest_inputs(values):
    digest = hashlib.sha256()
    for a in values:
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def run_samples(row, backend, routes, calls, expected, validate, repeats, seed, save):
    """Each call is timed alone; every output check is outside its sample clock."""
    row["scopes"] = {}
    for scope_index, (scope, functions) in enumerate(calls.items()):
        evidence = {"orders": mode_orders(repeats, seed + scope_index), "modes": {}}
        row["scopes"][scope] = evidence
        for mode in MODES:
            state = {"samples_seconds": [], "validation": [], "warmup": {}}
            evidence["modes"][mode] = state
            # Patch activation and restoration stay outside all clocks. The
            # application's internal matmul calls remain part of its clock.
            with override(backend, "matmul", routes.matmul(mode)):
                elapsed, value = measure(backend, functions[mode])
                state["first_call_seconds"] = elapsed
                actual = value if scope == "complete" else snapshot(value)
                state["validation"].append(validate(actual, expected, f"{scope}:{mode}:first"))
                start, count = time.monotonic(), 0
                while count < 3 or time.monotonic() - start < 0.1:
                    _, value = measure(backend, functions[mode])
                    count += 1
                actual = value if scope == "complete" else snapshot(value)
                state["validation"].append(validate(actual, expected, f"{scope}:{mode}:warmup"))
                state["warmup"] = {"calls": count, "seconds": time.monotonic() - start}
            save()
        for repeat, order in enumerate(evidence["orders"]):
            for mode in order:
                state = evidence["modes"][mode]
                with override(backend, "matmul", routes.matmul(mode)):
                    elapsed, value = measure(backend, functions[mode])
                state["samples_seconds"].append(elapsed)
                actual = value if scope == "complete" else snapshot(value)
                state["validation"].append(validate(actual, expected, f"{scope}:{mode}:{repeat}"))
            save()
        for state in evidence["modes"].values():
            samples = np.array(state["samples_seconds"])
            state["median_seconds"] = float(np.median(samples))
            state["min_seconds"] = float(np.min(samples))
        medians = {m: evidence["modes"][m]["median_seconds"] for m in MODES}
        evidence["candidate_speedup_vs_bracketing_controls"] = (
            medians["off-a"] + medians["off-b"]
        ) / (2 * medians["on"])
    row["execution_complete"] = True
    row["numerically_accepted"] = all(
        check["accepted"]
        for scope in row["scopes"].values()
        for state in scope["modes"].values()
        for check in state["validation"]
    )


def gemm_row(case, backend, routes, repeats, seed, save, row):
    host = inputs_for(case, seed)
    row["input_sha256"] = digest_inputs(host)
    a, b = (p.asarray(x, backend=backend) for x in host)
    shape = source_for(a.shape, b.shape).shape
    control, candidate = routes.kernels(a, b, shape, np.dtype("float64"))
    routes.frozen = True
    # Three independent buffers make the preallocated scope allocation-free.
    out = {
        mode: p.ndarray._wrap(backend.empty(shape, np.dtype("float64")), shape, "float64", backend)
        for mode in MODES
    }
    kernels = {mode: candidate if mode == "on" else control for mode in MODES}

    def launch(mode):
        kernel = kernels[mode]
        backend._run(
            kernel.source, kernel.blocks, a._data, b._data, out[mode]._data, threads=kernel.threads
        )
        return out[mode]

    elapsed, value = measure(backend, lambda: launch("off-a"))
    expected = snapshot(value)
    row["direct_reference_first_call_seconds"] = elapsed
    row["control_launch"] = {k: v for k, v in asdict(control).items() if k != "source"}
    row["candidate_launch"] = {k: v for k, v in asdict(candidate).items() if k != "source"}
    calls = {
        "preallocated_kernel": {m: lambda m=m: launch(m) for m in MODES},
        "allocating_public_matmul": {m: lambda: a @ b for m in MODES},
    }

    def validate(actual, expected, stage):
        result = compare(actual[0], expected[0])
        result["stage"] = stage
        if case.data == "finite" and not result.get("all_finite", False):
            result["accepted"] = False
        return result

    run_samples(row, backend, routes, calls, expected, validate, repeats, seed, save)


def application_row(case, backend, routes, repeats, seed, save, row):
    host = dense.make_inputs(case)
    row["input_sha256"] = digest_inputs(host)
    with threadpool_limits(limits=1, user_api="blas"):
        expected, _, _ = dense.reference(case, host)
    host = dense.runtime_inputs(case, host)
    resident = tuple(p.asarray(x, backend=backend) for x in host)
    plans, prepare = {}, {}
    routes.frozen = False
    for mode in MODES:
        with override(backend, "matmul", routes.matmul(mode)):
            start = time.perf_counter_ns()
            plans[mode] = p.compile(dense.function(case)).prepare(*dense.arguments(case, resident))
            prepare[mode] = {
                "seconds": (time.perf_counter_ns() - start) * 1e-9,
                "explain": plans[mode].explain(),
            }
    if len({id(plan._plan) for plan in plans.values()}) != 3:
        raise AssertionError("Application modes must use independent compiled plans")

    def resident_call(mode):
        return plans[mode](*dense.arguments(case, resident))

    # Capture every route and compile both source variants before freezing.
    # This setup traversal is reported separately from first timed calls.
    row["route_discovery_seconds"] = {}
    for mode in MODES:
        with override(backend, "matmul", routes.matmul(mode)):
            elapsed, value = measure(backend, lambda mode=mode: resident_call(mode))
        row["route_discovery_seconds"][mode] = elapsed
        if mode == "off-a":
            direct = snapshot(value)
    routes.frozen = True
    row["prepare"] = prepare
    row["original_parameters_every_call"] = True

    def complete(mode):
        uploaded = tuple(p.asarray(x, backend=backend) for x in host)
        return snapshot(plans[mode](*dense.arguments(case, uploaded)))

    calls = {
        "resident_prepared_application": {m: lambda m=m: resident_call(m) for m in MODES},
        "complete": {m: lambda m=m: complete(m) for m in MODES},
    }

    def validate(actual, unused, stage):
        diagnostic = {"diagnostics": {}, "validation_failures": []}
        dense.diagnose(diagnostic, dense.output_names(case), actual, expected, case, stage)
        exact = [compare(a, b) for a, b in zip(actual, direct)]
        accepted = (
            len(actual) == len(direct)
            and not diagnostic["validation_failures"]
            and all(item["accepted"] for item in exact)
        )
        return {
            "stage": stage,
            "accepted": accepted,
            "reference": diagnostic,
            "direct_kernel_comparison": exact,
        }

    run_samples(row, backend, routes, calls, expected, validate, repeats, seed, save)


def load_controls(path):
    if path is None:
        return {}
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema") != "openpacho.cuda-fp64-gemm.v1":
        raise ValueError("Unsupported control report schema")
    controls = {}
    for key, entry in report["captured_controls"].items():
        payload = dict(entry)
        checksum = payload.pop("sha256")
        if hashlib.sha256(payload["source"].encode()).hexdigest() != checksum:
            raise ValueError("Captured control source checksum mismatch")
        payload["shape"] = tuple(payload["shape"])
        kernel = Kernel(**payload)
        if "__shared__" in kernel.source or "acc+=" not in kernel.source:
            raise ValueError("Control report does not contain a direct kernel")
        controls[key] = kernel
    return controls


def provenance():
    package = Path(p.__file__).resolve().parent
    files = {
        "runner": Path(__file__),
        "candidate": Path(__file__).with_name("cuda_fp64_tile.py"),
        "application": Path(dense.network.__file__),
        "application_helper": Path(dense.__file__),
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_origin": str(Path(p.__file__).resolve()),
        "sources": {
            key: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in files.items()
        },
        "runtime_sources": {
            str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))
        },
    }


def run(output, *, profile="all", repeats=24, seed=SEED, control_report=None):
    mode_orders(repeats, seed)
    if profile not in ("quick", "all"):
        raise ValueError("profile must be quick or all")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cases = QUICK_CASES if profile == "quick" else ALL_CASES
    shapes = (
        ((31, 7, 11, 3),)
        if profile == "quick"
        else ((31, 7, 11, 3), (4096, 128, 256, 32), (4095, 127, 255, 31))
    )
    apps = tuple(dense.Case(w, *shape, "float64") for shape in shapes for w in ("inference", "sgd"))
    report = {
        "schema": "openpacho.cuda-fp64-gemm.v1",
        "backend": "cuda-custom",
        "profile": profile,
        "repeats": repeats,
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance(),
        "control_report": str(control_report) if control_report else None,
        "requirements": "External exclusive-device readiness gate before worker launch; "
        "no concurrent GPU work; explicit CUDA custom backend, no fallback",
        "compiler_policy": "Unchanged backend NVRTC: --fmad=false --ftz=false "
        "--prec-div=true --prec-sqrt=true; increasing K, float64",
        "timing_policy": "Synchronized wall seconds, 6 equally frequent mode permutations; "
        "separate first calls and >=3 calls/>=0.1s warmup; output checks "
        "outside clocks; complete includes all input uploads and output downloads; "
        "setup/discovery/prepare excluded. Patches activate outside clocks. "
        "GEMM direct reference first call and application route discovery calls "
        "precede per-scope first calls and may compile device kernels. "
        "Application/public-matmul samples use a shared Python route wrapper "
        "for every mode; these forced-route measurements do not directly "
        "measure the overhead of unchanged production dispatch.",
        "validation_policy": "Every sample: finite bits equal captured direct control; "
        "NaN/+Inf/-Inf classifications equal. Application additionally "
        "checks fixed dense_network float64 tolerance against NumPy. "
        "Numerical failures remain timed and invalidate report.",
        "captured_controls": {},
        "captured_candidates": {},
        "rows": [],
        "attempts_complete": False,
        "execution_complete": False,
        "all_numerically_accepted": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    backend, routes = None, None
    # Reserve exclusively before CUDA initialization; keep partial evidence.
    with path.open("x", encoding="utf-8") as stream:

        def save():
            if routes is not None:
                for name, kernels in (
                    ("captured_controls", routes.controls),
                    ("captured_candidates", routes.candidates),
                ):
                    report[name] = {
                        key: dict(
                            asdict(kernel),
                            sha256=hashlib.sha256(kernel.source.encode()).hexdigest(),
                        )
                        for key, kernel in kernels.items()
                    }
                report["route_calls"] = routes.calls
            stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.truncate()
            stream.flush()

        save()
        try:
            controls = load_controls(control_report)
            backend = p.get_backend("cuda", blas="custom")
            report["device"] = backend.device
            report["device_limits"] = device_limits(backend)
            report["architecture"] = backend._architecture
            routes = Routes(backend, report["device_limits"], controls)
            jobs = [("gemm", case) for case in cases] + [("application", case) for case in apps]
            # Fixed job order eases replication; only balanced sample order is randomized.
            for index, (kind, case) in enumerate(jobs):
                row = {
                    "kind": kind,
                    "case": asdict(case),
                    "execution_complete": False,
                    "numerically_accepted": False,
                }
                report["rows"].append(row)
                save()
                try:
                    routes.frozen = False
                    runner = gemm_row if kind == "gemm" else application_row
                    runner(case, backend, routes, repeats, seed + index * 101, save, row)
                except Exception as exc:  # noqa: BLE001 - retain partial workload evidence.
                    row["error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                save()
            report["attempts_complete"] = True
            report["execution_complete"] = all(r["execution_complete"] for r in report["rows"])
            report["all_numerically_accepted"] = all(
                r["numerically_accepted"] for r in report["rows"]
            )
        except Exception as exc:  # noqa: BLE001 - keep backend initialization failures explicit.
            report["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            if backend is not None:
                try:
                    backend.synchronize()
                    backend.close()
                except Exception as exc:  # noqa: BLE001 - preserve original errors during cleanup.
                    report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
            save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("quick", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--control-report")
    args = parser.parse_args(argv)
    report = run(**vars(args))
    return (
        0
        if (
            report["attempts_complete"]
            and report["execution_complete"]
            and report["all_numerically_accepted"]
            and "cleanup_error" not in report
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
