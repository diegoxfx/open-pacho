"""Paired compiled CLBlast transpose flags versus materialized transposes.

Run one isolated worker, using the common readiness policy between processes.
Completion and numerical acceptance are separate report fields.
"""

import argparse
import hashlib
import itertools
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import engine, measure, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.stability import nvidia_snapshot

CASES = tuple(
    {"name": f"gemm-{m}-{k}-{n}-{side}", "dimensions": (m, k, n), "side": side}
    for m, k, n in ((35, 33, 39), (257, 513, 129), (1025, 2047, 513), (65, 8191, 2051))
    for side in ("left", "right", "both")
) + (
    {"name": "centered-covariance-4097-129", "covariance": (4097, 129)},
    {"name": "centered-covariance-8191-513", "covariance": (8191, 513)},
)
MODES = ("off-a", "on", "off-b")
SCOPES = ("prepared", "complete")


@contextmanager
def layout_enabled(backend, enabled):
    """Disable only while capturing a control, and restore the exact override."""
    if enabled:
        yield
        return
    sentinel = object()
    original = vars(backend).get("prepare_layout_matmul", sentinel)
    backend.prepare_layout_matmul = None
    try:
        yield
    finally:
        if original is sentinel:
            del backend.prepare_layout_matmul
        else:
            backend.prepare_layout_matmul = original


def inputs(case, seed):
    rng = np.random.default_rng(seed)
    if "covariance" in case:
        return (rng.uniform(-0.5, 0.5, case["covariance"]).astype("float32"),)
    m, k, n = case["dimensions"]
    ta, tb = case["side"] in ("left", "both"), case["side"] in ("right", "both")
    return tuple(
        rng.uniform(-0.5, 0.5, shape).astype("float32")
        for shape in ((k, m) if ta else (m, k), (n, k) if tb else (k, n))
    )


def operation(case):
    if "covariance" in case:

        def covariance(a):
            centered = a - np.mean(a, axis=0, keepdims=True)
            return (centered.T @ centered) / (a.shape[0] - 1)

        return covariance
    ta, tb = case["side"] in ("left", "both"), case["side"] in ("right", "both")
    return lambda a, b: (a.T if ta else a) @ (b.T if tb else b)


def prepare_calls(backend, host, function):
    resident = tuple(p.asarray(value, backend=backend) for value in host)
    calls, explanations = {}, {}
    for mode in MODES:
        with layout_enabled(backend, mode == "on"):
            compiled = p.compile(function)
            prepared = compiled.prepare(*resident)
        explanations[mode] = prepared.explain()
        calls[mode] = {
            "prepared": lambda prepared=prepared: prepared(*resident),
            "complete": lambda prepared=prepared: p.asnumpy(
                prepared(*(p.asarray(value, backend=backend) for value in host))
            ),
        }
    return calls, explanations


def validate(actual, reference):
    if actual.shape != reference.shape or actual.dtype != np.dtype("float32"):
        raise ValueError("Reference shape and float32 output required")
    finite = np.isfinite(actual) & np.isfinite(reference)
    with np.errstate(over="ignore", invalid="ignore"):
        violations = ~finite | (
            np.abs(actual.astype("float64") - reference) > 5e-5 + 5e-5 * np.abs(reference)
        )
    return {
        "rtol": 5e-5,
        "atol": 5e-5,
        "finite": bool(finite.all()),
        "reference_close": not bool(violations.any()),
        "violations": int(violations.sum()),
        "examples": [
            {
                "index": index.tolist(),
                "actual": str(actual[tuple(index)]),
                "reference": str(reference[tuple(index)]),
            }
            for index in np.argwhere(violations)[:3]
        ],
    }


def run(output, backends, repeats=30, seed=983):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output file")
    if list(backends) != ["opencl-clblast"]:
        raise ValueError("Select exactly opencl-clblast")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 6 or repeats % 6:
        raise ValueError("Repeats must be an integer multiple of six, at least six")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    output.parent.mkdir(parents=True, exist_ok=True)
    package = Path(p.__file__).resolve().parent
    indices = list(range(len(CASES)))
    random.Random(seed).shuffle(indices)
    report = {
        "schema": "openpacho.compiled-layouts.v1",
        "backend": backends[0],
        "seed": seed,
        "repeats": repeats,
        "case_order": indices,
        "package_origin": str(Path(p.__file__).resolve()),
        "runtime_sources": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(package.glob("*.py"))
        },
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_sha256": hashlib.sha256(
            Path(__file__).with_name("backend_sharing.py").read_bytes()
        ).hexdigest(),
        "scope": "Three independent plans on identical resident arrays: off-a/on/off-b; controls disable only layout preparation; prepared includes allocations/compute/copies/synchronization; complete adds uploads and download; six mode permutations equally frequent; warmup and validation outside clocks.",
        "validation": "Fixed rtol=5e-5 and atol=5e-5 against float64 evaluation of identical float32 inputs; finite violations are visible and timed; nonfinite or changed replay bits abort; completion is not numerical acceptance.",
        "rows": [],
        "complete": False,
        "all_reference_close": True,
    }
    backend = None

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        _, backend = engine(backends[0])
        report["device"] = backend.device
        with threadpool_limits(limits=1):
            for index in indices:
                case = CASES[index]
                host = inputs(case, seed * 101 + index)
                function = operation(case)
                reference = function(*(value.astype("float64") for value in host))
                calls, explanations = prepare_calls(backend, host, function)
                rows, bits = {}, {}
                for mode in MODES:
                    actual = p.asnumpy(calls[mode]["prepared"]())
                    validation = validate(actual, reference)
                    row = {
                        "case_index": index,
                        "case": case,
                        "shapes": [a.shape for a in host],
                        "mode": mode,
                        "explain": explanations[mode],
                        "validation": validation,
                        "error": error_metrics(actual, reference),
                    }
                    rows[mode] = row
                    report["rows"].append(row)
                    report["all_reference_close"] &= validation["reference_close"]
                    if not validation["reference_close"]:
                        print(
                            f"Numerical validation failed: {case['name']} {mode}: {validation['violations']} violations",
                            flush=True,
                        )
                    if not validation["finite"]:
                        raise ValueError("Nonfinite output/reference")
                    bits[mode] = actual.view("uint32")
                for mode in MODES:
                    rows[mode]["bitwise_control_equal"] = bool(
                        np.array_equal(bits[mode], bits["off-a"])
                    )
                if not rows["off-b"]["bitwise_control_equal"]:
                    raise AssertionError("Two materialized controls differ")
                rows["on"]["before"] = nvidia_snapshot()
                for scope in SCOPES:
                    samples = {mode: [] for mode in MODES}
                    for mode in MODES:
                        rows[mode][scope + "_warmup"] = warmup(calls[mode][scope], p, backend)
                        _, _, value = measure(calls[mode][scope], p, backend)
                        actual = p.asnumpy(value) if scope == "prepared" else value
                        equal = actual.dtype == np.dtype("float32") and np.array_equal(
                            actual.view("uint32"), bits[mode]
                        )
                        rows[mode][scope + "_replay_bitwise_equal"] = bool(equal)
                        if not equal:
                            raise AssertionError(f"Replay changed output bits: {mode}/{scope}")
                        del value, actual
                    orders = list(itertools.permutations(MODES)) * (repeats // 6)
                    random.Random(seed + index + (100 if scope == "complete" else 0)).shuffle(
                        orders
                    )
                    for order in orders:
                        for mode in order:
                            seconds, _, value = measure(calls[mode][scope], p, backend)
                            samples[mode].append(seconds)
                            del value
                    for mode in MODES:
                        rows[mode][scope] = _sample_stats(samples[mode])
                rows["on"]["after"] = nvidia_snapshot()
                save()
                print(
                    case["name"],
                    {
                        scope: {
                            mode: round(rows[mode][scope]["median_seconds"] * 1e3, 4)
                            for mode in MODES
                        }
                        for scope in SCOPES
                    },
                    flush=True,
                )
                del calls, explanations, rows, bits, host, reference
        report["complete"] = True
    except Exception as exc:
        report["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            if backend is not None:
                close = getattr(backend, "close", None)
                (close if close is not None else backend.synchronize)()
        except Exception as exc:
            report["complete"] = False
            report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backends", default="opencl-clblast")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=983)
    args = parser.parse_args(argv)
    try:
        report = run(args.output, args.backends.split(","), args.repeats, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
