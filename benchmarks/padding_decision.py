"""Paired public padding decisions with two unpadded controls and numerical errors.

Use an isolated worker per backend with the project's GPU readiness policy.
Successful completion means the experiment finished, not that numerical validation
passed. Finite reference-tolerance failures remain visible and do not stop timing.
"""

import argparse
import hashlib
import itertools
import json
import operator
import random
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import engine, measure, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.stability import nvidia_snapshot

CASES = (
    ((257, 255), (255, 259)),
    ((1024, 1024), (1024, 1024)),
    ((1025, 1023), (1023, 1027)),
    ((2049, 2047), (2047, 2051)),
    ((4095, 4095), (4095, 4095)),
    ((4097, 4093), (4093, 4101)),
    ((2048, 2047), (2047, 2048)),
    ((2047, 2048), (2048, 2048)),
    ((2048, 2048), (2048, 2049)),
    ((1024, 8191), (8191, 1024)),
    ((65, 8191), (8191, 2051)),
    ((4097, 255), (255, 4099)),
    ((8191, 513), (513, 257)),
    ((127, 2047), (2047, 8191)),
    ((4, 257, 255), (4, 255, 259)),
    ((2, 1, 257, 511), (1, 3, 511, 259)),
)
MODES = (("off-a", False), ("on", True), ("off-b", False))
SCOPES = ("prepared", "complete")


def product(a, b):
    return a @ b


def prepare_calls(backend, host):
    """Capture each plan independently while sharing resident input arrays."""
    resident = tuple(p.asarray(value, backend=backend) for value in host)
    calls, explanations = {}, {}
    for mode, flag in MODES:
        function = p.compile(product, padding=flag)
        prepared = function.prepare(*resident)
        explanations[mode] = function.explain(*resident)
        calls[mode] = {
            "prepared": lambda prepared=prepared, resident=resident: prepared(*resident),
            "complete": lambda prepared=prepared, host=host: p.asnumpy(
                prepared(*(p.asarray(value, backend=backend) for value in host))
            ),
        }
    return calls, explanations


def validate_result(actual, reference):
    """Keep the fixed elementwise tolerance distinct from aggregate diagnostics."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.dtype != np.dtype("float32"):
        raise ValueError("Padding benchmark requires the reference shape and float32 output")
    finite = np.isfinite(actual) & np.isfinite(reference)
    with np.errstate(invalid="ignore", over="ignore"):
        difference = np.abs(actual.astype(np.float64) - reference)
        violations = ~finite | (difference > (5e-5 + 5e-5 * np.abs(reference)))
    locations = np.argwhere(violations)[:3]
    return {
        "rtol": 5e-5,
        "atol": 5e-5,
        "finite": bool(np.all(finite)),
        "reference_close": not bool(np.any(violations)),
        "violations": int(np.count_nonzero(violations)),
        "examples": [
            {
                "index": index.tolist(),
                "actual": float(actual[tuple(index)])
                if np.isfinite(actual[tuple(index)])
                else str(actual[tuple(index)]),
                "reference": float(reference[tuple(index)])
                if np.isfinite(reference[tuple(index)])
                else str(reference[tuple(index)]),
            }
            for index in locations
        ],
    }


def _indices(case_indices):
    if case_indices is None:
        return list(range(len(CASES)))
    try:
        requested = tuple(case_indices)
        if any(isinstance(index, (bool, np.bool_)) for index in requested):
            raise ValueError("Case indices must be integers")
        indices = [operator.index(index) for index in requested]
    except TypeError as exc:
        raise ValueError("Case indices must be integers") from exc
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Select at least one distinct case index")
    if any(index < 0 or index >= len(CASES) for index in indices):
        raise ValueError(f"Case indices must be between 0 and {len(CASES) - 1}")
    return indices


def _provenance():
    package = Path(p.__file__).resolve().parent
    return {
        "package_origin": str(Path(p.__file__).resolve()),
        "runtime_sources": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(package.glob("*.py"))
        },
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_sha256": hashlib.sha256(
            Path(__file__).with_name("backend_sharing.py").read_bytes()
        ).hexdigest(),
    }


def run(output, backends, repeats=30, seed=971, case_indices=None):
    """Measure finite-reference failures too; inspect all_reference_close separately."""
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output file")
    if len(backends) != 1 or backends[0] not in ("opencl", "vulkan", "cuda-custom"):
        raise ValueError("Select one supported custom GPU backend")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 6 or repeats % 6:
        raise ValueError("Repeats must be an integer multiple of six, at least six")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    indices = _indices(case_indices)
    random.Random(seed).shuffle(indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "openpacho.padding-decision.v1",
        "backend": backends[0],
        "seed": seed,
        "repeats": repeats,
        "case_order": indices,
        **_provenance(),
        "scope": (
            "public prepared/complete calls; shared resident inputs; all padding copies included; "
            "complete means upload + prepared execution + download, not eager matmul; "
            "two independent unpadded controls; all six mode orders equally frequent; "
            "case order shuffled by seed; warmup and validation outside sample clocks"
        ),
        "validation": (
            "rtol=5e-5, atol=5e-5 unchanged; finite reference violations are recorded without "
            "stopping other modes; reject nonfinite outputs, wrong shape/dtype or changed "
            "replay bits; complete records execution, all_reference_close records accuracy"
        ),
        "rows": [],
        "complete": False,
        "all_reference_close": None,
        "all_bitwise_control_equal": None,
    }
    backend = None

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        _, backend = engine(backends[0])
        report["device"] = backend.device
        with threadpool_limits(limits=1):
            for index in indices:
                shapes = CASES[index]
                rng = np.random.default_rng(seed * 101 + index)
                host = tuple(rng.uniform(-0.5, 0.5, shape).astype("float32") for shape in shapes)
                reference = host[0].astype("float64") @ host[1].astype("float64")
                functions, explanations = prepare_calls(backend, host)
                rows, values = {}, {}
                for mode in functions:
                    result = p.asnumpy(functions[mode]["prepared"]())
                    validation = validate_result(result, reference)
                    row = {
                        "case_index": index,
                        "shapes": shapes,
                        "mode": mode,
                        "error": error_metrics(result, reference),
                        "validation": validation,
                        "explain": explanations[mode],
                    }
                    rows[mode] = row
                    report["rows"].append(row)
                    report["all_reference_close"] = (
                        report["all_reference_close"] is not False and validation["reference_close"]
                    )
                    if not validation["reference_close"]:
                        print(
                            f"Numerical validation failed: case {index}, {mode}, "
                            f"{validation['violations']} reference-tolerance violations "
                            "(rtol=5e-5, atol=5e-5).",
                            flush=True,
                        )
                    if not validation["finite"]:
                        raise ValueError("Padding benchmark produced nonfinite output/reference")
                    values[mode] = result.view(np.uint32)
                for mode in functions:
                    equal = bool(np.array_equal(values[mode], values["off-a"]))
                    rows[mode]["bitwise_control_equal"] = equal
                    report["all_bitwise_control_equal"] = (
                        report["all_bitwise_control_equal"] is not False and equal
                    )
                rows["on"]["before"] = nvidia_snapshot()
                for scope in SCOPES:
                    samples = {mode: [] for mode in functions}
                    for mode in functions:
                        rows[mode][scope + "_warmup"] = warmup(functions[mode][scope], p, backend)
                        _, _, value = measure(functions[mode][scope], p, backend)
                        actual = p.asnumpy(value) if scope == "prepared" else value
                        replay_equal = actual.dtype == np.dtype("float32") and np.array_equal(
                            actual.view(np.uint32), values[mode]
                        )
                        rows[mode][scope + "_replay_bitwise_equal"] = bool(replay_equal)
                        if not replay_equal:
                            raise AssertionError(f"Replay changed output bits: {mode}/{scope}")
                        del value, actual
                    orders = list(itertools.permutations(functions)) * (repeats // 6)
                    random.Random(seed + index + (100 if scope == "complete" else 0)).shuffle(
                        orders
                    )
                    for order in orders:
                        for mode in order:
                            seconds, _, value = measure(functions[mode][scope], p, backend)
                            samples[mode].append(seconds)
                            del value
                    for mode in functions:
                        rows[mode][scope] = _sample_stats(samples[mode])
                rows["on"]["after"] = nvidia_snapshot()
                save()
                print(
                    index,
                    shapes,
                    {
                        scope: {
                            mode: round(rows[mode][scope]["median_seconds"] * 1e3, 4)
                            for mode in rows
                        }
                        for scope in SCOPES
                    },
                    "violations",
                    {mode: rows[mode]["validation"]["violations"] for mode in rows},
                    flush=True,
                )
                del functions, explanations, rows, values, host, reference, result
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
    parser.add_argument("--backends", default="opencl")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=971)
    parser.add_argument("--case-indices", help="Comma-separated distinct indices in the case table")
    args = parser.parse_args(argv)
    try:
        indices = (
            None
            if args.case_indices is None
            else [int(value) for value in args.case_indices.split(",")]
        )
        report = run(
            args.output,
            args.backends.split(","),
            repeats=args.repeats,
            seed=args.seed,
            case_indices=indices,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
