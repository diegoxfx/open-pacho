"""Benchmark private bare-extrema map-reduce plans against two eager plans.

Run one backend per isolated worker after the common GPU readiness check.
This experiment changes only a prepared plan, never runtime compiler policy.
Completion, reference acceptance and bitwise control equality are separate.
"""

import argparse
import hashlib
import itertools
import json
import platform
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import engine, measure, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.stability import nvidia_snapshot
from examples import normalization


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple
    operation: str
    axis: object


CASES = (
    Case("max-row-3x8193", (3, 8193), "max", 1),
    Case("max-row-1024x1024", (1024, 1024), "max", 1),
    Case("min-row-8193x33", (8193, 33), "min", 1),
    Case("max-column-8193x33", (8193, 33), "max", 0),
    Case("max-total-1048577", (1048577,), "max", None),
    Case("min-tuple-17x129x33", (17, 129, 33), "min", (0, 2)),
    Case("max-tiny-3x17", (3, 17), "max", 1),
    Case("softmax-32x8193", (32, 8193), "softmax", -1),
    Case("softmax-1024x1024", (1024, 1024), "softmax", -1),
    Case("softmax-8193x33", (8193, 33), "softmax", -1),
    Case("logsumexp-32x8193", (32, 8193), "logsumexp", -1),
)
MODES = ("off-a", "on", "off-b")
SCOPES = ("prepared", "complete")
BACKENDS = ("opencl", "vulkan", "cuda-custom")


def case_function(case):
    """Bind immutable axis/operation metadata in a unary function's closure."""
    axis = case.axis
    operation = {
        "min": p.min,
        "max": p.max,
        "softmax": normalization.softmax,
        "logsumexp": normalization.logsumexp,
    }[case.operation]

    def calculate(a):
        return operation(a, axis=axis)

    return calculate


def reference_result(case, host):
    """Evaluate identical float32 data in float64, with a stable max shift."""
    a = host.astype(np.float64)
    if case.operation in ("min", "max"):
        return np.asarray(getattr(np, case.operation)(a, axis=case.axis))
    maximum = np.max(a, axis=case.axis, keepdims=True)
    weights = np.exp(a - maximum)
    total = np.sum(weights, axis=case.axis, keepdims=True)
    return weights / total if case.operation == "softmax" else maximum + np.log(total)


def rewrite_extrema(prepared, enabled=True):
    """Force materialized extrema to map-reduce or eager, independent of defaults.

    Bare materialized inputs use an empty map group and the existing input
    expression as their leaf. An existing map-reduce step qualifies only when
    its arguments are exactly those input expressions; fused producers remain
    untouched. Every plan must contain at least one qualifying extrema step.
    Return the number of replaced steps: on re-prepares every qualifying step;
    off replaces only map-reduce steps and can return zero for an eager default.
    Graph, bindings, liveness and execution scope stay unchanged. Lowering
    failures propagate without a partial edit.
    """
    plan = prepared._plan
    if enabled and plan.lowering is None:
        raise ValueError("Candidate requires a native map-reduce lowerer")
    steps, count, eligible = [], 0, 0
    for step in plan.steps:
        node, kind, existing_arguments, _ = step
        leaves = tuple(node.inputs)
        materialized = len(existing_arguments) == len(leaves) and all(
            argument is leaf for argument, leaf in zip(existing_arguments, leaves)
        )
        if (
            kind in ("reduce", "map_reduce")
            and materialized
            and node.attrs.get("operation") in ("min", "max")
        ):
            eligible += 1
            if not enabled:
                if kind == "map_reduce":
                    steps.append((node, "reduce", node.inputs, None))
                    count += 1
                else:
                    steps.append(step)
                continue
            arguments, payload = plan.lowering.prepare_map_reduce(node, (), leaves)
            arguments = tuple(arguments)
            if len(arguments) != len(leaves) or any(
                argument is not leaf for argument, leaf in zip(arguments, leaves)
            ):
                raise AssertionError("Bare extrema lowering changed plan dependencies")
            steps.append((node, "map_reduce", arguments, payload))
            count += 1
        else:
            steps.append(step)
    if not eligible:
        raise AssertionError("Plan did not find any materialized extrema steps")
    plan.steps = steps
    return count


def prepare_calls(backend, host, function):
    """Prepare independent controls/candidate on the same resident input."""
    resident = p.asarray(host, backend=backend)
    calls, metadata = {}, {}
    for mode in MODES:
        started = time.perf_counter_ns()
        prepared = p.compile(function).prepare(resident)
        compile_seconds = (time.perf_counter_ns() - started) * 1e-9
        before = prepared.explain()
        started = time.perf_counter_ns()
        transformed = rewrite_extrema(prepared, enabled=mode == "on")
        rewrite_seconds = (time.perf_counter_ns() - started) * 1e-9
        metadata[mode] = {
            "compile_prepare_seconds": compile_seconds,
            "rewrite_seconds": rewrite_seconds,
            "plan_before": before,
            "plan_after": prepared.explain(),
            "transformed_steps": transformed,
            "forced_route": "map_reduce" if mode == "on" else "reduce",
        }
        calls[mode] = {
            "prepared": lambda prepared=prepared: prepared(resident),
            "complete": lambda prepared=prepared: p.asnumpy(
                prepared(p.asarray(host, backend=backend))
            ),
        }
    return calls, metadata


def validate_result(actual, reference, case):
    """Require exact finite extrema, or the fixed normalization tolerance."""
    actual = np.asarray(actual)
    if actual.shape != reference.shape or actual.dtype != np.dtype("float32"):
        raise ValueError("Expected float32 output with the reference shape")
    rtol, atol = (0.0, 0.0) if case.operation in ("min", "max") else (2e-5, 2e-6)
    finite = np.isfinite(actual) & np.isfinite(reference)
    with np.errstate(invalid="ignore", over="ignore"):
        violations = ~finite | (
            np.abs(actual.astype(np.float64) - reference) > atol + rtol * np.abs(reference)
        )
    return {
        "rtol": rtol,
        "atol": atol,
        "finite": bool(np.all(finite)),
        "reference_close": not bool(np.any(violations)),
        "violations": int(np.count_nonzero(violations)),
    }


def mode_orders(repeats, seed):
    """Every permutation occurs equally often, then order is seeded/shuffled."""
    orders = list(itertools.permutations(MODES)) * (repeats // 6)
    random.Random(seed).shuffle(orders)
    return orders


def _bits(actual):
    actual = np.asarray(actual)
    if actual.dtype != np.dtype("float32"):
        raise ValueError("Bitwise validation requires float32 outputs")
    return actual.view(np.uint32).copy()


def _observe(actual, reference, case, control_bits):
    validation = validate_result(actual, reference, case)
    validation["bitwise_control_equal"] = bool(np.array_equal(_bits(actual), control_bits))
    return validation, error_metrics(actual, reference)


def run(output, backend_name, repeats=30, seed=997):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output file")
    if backend_name not in BACKENDS:
        raise ValueError("Select one of opencl, vulkan or cuda-custom")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 6 or repeats % 6:
        raise ValueError("Repeats must be an integer multiple of six, at least six")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    package = Path(p.__file__).resolve().parent
    indices = list(range(len(CASES)))
    random.Random(seed).shuffle(indices)
    report = {
        "schema": "openpacho.extrema-planning.v2",
        "backend": backend_name,
        "seed": seed,
        "repeats": repeats,
        "case_order": indices,
        "cases": [
            {"name": case.name, "shape": case.shape, "operation": case.operation, "axis": case.axis}
            for case in CASES
        ],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_origin": str(Path(p.__file__).resolve()),
        "runtime_sources": {
            str(file.relative_to(package)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(package.rglob("*.py"))
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pre_promotion_runner_sha256": "ace492057524afce9467aea8b896af0f750c1d098c1181935549be2ad443cbe9",
        "measurement_sha256": hashlib.sha256(
            Path(__file__).with_name("backend_sharing.py").read_bytes()
        ).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(normalization.__file__).read_bytes()).hexdigest(),
        "data": "Per case default_rng(seed * 101 + case_index), uniform[-4,4], float32",
        "candidate": "Each independent prepared plan forces materialized min/max steps: "
        "off-a/off-b use eager reduce, on uses prepare_map_reduce(node, (), tuple(node.inputs)). "
        "Existing map-reduce qualifies only if its arguments are identity-equal to node.inputs; "
        "fused producer steps remain unchanged. Selection is independent of compiler defaults. "
        "Graph, bindings, liveness counts, execution scope and runtime policy unchanged. "
        "transformed_steps counts replaced steps, including re-preparation on the candidate; "
        "an already-eager control requires no replacement.",
        "timing_scope": "Three independent plans off-a/on/off-b share identical resident "
        "inputs; prepared includes allocations, compute and synchronization; complete adds "
        "upload and download. Each scope records equally frequent six mode permutations. "
        "Compilation, first calls, warmup, reference and bitwise checks are excluded.",
        "warmup_policy": "At least three calls and 0.1 seconds per mode per scope",
        "validation_policy": "Before and after samples, every mode in both scopes must "
        "match pre-timing off-a output bits. Finite extrema are exact against float64; "
        "normalization uses max-shift float64 with rtol=2e-5, atol=2e-6. Reference failures "
        "remain visible and timed; nonfinite or changed bits abort. Completion alone "
        "does not imply numerical acceptance.",
        "rows": [],
        "complete": False,
        "all_reference_close": True,
        "all_bitwise_control_equal": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path exclusively, including against a concurrent worker.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    backend = None
    failed = False

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        _, backend = engine(backend_name)
        report["device"] = backend.device
        with threadpool_limits(limits=1):
            for index in indices:
                case = CASES[index]
                report["active_case"] = case.name
                host = np.random.default_rng(seed * 101 + index).uniform(-4, 4, case.shape)
                host = host.astype(np.float32)
                reference = reference_result(case, host)
                telemetry_before = nvidia_snapshot()
                calls, metadata = prepare_calls(backend, host, case_function(case))
                rows, control_bits = {}, None
                for mode in MODES:
                    row = {
                        "case_index": index,
                        "case": case.name,
                        "shape": case.shape,
                        "operation": case.operation,
                        "axis": case.axis,
                        "dtype": "float32",
                        "input_sha256": hashlib.sha256(host.tobytes()).hexdigest(),
                        "input_seed": seed * 101 + index,
                        "mode": mode,
                        **metadata[mode],
                        "first_call": {},
                        "validation": {"before": {}, "after": {}},
                        "error": {"before": {}, "after": {}},
                        "telemetry_before": telemetry_before,
                    }
                    rows[mode] = row
                    report["rows"].append(row)
                    for scope in SCOPES:
                        seconds, returned, value = measure(calls[mode][scope], p, backend)
                        row["first_call"][scope] = {
                            "seconds": seconds,
                            "returned_seconds": returned,
                        }
                        actual = p.asnumpy(value) if scope == "prepared" else value
                        if control_bits is None:
                            control_bits = _bits(actual)
                        validation, error = _observe(actual, reference, case, control_bits)
                        row["validation"]["before"][scope] = validation
                        row["error"]["before"][scope] = error
                        report["all_reference_close"] &= validation["reference_close"]
                        report["all_bitwise_control_equal"] &= validation["bitwise_control_equal"]
                        if not validation["finite"] or not validation["bitwise_control_equal"]:
                            raise AssertionError(
                                f"Initial output nonfinite or bits differ: {mode}/{scope}"
                            )
                        del value, actual
                for scope_index, scope in enumerate(SCOPES):
                    samples = {mode: [] for mode in MODES}
                    for mode in MODES:
                        rows[mode][scope + "_warmup"] = warmup(
                            calls[mode][scope], p, backend, seconds=0.1
                        )
                    orders = mode_orders(repeats, seed + index + 100 * scope_index)
                    for mode in MODES:
                        rows[mode][scope + "_orders"] = orders
                        # Keep partial samples if a later execution fails.
                        rows[mode][scope + "_samples_seconds"] = samples[mode]
                    for order in orders:
                        for mode in order:
                            seconds, _, value = measure(calls[mode][scope], p, backend)
                            samples[mode].append(seconds)
                            del value
                    for mode in MODES:
                        rows[mode][scope] = _sample_stats(samples[mode])
                # Verify both scopes after all timings, against original off-a bits.
                for mode in MODES:
                    for scope in SCOPES:
                        _, _, value = measure(calls[mode][scope], p, backend)
                        actual = p.asnumpy(value) if scope == "prepared" else value
                        validation, error = _observe(actual, reference, case, control_bits)
                        rows[mode]["validation"]["after"][scope] = validation
                        rows[mode]["error"]["after"][scope] = error
                        report["all_reference_close"] &= validation["reference_close"]
                        report["all_bitwise_control_equal"] &= validation["bitwise_control_equal"]
                        if not validation["finite"] or not validation["bitwise_control_equal"]:
                            raise AssertionError(
                                f"Replay output nonfinite or bits differ: {mode}/{scope}"
                            )
                        del value, actual
                telemetry_after = nvidia_snapshot()
                for row in rows.values():
                    row["telemetry_after"] = telemetry_after
                save()
                print(
                    case.name,
                    {
                        scope: {mode: rows[mode][scope]["median_seconds"] for mode in MODES}
                        for scope in SCOPES
                    },
                    flush=True,
                )
                del calls, metadata, rows, control_bits, host, reference
        report.pop("active_case", None)
        report["complete"] = True
    except BaseException as exc:
        failed = True
        report["execution_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        try:
            if backend is not None:
                close = getattr(backend, "close", None)
                (close if close is not None else backend.synchronize)()
        except BaseException as exc:
            report["complete"] = False
            report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
            if not failed:
                raise
        finally:
            save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=997)
    args = parser.parse_args(argv)
    try:
        report = run(args.output, args.backend, args.repeats, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
