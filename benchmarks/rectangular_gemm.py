"""Paired small-row GEMM selection versus the existing direct GEMM.

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
from benchmarks.padding_decision import validate_result
from benchmarks.stability import nvidia_snapshot
from openpacho._gemm_rect import select_tile

CASES = tuple(
    ((m, k), (k, n))
    for m, k, n in (
        (65, 8191, 2051),
        (64, 8191, 2051),
        (65, 4095, 2051),
        (65, 8191, 2048),
        (95, 8191, 2051),
        (32, 8191, 4096),
        (33, 2049, 1025),
        (63, 4095, 1023),
        (65, 8191, 257),
        (65, 16383, 2051),
        (65, 4080, 2051),
        (65, 4081, 2051),
        (65, 4096, 2051),
        (32, 2048, 1024),
        (96, 8192, 2048),
        (97, 8191, 2051),
        (65, 8191, 4096),
        (64, 8191, 4096),
        (1024, 1024, 1024),
        (2049, 2047, 2051),
    )
) + (((2, 65, 4095), (4095, 257)),)
MODES = ("off-a", "on", "off-b")
SCOPES = ("prepared", "complete")


@contextmanager
def preserve_configuration(backend):
    """Restore the exact instance override, including its original absence."""
    sentinel = object()
    original = vars(backend).get("_small_row_gemm_config", sentinel)
    try:
        yield getattr(backend, "_small_row_gemm_config", None)
    finally:
        if original is sentinel:
            vars(backend).pop("_small_row_gemm_config", None)
        else:
            backend._small_row_gemm_config = original


def product(a, b):
    return a @ b


def prepare_calls(backend, host, original_config):
    """Capture three plans, activate their execution-time selector outside clocks."""
    resident = tuple(p.asarray(value, backend=backend) for value in host)
    calls, explanations = {}, {}
    for mode in MODES:
        config = original_config if mode == "on" else None

        def activate(config=config):
            backend._small_row_gemm_config = config

        compiled = p.compile(product)
        prepared = compiled.prepare(*resident)
        with preserve_configuration(backend):
            activate()
            tile = select_tile(backend, host[0].shape, host[1].shape)
        explanations[mode] = {
            **prepared.explain(),
            "candidate_config": config,
            "selected_tile": tile,
            "selected_route": "small-row" if tile is not None else "direct-fallback",
        }
        calls[mode] = {
            "prepared": lambda prepared=prepared: prepared(*resident),
            "complete": lambda prepared=prepared: p.asnumpy(
                prepared(*(p.asarray(value, backend=backend) for value in host))
            ),
        }
        for function in calls[mode].values():
            function._activate_variant = activate
    return calls, explanations


def run(output, backends, repeats=30, seed=991):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output file")
    if len(backends) != 1 or backends[0] not in ("opencl", "cuda-custom"):
        raise ValueError("Select one of opencl or cuda-custom")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 6 or repeats % 6:
        raise ValueError("Repeats must be an integer multiple of six, at least six")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    output.parent.mkdir(parents=True, exist_ok=True)
    package = Path(p.__file__).resolve().parent
    indices = list(range(len(CASES)))
    random.Random(seed).shuffle(indices)
    report = {
        "schema": "openpacho.rectangular-gemm.v1",
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
        "scope": "Three independent plans on identical resident arrays: off-a/on/off-b; controls disable small-row tile selection during execution; selection activation precedes clocks; prepared includes allocations/compute/copies/synchronization; complete adds uploads and download; six mode permutations equally frequent; warmup and validation outside clocks.",
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
        with preserve_configuration(backend) as original_config, threadpool_limits(limits=1):
            report["candidate_config"] = original_config
            for index in indices:
                shapes = CASES[index]
                rng = np.random.default_rng(seed * 101 + index)
                host = tuple(rng.uniform(-0.5, 0.5, shape).astype("float32") for shape in shapes)
                reference = host[0].astype("float64") @ host[1].astype("float64")
                calls, explanations = prepare_calls(backend, host, original_config)
                rows, bits = {}, {}
                for mode in MODES:
                    calls[mode]["prepared"]._activate_variant()
                    actual = p.asnumpy(calls[mode]["prepared"]())
                    validation = validate_result(actual, reference)
                    row = {
                        "case_index": index,
                        "case": shapes,
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
                            f"Numerical validation failed: {shapes} {mode}: {validation['violations']} violations",
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
                    raise AssertionError("Two direct GEMM controls differ")
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
                    shapes,
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
    parser.add_argument("--backends", default="opencl")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=991)
    args = parser.parse_args(argv)
    try:
        report = run(args.output, args.backends.split(","), args.repeats, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
