"""Paired opt-in GEMM padding: full costs, precision and scratch allocation.

Run via benchmarks.isolated for observed idle gates between GPU backends.
The prepacked diagnostic is host wall time (dispatch/allocation/synchronization),
not GPU event time, and excludes packing/cropping. It is not an API speedup.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import engine, measure, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.suite import source_hashes
from openpacho._array import ndarray
from openpacho._gemm import register_gemm_data


def cases():
    return [
        ((33, 17), (17, 65)),
        ((257, 255), (255, 259)),
        ((1025, 1023), (1023, 1027)),
        ((2049, 2047), (2047, 2051)),
        ((1024, 1024), (1024, 1024)),
        ((2, 1, 65, 47), (1, 3, 47, 67)),
    ]


def product(a, b):
    return a @ b


def build_calls(backend, inputs, padding):
    upload = lambda: tuple(p.asarray(x, backend=backend) for x in inputs)
    resident = upload()
    function = p.compile(product, padding=padding)
    prepared = function.prepare(*resident)
    return (
        {
            "resident": lambda: function(*resident),
            "prepared": lambda: prepared(*resident),
            "eager": lambda: p.matmul(*resident, padding=padding),
            "complete": lambda: p.asnumpy(function(*upload())),
        },
        function.explain(*resident),
        resident,
    )


def prepacked(backend, resident):
    """Separate diagnostic; no reuse promise for mutable public input arrays."""
    a, b = resident
    shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
    plan = backend.prepare_padded_matmul(a.shape, b.shape, shape, a.dtype)
    if plan is None:
        return None
    packed = tuple(
        ndarray._wrap(plan.copy(backend, value, target), target, value.dtype, backend)
        if value.shape != target
        else value
        for value, target in zip(resident, plan.padded_input_shapes)
    )

    def call():
        data = register_gemm_data(
            backend, *packed, plan.padded_output_shape, specialize_bounds=True
        )
        return ndarray._wrap(data, plan.padded_output_shape, a.dtype, backend)

    return call


def run(output, backends, repeats=21, seed=941):
    if output.exists() or len(backends) != 1 or repeats < 3:
        raise ValueError("Use a new output, one backend and at least three samples")
    name = backends[0]
    if name not in ("opencl", "vulkan", "cuda-custom"):
        raise ValueError("Select a supported custom GPU backend")
    output.parent.mkdir(parents=True, exist_ok=True)
    _, backend = engine(name)
    report = {
        "schema": "openpacho.padding.v1",
        "backend": name,
        "device": backend.device,
        "seed": seed,
        "repeats": repeats,
        "source_sha256": source_hashes(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "paired alternating padding false/true; full pack/GEMM/crop included in all public API clocks; no numerical-policy changes",
        "prepacked_scope": "diagnostic only: packing and cropping excluded; host wall including dispatch, allocation, synchronization, not device event time",
        "rows": [],
        "complete": False,
    }
    try:
        with threadpool_limits(limits=1):
            for index, (ashape, bshape) in enumerate(cases()):
                rng = np.random.default_rng(seed + index)
                inputs = tuple(
                    rng.uniform(-0.5, 0.5, s).astype("float32") for s in (ashape, bshape)
                )
                reference = inputs[0].astype("float64") @ inputs[1].astype("float64")
                expected = product(*inputs)
                functions, rows, outputs = {}, {}, {}
                order = (False, True) if index % 2 else (True, False)
                for flag in order:
                    calls, explanation, resident = build_calls(backend, inputs, flag)
                    first, _, value = measure(calls["resident"], p, backend)
                    actual = p.asnumpy(value)
                    del value
                    outputs[flag] = actual
                    rows[flag] = {
                        "input_shapes": (ashape, bshape),
                        "dtype": "float32",
                        "padding": flag,
                        "explanation": explanation,
                        "first_execution_seconds": first,
                        "error": error_metrics(actual, reference),
                        "validation": {
                            "rtol": 5e-5,
                            "atol": 5e-5,
                            "float64_reference_close": bool(
                                np.allclose(actual, reference, rtol=5e-5, atol=5e-5)
                            ),
                            "numpy_same_dtype_close": bool(
                                np.allclose(actual, expected, rtol=5e-5, atol=5e-5)
                            ),
                        },
                    }
                    report["rows"].append(rows[flag])
                    np.testing.assert_allclose(actual, reference, rtol=5e-5, atol=5e-5)
                    if flag:
                        diagnostic = prepacked(backend, resident)
                        if diagnostic is not None:
                            calls["prepacked_wall"] = diagnostic
                    functions[flag] = calls
                rows[True]["exact_unpadded_equality"] = bool(
                    np.array_equal(outputs[True], outputs[False])
                )
                for scope in ("resident", "prepared", "eager", "complete", "prepacked_wall"):
                    modes = [flag for flag in order if scope in functions[flag]]
                    samples = {flag: [] for flag in modes}
                    for flag in modes:
                        rows[flag][scope + "_warmup"] = warmup(functions[flag][scope], p, backend)
                        _, _, value = measure(functions[flag][scope], p, backend)
                        actual = p.asnumpy(value) if isinstance(value, ndarray) else value
                        if scope == "prepacked_wall":
                            actual = actual[..., : ashape[-2], : bshape[-1]]
                        np.testing.assert_allclose(actual, reference, rtol=5e-5, atol=5e-5)
                        del value
                    for pair in range(repeats):
                        for flag in modes if pair % 2 else modes[::-1]:
                            seconds, _, value = measure(functions[flag][scope], p, backend)
                            samples[flag].append(seconds)
                            del value
                    for flag in modes:
                        rows[flag][scope] = _sample_stats(samples[flag])
                print(
                    ashape,
                    bshape,
                    {
                        flag: round(rows[flag]["resident"]["median_seconds"] * 1e6, 1)
                        for flag in order
                    },
                    flush=True,
                )
        report["complete"] = True
    finally:
        try:
            getattr(backend, "close", backend.synchronize)()
        finally:
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="opencl")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=941)
    args = parser.parse_args()
    run(args.output, args.backends.split(","), args.repeats, args.seed)
