"""Paired RNG routing control on one backend/context with unchanged kernels."""

import argparse
import hashlib
import json
import random
import time
import types
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import engine, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.random_generation import converted, moments
from benchmarks.suite import source_hashes


def historical_rng(wheel_path):
    with zipfile.ZipFile(wheel_path) as wheel:
        source = wheel.read("openpacho/_random.py")
    module = types.ModuleType("openpacho._routing_control")
    module.__package__ = "openpacho"
    # Explicit trusted local wheel, never downloaded or inferred executable code.
    exec(compile(source, str(wheel_path), "exec"), module.__dict__)  # noqa: S102
    return module, hashlib.sha256(source).hexdigest()


def run(output, backends, repeats, seed, control_wheel):
    if output.exists() or len(backends) != 1 or repeats < 3 or control_wheel is None:
        raise ValueError("Select one backend, new output, >=3 samples and a historical wheel")
    if backends[0] not in ("opencl", "vulkan", "cuda-custom", "cuda-cublas"):
        raise ValueError("Routing control requires an OpenPacho GPU backend")
    output.parent.mkdir(parents=True, exist_ok=True)
    legacy, digest = historical_rng(control_wheel)
    _, backend = engine(backends[0])
    current = backend.random_data
    previous = types.MethodType(legacy.random_data, backend)
    report = {
        "schema": "openpacho.routing.v1",
        "backend": backends[0],
        "device": backend.device,
        "source_sha256": source_hashes(),
        "control_source_sha256": digest,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "repeats": repeats,
        "rows": [],
        "scope": "same device/context/kernels and synchronization; public RNG allocation included; warmup/validation excluded; complete also downloads",
    }
    order = ["baseline", "adapter"]
    random.Random(seed).shuffle(order)
    specs = [
        (dtype, shape, distribution)
        for dtype in (np.dtype("float32"), np.dtype("float64"))
        for shape in ((33,), (257, 129), (1025, 1023))
        for distribution in ("random", "uniform", "normal", "exponential")
    ]
    try:
        with threadpool_limits(limits=1):
            for index, (dtype, shape, distribution) in enumerate(specs):
                if distribution not in backend.random_distributions(dtype):
                    report["rows"].append(
                        {
                            "dtype": str(dtype),
                            "shape": shape,
                            "distribution": distribution,
                            "status": "unsupported",
                        }
                    )
                    continue
                expected = None
                kwargs = {"low": -3, "high": 7} if distribution == "uniform" else {}
                reference = p.asnumpy(
                    getattr(p.random.default_rng(seed, backend="numpy"), distribution)(
                        size=shape, dtype=dtype, **kwargs
                    )
                )
                for mode in order if index % 2 == 0 else list(reversed(order)):
                    backend.synchronize()
                    backend.random_data = previous if mode == "baseline" else current
                    rng = p.random.default_rng(seed, backend=backend)
                    fn = partial(getattr(rng, distribution), size=shape, dtype=dtype, **kwargs)
                    actual = p.asnumpy(fn())
                    if expected is None:
                        expected = actual
                    else:
                        np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_allclose(
                        actual,
                        reference,
                        rtol=5e-6 if dtype.itemsize == 4 else 1e-13,
                        atol=5e-6 if dtype.itemsize == 4 else 1e-13,
                    )
                    row = {
                        "dtype": str(dtype),
                        "shape": shape,
                        "distribution": distribution,
                        "mode": mode,
                        "status": "ok",
                        "moments": moments(actual, distribution),
                        "error": error_metrics(
                            actual,
                            reference,
                            reference="same Philox transform in NumPy",
                            precision=str(dtype),
                        ),
                    }
                    for delivery, call in (
                        ("resident", fn),
                        ("complete", partial(converted, fn, p.asnumpy)),
                    ):
                        warming = warmup(call, p, backend)
                        samples = []
                        for _ in range(repeats):
                            backend.synchronize()
                            start = time.perf_counter_ns()
                            value = call()
                            backend.synchronize()
                            samples.append((time.perf_counter_ns() - start) * 1e-9)
                            del value
                        row[delivery] = {"warmup": warming, **_sample_stats(samples)}
                    report["rows"].append(row)
        report["complete"] = True
    finally:
        backend.random_data = current
        backend.synchronize()
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="opencl")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--control-wheel", type=Path, required=True)
    parser.add_argument("--control-component", default="methods", choices=["methods"])
    args = parser.parse_args()
    run(args.output, args.backends.split(","), args.repeats, args.seed, args.control_wheel)
