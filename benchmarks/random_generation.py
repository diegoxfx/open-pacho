"""Random generation: ready-on-device, delivered-to-host and CPU-then-upload."""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import engine, sync, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.suite import source_hashes


def converted(function, convert):
    return convert(function())


def native_numpy(rng, distribution, shape, dtype):
    if distribution == "random":
        return rng.random(shape, dtype=dtype)
    if distribution == "normal":
        return rng.standard_normal(shape, dtype=dtype)
    if distribution == "exponential":
        return rng.standard_exponential(shape, dtype=dtype)
    # NumPy uniform has no dtype argument. Include the conversion in its clock.
    return rng.uniform(-3, 7, shape).astype(dtype, copy=False)


def moments(value, distribution):
    x = value.astype(np.float64).ravel()
    mean, variance = {
        "random": (0.5, 1 / 12),
        "uniform": (2, 100 / 12),
        "normal": (0, 1),
        "exponential": (1, 1),
    }[distribution]
    result = {
        "mean": float(x.mean()),
        "variance": float(x.var()),
        "expected_mean": mean,
        "expected_variance": variance,
        "minimum": float(x.min()),
        "maximum": float(x.max()),
        "nonfinite": int(np.count_nonzero(~np.isfinite(x))),
    }
    if result["nonfinite"]:
        raise AssertionError("Nonfinite random sample")
    if distribution in ("random", "uniform"):
        low, high = (0, 1) if distribution == "random" else (-3, 7)
        if not (np.all(x >= low) and np.all(x < high)):
            raise AssertionError("Uniform sample outside range")
    elif distribution == "exponential" and np.any(x < 0):
        raise AssertionError("Negative exponential sample")
    if x.size > 10000:
        if abs(result["mean"] - mean) > 8 * np.sqrt(variance / x.size):
            raise AssertionError("Mean outside eight-standard-error smoke bound")
        if abs(result["variance"] - variance) > variance * 0.07:
            raise AssertionError("Variance smoke bound failed")
        result["lag1_correlation"] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return result


def run(output, backends, repeats=21, seed=941):
    if output.exists() or repeats < 3 or len(backends) != 1:
        raise ValueError("One isolated backend, new output file and >=3 samples required")
    name = backends[0]
    xp, backend = engine(name)
    report = {
        "backend": name,
        "device": backend.device if backend is not None else name,
        "source_sha256": source_hashes(),
        "rows": [],
        "seed": seed,
        "repeats": repeats,
        "scope": "allocation + synchronized generation; download only in host-delivered; warmup/validation outside clocks",
        "reference_algorithms": "OpenPacho Philox4x32-10, NumPy default_rng PCG64, CuPy RandomState cuRAND default; no sequence equivalence claimed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with threadpool_limits(limits=1):
            for dtype in (np.dtype("float32"), np.dtype("float64")):
                for shape in ((33,), (257, 129), (1025, 1023)):
                    for distribution in ("random", "uniform", "normal", "exponential"):
                        row = {"distribution": distribution, "shape": shape, "dtype": str(dtype)}
                        report["rows"].append(row)
                        if backend is not None and distribution not in backend.random_distributions(
                            dtype
                        ):
                            row["status"] = "unsupported"
                            continue
                        if xp is p:
                            rng = p.random.default_rng(seed, backend=backend)
                            kwargs = {"low": -3, "high": 7} if distribution == "uniform" else {}
                            fn = partial(
                                getattr(rng, distribution), size=shape, dtype=dtype, **kwargs
                            )
                            host = p.asnumpy
                        elif xp is np:
                            rng = np.random.default_rng(seed)
                            fn = partial(native_numpy, rng, distribution, shape, dtype)
                            host = lambda v: v
                        else:
                            rng = xp.random.RandomState(seed)
                            method = "random_sample" if distribution == "random" else distribution
                            kwargs = {"low": -3, "high": 7} if distribution == "uniform" else {}
                            fn = partial(getattr(rng, method), size=shape, dtype=dtype, **kwargs)
                            host = xp.asnumpy
                        sync(xp, backend)
                        start = time.perf_counter_ns()
                        value = fn()
                        sync(xp, backend)
                        row["first_call_seconds"] = (time.perf_counter_ns() - start) * 1e-9
                        actual = host(value)
                        row["moments"] = moments(actual, distribution)
                        if xp is p:
                            reference = getattr(
                                p.random.default_rng(seed, backend="numpy"), distribution
                            )(size=shape, dtype=dtype, **kwargs)
                            ref = p.asnumpy(reference)
                            row["error"] = error_metrics(
                                actual,
                                ref,
                                reference="same Philox transform in NumPy",
                                precision=str(dtype),
                            )
                            if distribution == "random":
                                np.testing.assert_array_equal(actual, ref)
                            else:
                                np.testing.assert_allclose(
                                    actual,
                                    ref,
                                    rtol=5e-6 if dtype.itemsize == 4 else 1e-13,
                                    atol=5e-6 if dtype.itemsize == 4 else 1e-13,
                                )
                        modes = {"resident": fn, "host_delivered": partial(converted, fn, host)}
                        if xp is not np:
                            cpu_rng = np.random.default_rng(seed)
                            upload = (
                                (lambda a: p.asarray(a, backend=backend)) if xp is p else xp.asarray
                            )
                            modes["numpy_then_upload"] = partial(
                                converted,
                                partial(native_numpy, cpu_rng, distribution, shape, dtype),
                                upload,
                            )
                        for mode, function in modes.items():
                            warming = warmup(function, xp, backend)
                            times = []
                            for _ in range(repeats):
                                sync(xp, backend)
                                start = time.perf_counter_ns()
                                value = function()
                                sync(xp, backend)
                                times.append((time.perf_counter_ns() - start) * 1e-9)
                                del value
                            row[mode] = {"warmup": warming, **_sample_stats(times)}
                        row["status"] = "ok"
        report["complete"] = True
    finally:
        sync(xp, backend)
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
