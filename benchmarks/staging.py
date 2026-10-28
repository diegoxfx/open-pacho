"""Same-device Vulkan mapping control; isolated runner supplies idle gates."""

import argparse
import hashlib
import json
import os
import random
import types
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.suite import source_hashes
from benchmarks.vulkan_dispatch import Diagnostic, measure, workload
from openpacho._vulkan import VulkanBackend


def run(output, backends, repeats=21, seed=941, control_wheel=None):
    if backends != ["vulkan"] or output.exists() or repeats < 3:
        raise ValueError("Use Vulkan only, a new output file and at least three repetitions")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"rows": [], "source_sha256": source_hashes(), "repeats": repeats, "seed": seed}
    backend = VulkanBackend(os.environ.get("PACHO_VULKAN_DEVICE"))
    current_methods = (backend.upload, backend.download)
    old_methods = None
    if control_wheel is not None:
        with zipfile.ZipFile(control_wheel) as wheel:
            source = wheel.read("openpacho/_vulkan.py")
        module = types.ModuleType("openpacho._staging_control")
        module.__package__ = "openpacho"
        exec(compile(source, str(control_wheel), "exec"), module.__dict__)  # noqa: S102 - explicit local historical wheel control
        old_methods = tuple(
            types.MethodType(getattr(module.VulkanBackend, method), backend)
            for method in ("upload", "download")
        )
        report["control_source_sha256"] = hashlib.sha256(source).hexdigest()
    report["device"] = backend.device
    report["scope"] = (
        "same backend/context, mode blocks with identical warmup; allocation included; diagnostic separate"
    )
    order = ["transient", "readback"]
    random.Random(seed).shuffle(order)
    specs = [
        (op, shape, dtype)
        for dtype in ("float32", "float64")
        for op, shape in (
            ("upload", (1,)),
            ("download", (1,)),
            ("upload", (1048576,)),
            ("download", (1048576,)),
            ("energy", (2049, 33)),
            ("gemm", (257, 129, 513)),
            ("covariance", (1025, 65)),
            ("cg", (129, 129)),
        )
    ]
    try:
        with threadpool_limits(limits=1):
            for index, spec in enumerate(specs):
                fn, expected, metadata = workload(backend, spec)
                for mode in order if index % 2 == 0 else list(reversed(order)):
                    backend.release_cached_memory()
                    backend._staging_mode = mode
                    backend.upload, backend.download = (
                        old_methods if mode == "transient" and old_methods else current_methods
                    )
                    warming = warmup(fn, p, backend)
                    times, full_times = [], []
                    for _ in range(repeats):
                        seconds, value = measure(backend, fn)
                        times.append(seconds)
                        del value
                    value, diagnostic = Diagnostic(backend).run(fn)
                    actual = p.asnumpy(value) if isinstance(value, p.ndarray) else value
                    numeric = error_metrics(actual, expected)
                    if hasattr(fn, "validate"):
                        numeric.update(fn.validate(actual))
                    else:
                        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
                    if hasattr(fn, "end_to_end"):
                        warmup(fn.end_to_end, p, backend)
                        for _ in range(repeats):
                            seconds, _value = measure(backend, fn.end_to_end)
                            full_times.append(seconds)
                    report["rows"].append(
                        {
                            "case": spec,
                            "mode": mode,
                            "warmup": warming,
                            "wall": _sample_stats(times),
                            "samples_seconds": times,
                            "end_to_end": _sample_stats(full_times) if full_times else None,
                            "end_to_end_samples_seconds": full_times,
                            "error": numeric,
                            "diagnostic": diagnostic,
                            "metadata": metadata,
                        }
                    )
                    del value, actual
                del fn
        report["complete"] = True
    finally:
        backend.upload, backend.download = current_methods
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        backend.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backends", default="vulkan")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--control-wheel", type=Path)
    parser.add_argument("--control-component", default="methods", choices=["methods"])
    args = parser.parse_args()
    run(args.output, args.backends.split(","), args.repeats, args.seed, args.control_wheel)
