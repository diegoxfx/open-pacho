"""Same-device old/idle-elision/fence control; GPU idle gates live in isolated."""

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


def run(output, backends, repeats, seed, control_wheel):
    if output.exists() or backends != ["vulkan"] or repeats < 3 or control_wheel is None:
        raise ValueError("Select Vulkan, a new output, >=3 samples and a historical wheel")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(control_wheel) as wheel:
        source = wheel.read("openpacho/_vulkan.py")
    module = types.ModuleType("openpacho._completion_control")
    module.__package__ = "openpacho"
    exec(compile(source, str(control_wheel), "exec"), module.__dict__)  # noqa: S102 - explicit trusted local control
    backend = VulkanBackend(os.environ.get("PACHO_VULKAN_DEVICE"))
    current = backend.synchronize
    previous = types.MethodType(module.VulkanBackend.synchronize, backend)
    report = {
        "schema": "openpacho.completion.v1",
        "rows": [],
        "source_sha256": source_hashes(),
        "control_source_sha256": hashlib.sha256(source).hexdigest(),
        "device": backend.device,
        "seed": seed,
        "repeats": repeats,
        "scope": "same VkDevice/context and kernels; warmup >=0.25s per mode; allocation included; diagnostics separate",
    }
    modes = ["baseline", "queue", "fence"]
    random.Random(seed).shuffle(modes)
    specs = [
        (op, shape, dtype)
        for dtype in ("float32", "float64")
        for op, shape in (
            ("idle", (1,)),
            ("add", (33,)),
            ("download", (1,)),
            ("download", (1048576,)),
            ("energy", (2049, 33)),
            ("covariance", (1025, 65)),
            ("cg", (129, 129)),
            ("sd", (129, 129)),
        )
    ]
    try:
        with threadpool_limits(limits=1):
            for index, spec in enumerate(specs):
                if spec[0] == "idle":
                    expected = np.zeros(1, spec[2])
                    fn = lambda expected=expected: (backend.synchronize(), expected)[1]
                    metadata = {"scope": "explicit synchronize on an empty owned queue"}
                else:
                    fn, expected, metadata = workload(backend, spec)
                for mode in modes if index % 2 == 0 else list(reversed(modes)):
                    backend.synchronize()
                    backend.synchronize = previous if mode == "baseline" else current
                    backend._completion_mode = "fence" if mode == "fence" else "queue"
                    warming = warmup(fn, p, backend)
                    samples = []
                    for _ in range(repeats):
                        seconds, value = measure(backend, fn)
                        samples.append(seconds)
                        del value
                    value, diagnostic = Diagnostic(backend).run(fn)
                    actual = p.asnumpy(value) if isinstance(value, p.ndarray) else value
                    del value
                    precision = str(np.asarray(expected).dtype)
                    metrics = error_metrics(
                        actual, expected, reference=f"numpy-{precision}", precision=precision
                    )
                    if hasattr(fn, "validate"):
                        metrics.update(fn.validate(actual))
                    else:
                        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
                    row = {
                        "case": spec,
                        "mode": mode,
                        "warmup": warming,
                        "resident": _sample_stats(samples),
                        "error": metrics,
                        "diagnostic": diagnostic,
                        "metadata": metadata,
                        "iterations": getattr(fn, "iterations", None),
                    }
                    if hasattr(fn, "end_to_end"):
                        warmup(fn.end_to_end, p, backend)
                        full = []
                        for _ in range(repeats):
                            seconds, value = measure(backend, fn.end_to_end)
                            full.append(seconds)
                            del value
                        row["complete"] = _sample_stats(full)
                    report["rows"].append(row)
                    del actual
                del fn
        report["complete"] = True
    finally:
        backend.synchronize = current
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        backend.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="vulkan")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--control-wheel", type=Path, required=True)
    parser.add_argument("--control-component", default="methods", choices=["methods"])
    args = parser.parse_args()
    run(args.output, args.backends.split(","), args.repeats, args.seed, args.control_wheel)
