"""Paired copy-preparation/scope controls, with identical current GEMM kernels.

The explicit historical wheel contributes only _padding_copy.py. The current
compiler, backend and GEMM execute every mode. Run in an isolated process.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import engine, measure, warmup
from benchmarks.cuda_dispatch import diagnose
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.padding import cases, product
from benchmarks.suite import source_hashes
from openpacho import _padding_copy


def load_control(wheel):
    with zipfile.ZipFile(wheel) as archive:
        source = archive.read("openpacho/_padding_copy.py")
    namespace = {"__name__": "padding_copy_control"}
    exec(compile(source, str(wheel), "exec"), namespace)  # noqa: S102 - explicit local control
    return namespace, {
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "copy_source_sha256": hashlib.sha256(source).hexdigest(),
        "scope": "only _padding_copy.py; all GEMM kernels and backend code are current",
    }


def prepare_calls(backend, inputs, old):
    """Construct independent plans; never change shared callables inside clocks."""
    resident = tuple(p.asarray(x, backend=backend) for x in inputs)
    key = backend.name + "_copy"
    copies = {"old_copy": old[key], "cached_copy": getattr(_padding_copy, key)}
    calls = {}
    for mode, copy in copies.items():
        with patch.object(_padding_copy, key, copy):
            prepared = p.compile(product, padding=True).prepare(*resident)
        # Ablation: historical logical-step rule did not scope one padded GEMM.
        if backend.name == "cuda":
            prepared._plan._execution_scope = None
        calls[mode] = lambda prepared=prepared: prepared(*resident)
    current = p.compile(product, padding=True).prepare(*resident)
    calls["current"] = lambda: current(*resident)
    ordinary = p.compile(product).prepare(*resident)
    calls["unpadded"] = lambda: ordinary(*resident)
    return calls


def assert_identical_copy_sources(old, ashape, bshape):
    m, k, n = ashape[-2], ashape[-1], bshape[-1]
    mp, kp, npad = ((x + t - 1) // t * t for x, t in zip((m, k, n), (32, 16, 32)))
    batch = np.broadcast_shapes(ashape[:-2], bshape[:-2])
    pairs = [
        (ashape, ashape[:-2] + (mp, kp)),
        (bshape, bshape[:-2] + (kp, npad)),
        (batch + (mp, npad), batch + (m, n)),
    ]
    for source, target in pairs:
        for key in ("_opencl_source", "_cuda_source"):
            assert old[key](source, target) == getattr(_padding_copy, key)(source, target)
        for index_type in ("uint", "unsigned int"):
            assert old["_body"](source, target, index_type) == _padding_copy._body(
                source, target, index_type
            )


def run(output, backends, repeats=21, seed=947, control_wheel=None):
    if output.exists() or len(backends) != 1 or repeats < 3 or control_wheel is None:
        raise ValueError("Use one backend, a new output, >=3 repeats and a control wheel")
    name = backends[0]
    if name not in ("opencl", "vulkan", "cuda-custom"):
        raise ValueError("Select a custom GPU backend")
    old, control = load_control(control_wheel)
    _, backend = engine(name)
    report = {
        "schema": "openpacho.padding-copy-review.v1",
        "backend": name,
        "device": backend.device,
        "control": control,
        "source_sha256": source_hashes(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "repeats": repeats,
        "scope": "prepared public calls with resident inputs, includes allocation/pack/GEMM/crop/synchronization; current GEMM in all modes",
        "rows": [],
        "complete": False,
    }
    try:
        with threadpool_limits(limits=1):
            for index, shapes in enumerate(cases()):
                assert_identical_copy_sources(old, *shapes)
                rng = np.random.default_rng(seed + index)
                host = tuple(rng.uniform(-0.5, 0.5, s).astype("float32") for s in shapes)
                reference = host[0].astype("float64") @ host[1].astype("float64")
                calls = prepare_calls(backend, host, old)
                rows, values = {}, {}
                for mode, call in calls.items():
                    w = warmup(call, p, backend)
                    actual = p.asnumpy(call())
                    np.testing.assert_allclose(actual, reference, rtol=5e-5, atol=5e-5)
                    values[mode] = actual.view(np.uint32)
                    rows[mode] = {
                        "shapes": shapes,
                        "mode": mode,
                        "error": error_metrics(actual, reference),
                        "warmup": w,
                        "samples": [],
                    }
                for mode in calls:
                    np.testing.assert_array_equal(values[mode], values["old_copy"])
                    rows[mode]["bitwise_control_equality"] = True
                modes = list(calls)
                for sample in range(repeats):
                    order = modes[sample % len(modes) :] + modes[: sample % len(modes)]
                    for mode in order if sample % 2 else order[::-1]:
                        seconds, _, result = measure(calls[mode], p, backend)
                        rows[mode]["samples"].append(seconds)
                        del result
                for mode, row in rows.items():
                    row["wall"] = _sample_stats(row.pop("samples"))
                    if backend.name == "cuda":
                        row["diagnostic"] = diagnose(backend, calls[mode])
                    report["rows"].append(row)
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(
                    shapes,
                    {m: round(r["wall"]["median_seconds"] * 1e6, 2) for m, r in rows.items()},
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
    parser.add_argument("--seed", type=int, default=947)
    parser.add_argument("--control-wheel", type=Path, required=True)
    parser.add_argument("--control-component", choices=["methods"], default="methods")
    args = parser.parse_args()
    run(args.output, args.backends.split(","), args.repeats, args.seed, args.control_wheel)
