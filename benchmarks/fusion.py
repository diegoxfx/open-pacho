"""CUDA lowering control and portable application references, isolated by process."""

import argparse
import ast
import hashlib
import inspect
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.applications import (
    ApplicationCase,
    CuPyAdapter,
    Implementation,
    NumpyAdapter,
    PachoAdapter,
)
from benchmarks.applications import (
    _reference as reference_outputs,
)
from benchmarks.applications import (
    make_inputs as application_inputs,
)
from benchmarks.backend_sharing import engine, host, measure, sync, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.suite import source_hashes


def chain(a, b):
    return (a + b) * 0.5 - a


def energy(a, b):
    shifted = a + b
    return (shifted * shifted).sum()


def row_energy(a, b):
    shifted = a + b
    return (shifted * shifted).sum(axis=-1)


def row_center(a, b):
    shifted = a + b
    return shifted - shifted.mean(axis=-1, keepdims=True)


def loss(a, w, y):
    residual = a @ w - y
    return (residual * residual).mean()


FUNCTIONS = {
    "chain": chain,
    "energy": energy,
    "row_energy": row_energy,
    "row_center": row_center,
    "loss": loss,
}


def cases():
    return [
        (op, shape, dtype)
        for dtype in ("float32", "float64")
        for op, shape in (
            ("chain", (33,)),
            ("chain", (1048577,)),
            ("energy", (1025, 1023)),
            ("row_energy", (257, 129)),
            ("row_energy", (17, 4099)),
            ("row_center", (2049, 33)),
            ("loss", (2049, 1023)),
            ("covariance", (1025, 65)),
            ("ridge", (2049, 129)),
        )
    ]


def verify_control(path):
    """The old CUDA implementation used the same primitives, without a lowerer."""
    from openpacho._cuda import CUDABackend

    with zipfile.ZipFile(path) as wheel:
        source = wheel.read("openpacho/_cuda.py")
    old = next(
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "CUDABackend"
    )
    current = ast.parse(inspect.getsource(CUDABackend)).body[0]
    old_methods = {
        n.name: ast.dump(n, include_attributes=False)
        for n in old.body
        if isinstance(n, ast.FunctionDef)
    }
    current_methods = {
        n.name: ast.dump(n, include_attributes=False)
        for n in current.body
        if isinstance(n, ast.FunctionDef)
    }
    assert "create_lowering" not in old_methods
    unchanged = (
        "_run",
        "_compile",
        "_activate",
        "empty",
        "matmul",
        "reduce",
        "elementwise",
        "synchronize",
    )
    assert all(old_methods[name] == current_methods[name] for name in unchanged)
    return {
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "unchanged_methods": unchanged,
        "scope": "same backend/device/primitives; baseline prepares a plan with no lowerer as the previous wheel did",
    }


def diagnostic(backend, call, xp):
    if backend is None or backend.name != "cuda":
        return None
    original_run, original_empty = backend._run, backend.empty
    result = {"kernel_launches": 0, "requested_allocation_bytes": 0, "shapes": []}

    def run(source, blocks, *args, **kwargs):
        result["kernel_launches"] += bool(blocks)
        return original_run(source, blocks, *args, **kwargs)

    def empty(shape, dtype):
        result["shapes"].append(shape)
        result["requested_allocation_bytes"] += int(np.prod(shape)) * np.dtype(dtype).itemsize
        return original_empty(shape, dtype)

    backend._run, backend.empty = run, empty
    try:
        _, _, value = measure(call, xp, backend)
        del value
    finally:
        backend._run, backend.empty = original_run, original_empty
    result["scope"] = (
        "separate diagnostic: CUDA custom launches and requested output bytes; not cuBLAS launches, peak VRAM or fresh native allocations"
    )
    return result


def run(output, backends, repeats=21, seed=941, control_wheel=None):
    if output.exists() or len(backends) != 1 or repeats < 3:
        raise ValueError("New output, one isolated backend and >=3 samples required")
    output.parent.mkdir(parents=True, exist_ok=True)
    name = backends[0]
    xp, backend = engine(name)
    is_cuda = backend is not None and backend.name == "cuda"
    if is_cuda and control_wheel is None:
        raise ValueError("CUDA comparison requires a historical wheel")
    factory = getattr(backend, "create_lowering", None)
    adapter = (
        PachoAdapter(name, backend)
        if xp is p
        else NumpyAdapter(int(name[6]))
        if xp is np
        else CuPyAdapter(xp)
    )
    modes = (
        ["primitive-plan", "fused-plan"] if is_cuda else ["compiled-plan"] if xp is p else ["eager"]
    )
    random.Random(seed).shuffle(modes)
    report = {
        "backend": name,
        "device": getattr(backend, "device", name),
        "seed": seed,
        "repeats": repeats,
        "source_sha256": source_hashes(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "control": verify_control(control_wheel) if is_cuda else None,
        "rows": [],
        "scope": "synchronized resident and complete upload/prepare/compute/download; warmup and validation excluded; same-case mode order alternates",
    }
    try:
        with threadpool_limits(limits=adapter.threads):
            for index, (op, shape, dtype) in enumerate(cases()):
                application = op in ("covariance", "ridge")
                rng = np.random.default_rng(seed)
                if application:
                    case = ApplicationCase(op, *shape, dtype)
                    inputs = application_inputs(case, seed=seed)
                    # application_inputs includes the reference coefficient vector;
                    # only X/y are user inputs to ridge, and only X to covariance.
                    inputs = inputs[: 1 if op == "covariance" else 2]
                    expected, reference = reference_outputs(case, inputs)
                else:
                    shapes = (shape, (shape[1],), (shape[0],)) if op == "loss" else (shape, shape)
                    inputs = tuple(rng.uniform(-0.5, 0.5, s).astype(dtype) for s in shapes)
                    expected = FUNCTIONS[op](*inputs)
                    reference = FUNCTIONS[op](*(a.astype(np.float64) for a in inputs))
                resident = adapter.upload(inputs)
                case_modes = (
                    ["eager", "cupy-fuse"]
                    if name == "cupy" and op in ("chain", "energy", "row_energy")
                    else modes
                )
                for mode in case_modes if index % 2 == 0 else list(reversed(case_modes)):
                    if is_cuda:
                        backend.create_lowering = (
                            (lambda: None) if mode == "primitive-plan" else factory
                        )
                    if application:
                        impl = Implementation(adapter, name, compiled=xp is p)
                        prepared = impl.prepare(case, resident)
                        call = lambda impl=impl, case=case, prepared=prepared: impl.calculate(
                            case, prepared
                        )
                        complete = lambda impl=impl, case=case, inputs=inputs: impl.host(
                            impl.calculate(case, impl.prepare(case, impl.upload(inputs)))
                        )
                        plan_function = (
                            impl.covariance_function
                            if op == "covariance"
                            else impl.ridge_step_function
                        )
                    else:
                        function = (
                            p.compile(FUNCTIONS[op])
                            if xp is p
                            else xp.fuse(FUNCTIONS[op])
                            if mode == "cupy-fuse"
                            else FUNCTIONS[op]
                        )
                        call = lambda function=function, resident=resident: function(*resident)
                        complete = lambda function=function, inputs=inputs: adapter.host(
                            function(*adapter.upload(inputs))
                        )
                        plan_function = function
                    sync(xp, backend)
                    start = time.perf_counter_ns()
                    try:
                        value = call()
                    except (NotImplementedError, AttributeError) as exc:
                        if mode != "cupy-fuse":
                            raise
                        report["rows"].append(
                            {
                                "case": (op, shape, dtype),
                                "mode": mode,
                                "status": "unsupported",
                                "reason": str(exc),
                            }
                        )
                        continue
                    sync(xp, backend)
                    first = (time.perf_counter_ns() - start) * 1e-9
                    actual = host(value, xp)
                    del value
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=5e-5 if dtype == "float32" else 1e-11,
                        atol=5e-6 if dtype == "float32" else 1e-12,
                    )
                    row = {
                        "case": (op, shape, dtype),
                        "mode": mode,
                        "first_call_seconds": first,
                        "error": error_metrics(actual, reference),
                        "steps": 20 if op == "ridge" else None,
                    }
                    if xp is p:
                        row["plan"] = [plan.explain() for plan in plan_function._plans.values()]
                    for scope, fn in (("resident", call), ("complete", complete)):
                        warming = warmup(fn, xp, backend)
                        samples = []
                        for _ in range(repeats):
                            seconds, _, value = measure(fn, xp, backend)
                            samples.append(seconds)
                            del value
                        row[scope] = {"warmup": warming, **_sample_stats(samples)}
                    row["diagnostic"] = diagnostic(backend, call, xp)
                    report["rows"].append(row)
                    del call, complete, actual, fn, plan_function
                del resident
        report["complete"] = True
    finally:
        if is_cuda:
            backend.create_lowering = factory
        sync(xp, backend)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="cuda-custom")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--control-wheel", type=Path)
    parser.add_argument("--control-component", choices=["methods"], default="methods")
    args = parser.parse_args()
    run(args.output, args.backends.split(","), args.repeats, args.seed, args.control_wheel)
