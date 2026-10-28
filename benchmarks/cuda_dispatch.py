"""Alternating CUDA parameter-packing control with unchanged kernels and plans."""

import argparse
import ast
import hashlib
import inspect
import json
import time
import types
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.applications import ApplicationCase, Implementation, PachoAdapter
from benchmarks.backend_sharing import measure, warmup
from benchmarks.fusion import FUNCTIONS, application_inputs, cases, reference_outputs
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.suite import source_hashes
from openpacho import _cuda


def previous_run(backend, wheel_path):
    """Load only the old launch method; reuse the current allocation identity."""
    with zipfile.ZipFile(wheel_path) as wheel:
        source = wheel.read("openpacho/_cuda.py")
        for name, digest in source_hashes().items():
            if not name.startswith("src/openpacho/"):
                continue
            name = Path(name).name
            if name in ("_cuda.py", "_cuda_arguments.py"):
                continue
            assert hashlib.sha256(wheel.read("openpacho/" + name)).hexdigest() == digest, name
    old_class = next(
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "CUDABackend"
    )
    current = ast.parse(inspect.getsource(_cuda.CUDABackend)).body[0]
    old = {n.name: n for n in old_class.body if isinstance(n, ast.FunctionDef)}
    for node in current.body:
        if isinstance(node, ast.FunctionDef) and node.name not in ("_run", "_compile"):
            assert ast.dump(node) == ast.dump(old[node.name]), node.name
    namespace = vars(_cuda).copy()
    tree = ast.Module(body=[old["_run"]], type_ignores=[])
    exec(compile(tree, str(wheel_path), "exec"), namespace)  # noqa: S102 - explicit local control
    return types.MethodType(namespace["_run"], backend), {
        "wheel_sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
        "cuda_source_sha256": hashlib.sha256(source).hexdigest(),
        "scope": "only old _run; same current context/cache/primitives/graphs/kernel sources and compile flags",
    }


def diagnose(backend, call):
    """Instrument one warm call separately. Durations include instrumentation."""
    driver, events, saved = backend.driver, [], []
    result = {
        "scope": "separate diagnostic; host times inclusive/not additive; events bracket custom kernels only",
        "calls": {},
    }
    originals = {
        name: getattr(driver, name)
        for name in (
            "cuLaunchKernel",
            "cuCtxPushCurrent",
            "cuCtxPopCurrent",
            "cuStreamSynchronize",
            "cuMemAlloc",
            "cuMemcpyHtoD",
            "cuMemcpyDtoH",
        )
    }

    def track(obj, name, label, function):
        def wrapped(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                row = result["calls"].setdefault(label, {"count": 0, "seconds": 0.0})
                row["count"] += 1
                row["seconds"] += (time.perf_counter_ns() - start) * 1e-9

        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, wrapped)

    def launch(*args, **kwargs):
        before = _cuda._check(driver.cuEventCreate(0))
        after = _cuda._check(driver.cuEventCreate(0))
        events.append((before, after))
        _cuda._check(driver.cuEventRecord(before, backend._stream))
        value = originals["cuLaunchKernel"](*args, **kwargs)
        _cuda._check(driver.cuEventRecord(after, backend._stream))
        return value

    backend.synchronize()
    try:
        for name, function in originals.items():
            track(driver, name, name, launch if name == "cuLaunchKernel" else function)
        for name in ("empty", "_compile", "_run"):
            track(backend, name, name, getattr(backend, name))
        seconds, _, value = measure(call, p, backend)
        del value
        result["instrumented_wall_seconds"] = seconds
        with backend._activate():
            result["custom_kernel_event_seconds"] = [
                _cuda._check(driver.cuEventElapsedTime(a, b)) * 1e-3 for a, b in events
            ]
    finally:
        for obj, name, function in reversed(saved):
            setattr(obj, name, function)
        with backend._activate():
            for pair in events:
                for event in pair:
                    _cuda._check(driver.cuEventDestroy(event))
    return result


def run(output, backends, repeats=31, seed=941, control_wheel=None):
    if len(backends) != 1 or output.exists() or repeats < 3:
        raise ValueError("Use one backend, new output and at least three samples")
    name = backends[0]
    if name not in ("cuda-custom", "cuda-cublas"):
        from benchmarks.fusion import run as reference_run

        return reference_run(output, backends, repeats, seed)
    if control_wheel is None:
        raise ValueError("CUDA control requires a historical wheel")
    output.parent.mkdir(parents=True, exist_ok=True)
    backend = p.get_backend("cuda", blas=name.split("-")[1])
    current = backend._run
    try:
        old, control = previous_run(backend, control_wheel)
    except Exception:
        backend.close()
        raise
    methods = {"numpy-arguments": old, "cached-arguments": current}
    modes = list(methods)
    adapter = PachoAdapter(name, backend)
    report = {
        "backend": name,
        "device": backend.device,
        "seed": seed,
        "repeats": repeats,
        "source_sha256": source_hashes(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "control": control,
        "rows": [],
        "scope": "paired alternating mode order, identical compiled function/inputs/context; resident and complete clocks separate",
    }
    try:
        with threadpool_limits(limits=1):
            for index, (op, shape, dtype) in enumerate(cases()):
                if op in ("ridge", "covariance"):
                    case = ApplicationCase(op, *shape, dtype)
                    inputs = application_inputs(case, seed=seed)[: 1 if op == "covariance" else 2]
                    expected, reference = reference_outputs(case, inputs)
                    impl = Implementation(adapter, name, compiled=True)
                    resident = impl.prepare(case, impl.upload(inputs))
                    call = lambda impl=impl, case=case, resident=resident: impl.calculate(
                        case, resident
                    )
                    complete = lambda impl=impl, case=case, inputs=inputs: impl.host(
                        impl.calculate(case, impl.prepare(case, impl.upload(inputs)))
                    )
                else:
                    rng = np.random.default_rng(seed)
                    shapes = (shape, (shape[1],), (shape[0],)) if op == "loss" else (shape, shape)
                    inputs = tuple(rng.uniform(-0.5, 0.5, s).astype(dtype) for s in shapes)
                    expected = FUNCTIONS[op](*inputs)
                    reference = FUNCTIONS[op](*(x.astype(np.float64) for x in inputs))
                    resident = adapter.upload(inputs)
                    function = p.compile(FUNCTIONS[op])
                    call = lambda function=function, resident=resident: function(*resident)
                    complete = lambda function=function, inputs=inputs: adapter.host(
                        function(*adapter.upload(inputs))
                    )
                rows, outputs = {}, {}
                order = modes if (index + seed) % 2 else modes[::-1]
                for mode in order:
                    backend._run = methods[mode]
                    seconds, _, value = measure(call, p, backend)
                    actual = p.asnumpy(value)
                    del value
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=5e-5 if dtype == "float32" else 1e-11,
                        atol=5e-6 if dtype == "float32" else 1e-12,
                    )
                    outputs[mode] = actual
                    rows[mode] = {
                        "case": (op, shape, dtype),
                        "mode": mode,
                        "first_call_seconds": seconds,
                        "error": error_metrics(actual, reference),
                    }
                np.testing.assert_array_equal(outputs[modes[0]], outputs[modes[1]])
                for scope, fn in (("resident", call), ("complete", complete)):
                    samples = {mode: [] for mode in modes}
                    for mode in order:
                        backend._run = methods[mode]
                        rows[mode][scope + "_warmup"] = warmup(fn, p, backend)
                    for pair in range(repeats):
                        for mode in order if pair % 2 == 0 else order[::-1]:
                            backend._run = methods[mode]
                            seconds, _, value = measure(fn, p, backend)
                            samples[mode].append(seconds)
                            del value
                    for mode in modes:
                        rows[mode][scope] = _sample_stats(samples[mode])
                for mode in order:
                    backend._run = methods[mode]
                    rows[mode]["diagnostic"] = diagnose(backend, call)
                    rows[mode]["exact_control_equality"] = True
                    report["rows"].append(rows[mode])
                print(
                    op,
                    dtype,
                    {
                        mode: round(rows[mode]["resident"]["median_seconds"] * 1e6, 1)
                        for mode in modes
                    },
                    flush=True,
                )
        report["complete"] = True
    finally:
        backend._run = current
        backend.close()
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="cuda-custom")
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--control-wheel", type=Path)
    parser.add_argument("--control-component", choices=["methods"], default="methods")
    args = parser.parse_args()
    run(args.output, args.backends.split(","), args.repeats, args.seed, args.control_wheel)
