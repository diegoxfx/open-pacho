"""Irregular operations for backend changes, with synchronized wall and error.

Provider order is interleaved. First call, warmup and validation are not in the
resident samples. End-to-end samples are separate upload/compute/download calls.
The same runner can load a historical wheel through PYTHONPATH: package hashes
come from the imported package, not the workspace checkout.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import platform
import random
import time
import types
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics


def cases():
    return [
        (name, shapes, dtype)
        for dtype in ("float32", "float64")
        for name, shapes in (
            ("chain", ((33,), (33,))),
            ("chain", ((1048577,), (1048577,))),
            ("transpose", ((4097, 129),)),
            ("reverse", ((4097, 129),)),
            ("gemv", ((2049, 1023), (1023,))),
            ("gemm", ((257, 129), (129, 513))),
            ("gemm", ((1025, 513), (513, 769))),
            ("gemm", ((512, 256), (256, 768))),
            ("gemm", ((1024, 1024), (1024, 1024))),
            ("batched", ((7, 33, 17), (7, 17, 65))),
            ("broadcast", ((2, 1, 33, 17), (1, 3, 17, 65))),
        )
    ]


def operation(name, arrays):
    a = arrays[0]
    if name == "chain":
        return (a + arrays[1]) * 0.5 - a
    if name == "transpose":
        return a.T.copy()
    if name == "reverse":
        return a[::-1, ::-1].copy()
    return a @ arrays[1]


def engine(name):
    if name.startswith("numpy"):
        return np, None
    if name == "cupy":
        os.environ.setdefault("CUPY_CACHE_DIR", str(Path.cwd() / ".artifacts" / "cupy-cache"))
        import cupy as cp

        return cp, None
    backend_name, _, provider = name.partition("-")
    return p, p.get_backend(backend_name, blas=provider or None)


def sync(xp, backend):
    if backend is not None:
        backend.synchronize()
    elif xp is not np:
        xp.cuda.get_current_stream().synchronize()


def host(value, xp):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def measure(function, xp, backend):
    activate = getattr(function, "_activate_variant", None)
    if activate is not None:
        activate()
    sync(xp, backend)
    start = time.perf_counter_ns()
    value = function()
    returned = time.perf_counter_ns()
    sync(xp, backend)
    stop = time.perf_counter_ns()
    return (stop - start) * 1e-9, (returned - start) * 1e-9, value


def warmup(function, xp, backend, *, seconds=0.25):
    """Identical minimum duration/count after idle, outside sample clocks."""
    start, count = time.monotonic(), 0
    while count < 3 or time.monotonic() - start < seconds:
        _, _, value = measure(function, xp, backend)
        del value
        count += 1
    return {"calls": count, "seconds": time.monotonic() - start}


def run(output, backends, repeats=21, seed=941, control_wheel=None, control_component="methods"):
    if control_component not in ("methods", "gemm"):
        raise ValueError("Unknown control component")
    if control_component == "gemm" and (control_wheel is None or len(backends) != 1):
        raise ValueError("GEMM control requires one backend and a historical wheel")
    restore = []
    try:
        return _run(output, backends, repeats, seed, control_wheel, control_component, restore)
    finally:
        for obj, name, value in reversed(restore):
            setattr(obj, name, value)


def _run(output, backends, repeats, seed, control_wheel, control_component, restore):
    if output.exists() or repeats < 3:
        raise ValueError("Use a new output file and at least three samples")
    output.parent.mkdir(parents=True, exist_ok=True)
    package = importlib.resources.files("openpacho")
    report = {
        "schema": "openpacho.backend-sharing.v1",
        "package_origin": p.__file__,
        "source_sha256": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in package.iterdir()
            if file.name.endswith(".py")
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {},
        "seed": seed,
        "repeats": repeats,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backends": backends,
        "rows": [],
        "unavailable": [],
        "timing_scope": "separate synchronized resident and end-to-end samples; first call/warmup/validation excluded; call time includes Python/API and any internal waits, not CPU utilization",
        "warmup_policy": "at least three calls and 0.25 seconds per resident variant; first-call compilation excluded",
        "tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "control": None
        if control_wheel is None
        else {
            "wheel": str(control_wheel.resolve()),
            "wheel_sha256": hashlib.sha256(control_wheel.read_bytes()).hexdigest(),
            "component": control_component,
            "scope": "previous/current selected component on the same backend/context/inputs; selection outside timers; all other runtime components remain current; GEMM control is process-global and single-threaded",
        },
    }
    for package_name in (
        "numpy",
        "pyopencl",
        "vulkan",
        "cuda-bindings",
        "nvmath-python",
        "cupy-cuda13x",
    ):
        try:
            report["versions"][package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            pass
    engines = {}
    for name in backends:
        try:
            xp, backend = engine(name)
            if control_wheel is None:
                engines[name] = xp, backend, None
            else:
                if backend is None or backend.name not in ("opencl", "cuda"):
                    raise ValueError("method control supports OpenCL and CUDA only")
                module_name = "_opencl" if backend.name == "opencl" else "_cuda"
                if control_component == "gemm":
                    module_name = "_gemm"
                with zipfile.ZipFile(control_wheel) as archive:
                    code = archive.read(f"openpacho/{module_name}.py")
                module = types.ModuleType(f"openpacho._control{module_name}")
                module.__package__ = "openpacho"
                exec(  # noqa: S102 - explicit local benchmark wheel, equivalent to importing its code
                    compile(code, f"{control_wheel}!/{module_name}.py", "exec"), module.__dict__
                )
                if control_component == "gemm":
                    import openpacho._gemm as gemm

                    original = gemm.register_gemm_data
                    restore.append((gemm, "register_gemm_data", original))
                    for variant, function in (
                        ("previous", module.register_gemm_data),
                        ("current", original),
                    ):

                        def activate(function=function):
                            gemm.register_gemm_data = function

                        engines[f"{name}:{variant}"] = xp, backend, activate
                    continue
                old_class = module.OpenCLBackend if backend.name == "opencl" else module.CUDABackend
                methods = ("elementwise", "gather") if backend.name == "opencl" else ("matmul",)
                restore.extend((backend, method, getattr(backend, method)) for method in methods)
                for variant, cls in (("previous", old_class), ("current", type(backend))):

                    def activate(cls=cls, methods=methods, backend=backend):
                        for method in methods:
                            setattr(backend, method, getattr(cls, method).__get__(backend))

                    engines[f"{name}:{variant}"] = xp, backend, activate
        except Exception as exc:  # noqa: BLE001 - record unavailable providers
            report["unavailable"].append({"backend": name, "error": repr(exc)})
    for spec in cases():
        name, shapes, dtype = spec
        inputs = tuple(
            np.random.default_rng(941 + i).uniform(-0.5, 0.5, shape).astype(dtype)
            for i, shape in enumerate(shapes)
        )
        reference = operation(name, tuple(a.astype(np.float64) for a in inputs))
        states = []
        for label, (xp, backend, activate) in engines.items():
            row = {"case": spec, "backend": label}
            try:
                upload = lambda xp=xp, backend=backend, inputs=inputs: tuple(
                    xp.asarray(a, backend=backend) if backend is not None else xp.asarray(a)
                    for a in inputs
                )
                resident = upload()
                function = lambda resident=resident, name=name: operation(name, resident)
                full = lambda upload=upload, xp=xp, name=name: host(operation(name, upload()), xp)
                function._activate_variant = full._activate_variant = activate
                threads = 1 if label == "numpy-1t" else 8
                with threadpool_limits(limits=threads):
                    first, _, value = measure(function, xp, backend)
                    actual = host(value, xp)
                    del value
                    np.testing.assert_allclose(
                        actual,
                        reference,
                        rtol=8e-4 if dtype == "float32" else 2e-10,
                        atol=2e-5 if dtype == "float32" else 1e-10,
                    )
                    row["warmup"] = warmup(function, xp, backend)
                    _, _, value = measure(full, xp, backend)
                    np.testing.assert_array_equal(value, actual)
                    del value
                row.update(
                    {
                        "status": "ok",
                        "first_seconds": first,
                        "error": error_metrics(actual, reference, reference="numpy-float64"),
                        "input_bytes": sum(a.nbytes for a in inputs),
                        "threads": threads,
                        "device": backend.device
                        if backend is not None
                        else "CPU"
                        if xp is np
                        else str(xp.cuda.runtime.getDeviceProperties(0)["name"]),
                        "register_gemm": getattr(backend, "_register_gemm", None),
                        "push_descriptor_limit": getattr(backend, "_push_descriptor_limit", None),
                    }
                )
                states.append((row, xp, backend, function, full, [], [], []))
            except NotImplementedError as exc:
                row.update(status="unsupported", reason=str(exc))
            except Exception as exc:  # noqa: BLE001 - keep all evidence
                row.update(status="error", reason=repr(exc))
            report["rows"].append(row)
        rng = random.Random(seed)
        for _ in range(repeats):
            order = list(states)
            rng.shuffle(order)
            for row, xp, backend, function, full, wall, call, complete in order:
                with threadpool_limits(limits=row["threads"]):
                    elapsed, returned, value = measure(function, xp, backend)
                    wall.append(elapsed)
                    call.append(returned)
                    del value
                    elapsed, _, value = measure(full, xp, backend)
                    complete.append(elapsed)
                    del value
        for row, xp, backend, function, full, wall, call, complete in states:
            row.update(
                wall=_sample_stats(wall),
                call=_sample_stats(call),
                end_to_end=_sample_stats(complete),
            )
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"completed {spec}", flush=True)
    report["all_supported_cases_valid"] = not report["unavailable"] and all(
        r["status"] != "error" for r in report["rows"]
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends",
        default="numpy-1t,numpy-8t,opencl,opencl-clblast,vulkan,cuda-custom,cuda-cublas,cupy",
    )
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--control-wheel", type=Path)
    parser.add_argument("--control-component", choices=("methods", "gemm"), default="methods")
    args = parser.parse_args()
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    report = run(
        args.output,
        args.backends.split(","),
        args.repeats,
        args.seed,
        args.control_wheel,
        args.control_component,
    )
    raise SystemExit(0 if report["all_supported_cases_valid"] else 1)


if __name__ == "__main__":
    main()
