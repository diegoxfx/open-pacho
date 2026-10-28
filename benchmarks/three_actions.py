"""Paired ablations of compiled binding, CUDA context scope and cooperative GEMV."""

import argparse
import ast
import hashlib
import inspect
import json
import types
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.applications import ApplicationCase, Implementation, PachoAdapter
from benchmarks.backend_sharing import engine, measure, warmup
from benchmarks.cuda_dispatch import diagnose
from benchmarks.fusion import FUNCTIONS, application_inputs, cases, reference_outputs
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.suite import source_hashes
from openpacho import _compiler, _cuda


def old_methods(wheel_path, module, cls, allowed):
    """Extract explicit changed methods only, retaining current class identities."""
    with zipfile.ZipFile(wheel_path) as wheel:
        source = wheel.read("openpacho/" + module.__name__.rsplit(".", 1)[-1] + ".py")
    old = next(
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls.__name__
    )
    methods = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
    current = ast.parse(inspect.getsource(cls)).body[0]
    for node in current.body:
        if isinstance(node, ast.FunctionDef) and node.name not in allowed:
            assert ast.dump(node) == ast.dump(methods[node.name]), node.name
    namespace = vars(module).copy()
    tree = ast.Module(body=[methods[name] for name in allowed if name in methods], type_ignores=[])
    exec(compile(tree, str(wheel_path), "exec"), namespace)  # noqa: S102 - explicit local control
    return {name: namespace[name] for name in allowed if name in methods}


class Controls:
    def __init__(self, backend, wheel):
        self.backend = backend
        self.current_plan = _compiler.CompiledFunction._plan
        self.old_plan = old_methods(
            wheel, _compiler, _compiler.CompiledFunction, ("__init__", "_plan")
        )["_plan"]
        self.cuda = backend.name == "cuda"
        self.saved = {}
        if self.cuda:
            self.saved = {
                name: getattr(backend, name) for name in ("_activate", "execution_scope", "matmul")
            }
            old = old_methods(
                wheel, _cuda, _cuda.CUDABackend, ("_activate", "execution_scope", "matmul")
            )
            self.old = {name: types.MethodType(fn, backend) for name, fn in old.items()}
        # Verify all other runtime files match the historical wheel exactly.
        with zipfile.ZipFile(wheel) as archive:
            for name, digest in source_hashes().items():
                if not name.startswith("src/openpacho/") or Path(name).name in (
                    "_cuda.py",
                    "_compiler.py",
                    "_cuda_gemv.py",
                ):
                    continue
                assert (
                    hashlib.sha256(archive.read("openpacho/" + Path(name).name)).hexdigest()
                    == digest
                ), name

    def activate(self, mode):
        if mode not in ("baseline", "binding", "scope", "gemv", "all"):
            raise ValueError(mode)
        _compiler.CompiledFunction._plan = (
            self.current_plan if mode in ("binding", "all") else self.old_plan
        )
        if self.cuda:
            scope = mode in ("scope", "all")
            self.backend._activate = self.saved["_activate"] if scope else self.old["_activate"]
            self.backend.execution_scope = self.saved["execution_scope"] if scope else None
            self.backend.matmul = (
                self.saved["matmul"] if mode in ("gemv", "all") else self.old["matmul"]
            )

    def restore(self):
        _compiler.CompiledFunction._plan = self.current_plan
        for name, value in self.saved.items():
            setattr(self.backend, name, value)


def matvec(a, b):
    return a @ b


def benchmark_cases(name):
    standard = cases()
    if name not in ("cuda-custom", "cuda-cublas"):
        standard = [
            case for case in standard if case[0] in ("chain", "row_energy", "loss", "ridge")
        ]
    if name == "cuda-custom":
        standard += [
            ("matvec", shape, dtype)
            for dtype in ("float32", "float64")
            for shape in (
                (33, 63),
                (33, 64),
                (33, 65),
                (33, 255),
                (33, 256),
                (33, 257),
                (1025, 65),
                (2049, 1023),
                (129, 8193),
                (7, 33, 65),
                (9001,),
                (1048577,),
            )
        ]
    return standard


def build_call(adapter, op, shape, dtype, seed):
    if op in ("ridge", "covariance"):
        case = ApplicationCase(op, *shape, dtype)
        inputs = application_inputs(case, seed=seed)[: 1 if op == "covariance" else 2]
        expected, reference = reference_outputs(case, inputs)
        impl = Implementation(adapter, adapter.name, compiled=True)
        resident = impl.prepare(case, impl.upload(inputs))
        call = lambda: impl.calculate(case, resident)
        complete = lambda: impl.host(impl.calculate(case, impl.prepare(case, impl.upload(inputs))))
    else:
        shapes = (
            (shape, (shape[-1],))
            if op == "matvec"
            else (shape, (shape[1],), (shape[0],))
            if op == "loss"
            else (shape, shape)
        )
        rng = np.random.default_rng(seed)
        inputs = tuple(rng.uniform(-0.5, 0.5, s).astype(dtype) for s in shapes)
        fn = matvec if op == "matvec" else FUNCTIONS[op]
        expected = fn(*inputs)
        reference = fn(*(x.astype(np.float64) for x in inputs))
        function = p.compile(fn)
        resident = adapter.upload(inputs)
        call = lambda: function(*resident)
        complete = lambda: adapter.host(function(*adapter.upload(inputs)))
    return call, complete, expected, reference


def run(output, backends, repeats=31, seed=941, control_wheel=None):
    if output.exists() or len(backends) != 1 or repeats < 3 or control_wheel is None:
        raise ValueError(
            "Use a new output, one backend, >=3 samples and an explicit historical wheel"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    name = backends[0]
    _, backend = engine(name)
    if backend is None:
        raise ValueError("Select an OpenPacho backend")
    control = None
    report = {
        "backend": name,
        "device": backend.device,
        "seed": seed,
        "repeats": repeats,
        "source_sha256": source_hashes(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wheel_sha256": hashlib.sha256(control_wheel.read_bytes()).hexdigest(),
        "rows": [],
        "scope": "paired rotating/reversed ablations; separate per-mode plan, same inputs, no timing of control selection; no numerical-policy changes",
    }
    try:
        control = Controls(backend, control_wheel)
        modes = ["baseline", "binding"]
        if control.cuda:
            modes += ["scope"] + (["gemv"] if name == "cuda-custom" else []) + ["all"]
        adapter = PachoAdapter(name, backend)
        with threadpool_limits(limits=1):
            for index, (op, shape, dtype) in enumerate(benchmark_cases(name)):
                rows, functions, outputs = {}, {}, {}
                order = modes[index % len(modes) :] + modes[: index % len(modes)]
                for mode in order:
                    control.activate(mode)
                    call, complete, expected, reference = build_call(
                        adapter, op, shape, dtype, seed
                    )
                    call._activate_variant = complete._activate_variant = lambda mode=mode: (
                        control.activate(mode)
                    )
                    seconds, _, value = measure(call, p, backend)
                    actual = p.asnumpy(value)
                    del value
                    rtol, atol = (5e-5, 5e-6) if dtype == "float32" else (1e-11, 1e-12)
                    outputs[mode] = actual
                    functions[mode] = (call, complete)
                    rows[mode] = {
                        "case": (op, shape, dtype),
                        "mode": mode,
                        "first_call_seconds": seconds,
                        "error": error_metrics(actual, reference),
                        "validation": {
                            "rtol": rtol,
                            "atol": atol,
                            "numpy_same_dtype_close": bool(
                                np.allclose(actual, expected, rtol=rtol, atol=atol)
                            ),
                            "float64_reference_close": bool(
                                np.allclose(actual, reference, rtol=rtol, atol=atol)
                            ),
                        },
                    }
                for mode in modes:
                    rows[mode]["exact_baseline_equality"] = bool(
                        np.array_equal(outputs[mode], outputs["baseline"])
                    )
                    if mode in ("binding", "scope"):
                        np.testing.assert_array_equal(outputs[mode], outputs["baseline"])
                for scope_index, scope in enumerate(("resident", "complete")):
                    samples = {mode: [] for mode in modes}
                    for mode in order:
                        rows[mode][scope + "_warmup"] = warmup(
                            functions[mode][scope_index], p, backend
                        )
                    for pair in range(repeats):
                        rotated = order[pair % len(order) :] + order[: pair % len(order)]
                        for mode in rotated if pair % 2 else rotated[::-1]:
                            seconds, _, value = measure(functions[mode][scope_index], p, backend)
                            samples[mode].append(seconds)
                            del value
                    for mode in modes:
                        rows[mode][scope] = _sample_stats(samples[mode])
                for mode in modes:
                    if control.cuda and op in ("loss", "ridge"):
                        control.activate(mode)
                        rows[mode]["diagnostic"] = diagnose(backend, functions[mode][0])
                    report["rows"].append(rows[mode])
                print(
                    op,
                    shape,
                    dtype,
                    {
                        mode: round(rows[mode]["resident"]["median_seconds"] * 1e6, 1)
                        for mode in modes
                    },
                    flush=True,
                )
        report["complete"] = True
    finally:
        if control is not None:
            control.restore()
        try:
            close = getattr(backend, "close", backend.synchronize)
            close()
        finally:
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="cuda-custom")
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--control-wheel", type=Path, required=True)
    parser.add_argument("--control-component", choices=["methods"], default="methods")
    args = parser.parse_args()
    run(args.output, args.backends.split(","), args.repeats, args.seed, args.control_wheel)
