"""Reproducible compiler/Vulkan/CUDA comparisons; no CUDA runtime backend.

Run: python -m benchmarks.execution --output .artifacts/benchmarks/execution/raw.json
Optional packages and native shader/BLAS libraries must already be installed.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import random
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import openpacho as pacho
from benchmarks.cuda_references import availability_errors, available_references
from benchmarks.numerical_metrics import precision_metadata, record_error, solver_metrics
from benchmarks.suite import Case, operation
from benchmarks.suite import make_inputs as solver_inputs


def matmul(a, b):
    return a @ b


def energy(a, b, c):
    x = a * b + c
    y = 2 * x
    return np.sum(y * y, axis=-1)


def update(x, b, rate):
    return (1 - rate) * x + rate * b


def jacobi_step(a, x, b, diagonal):
    return x + (b - a @ x) / diagonal


def gradient_step(a, x, b, rate):
    return x - rate * (a @ x - b)


def make_inputs(case, size, dtype):
    if case.endswith("_solve"):
        return solver_inputs(Case(case.removesuffix("_solve"), (size, size), dtype), 47)[0]
    rng = np.random.default_rng(47)
    if case == "matmul":
        return tuple(rng.uniform(-1, 1, (size, size)).astype(dtype) for _ in range(2))
    if case == "energy":
        return tuple(rng.uniform(-1, 1, (size, 32)).astype(dtype) for _ in range(3))
    if case == "updates100":
        return tuple(rng.uniform(-1, 1, size).astype(dtype) for _ in range(2))
    # Symmetric, strictly diagonally dominant: both fixed-step iterations converge.
    a = rng.uniform(-0.1, 0.1, (size, size)).astype(dtype)
    a = (a + a.T) * np.asarray(0.5, dtype=dtype)
    a[np.diag_indices(size)] = np.sum(np.abs(a), axis=1) + 1
    b = rng.uniform(-1, 1, size).astype(dtype)
    return a, np.zeros(size, dtype=dtype), b, np.diag(a).copy()


class Runner:
    def __init__(self, name, case, inputs, engine=None, reference=None, compiled=False):
        self.name, self.case, self.inputs = name, case, inputs
        self.engine, self.reference = engine, reference
        self.plan = None
        self.fn = {
            "matmul": matmul,
            "energy": energy,
            "updates100": update,
            "jacobi20": jacobi_step,
            "gradient20": gradient_step,
        }.get(case)
        if compiled:
            self.fn = pacho.compile(self.fn)
        self.cp = reference.cp if reference else None
        self.rate = float(0.5 / np.max(inputs[3])) if case == "gradient20" else 0.01

    def sync(self):
        if self.engine:
            self.engine.synchronize()
        elif self.reference:
            self.reference.sync()

    def upload(self):
        if self.engine:
            return tuple(pacho.asarray(a, backend=self.engine) for a in self.inputs)
        if self.reference:
            if self.case == "matmul":
                return self.reference.upload(self.inputs)
            return tuple(self.cp.array(a, copy=True) for a in self.inputs)
        return self.inputs

    def host(self, result):
        if self.engine:
            return pacho.asnumpy(result)
        if self.reference:
            return self.reference.host(result)
        return result

    def setup(self, resident):
        if self.reference and self.case == "matmul":
            self.plan = self.reference.setup(resident)

    def compute(self, resident):
        if self.case.endswith("_solve"):
            xp = pacho if self.engine else (self.cp if self.cp else np)
            case = Case(
                self.case.removesuffix("_solve"), self.inputs[0].shape, str(self.inputs[0].dtype)
            )
            result, self.iterations = operation(case, xp, resident, diagnostics=True)
            return result
        if self.case == "matmul":
            if self.reference:
                return self.reference.compute(resident, plan=self.plan)
            return self.fn(*resident)
        if self.case == "energy":
            if self.cp:
                a, b, c = resident
                y = 2 * (a * b + c)
                return self.cp.sum(y * y, axis=-1)
            return self.fn(*resident)
        if self.case == "updates100":
            x, b = resident
            for _ in range(100):
                x = self.fn(x, b, self.rate)
            return x
        a, x, b, diagonal = resident
        for _ in range(20):
            x = self.fn(a, x, b, diagonal if self.case == "jacobi20" else self.rate)
        return x

    def end_to_end(self):
        result = self.compute(self.upload())
        host = self.host(result)
        self.sync()
        return host


def measure(runner, fn):
    runner.sync()
    start = time.perf_counter()
    result = fn()
    runner.sync()
    return time.perf_counter() - start, result


def check(actual, expected, dtype):
    tolerance = 8e-4 if dtype == "float32" else 2e-10
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(f"Metadata mismatch: {actual.dtype}/{actual.shape}")
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    return float(np.max(np.abs(actual - expected)))


def run(output, repeats=7, quick=False, solvers_only=False):
    errors, rows, engines = {}, [], {}
    for name, options in (("opencl", {}), ("opencl-clblast", {"blas": "clblast"}), ("vulkan", {})):
        try:
            engines[name] = pacho.get_backend("vulkan" if name == "vulkan" else "opencl", **options)
        except Exception as exc:  # noqa: BLE001 - preserve failed measurements in the report
            errors[name] = f"{type(exc).__name__}: {exc}"
    references = available_references()
    errors.update(availability_errors())
    cases = [
        ("matmul", n, dt)
        for dt, ns in (
            ("float32", [64, 512] if quick else [64, 512, 1024, 2048]),
            ("float64", [64] if quick else [64, 1024]),
        )
        for n in ns
    ]
    cases += [("energy", n, "float32") for n in ([8, 2048] if quick else [8, 2048, 32768])]
    cases += [("updates100", n, "float32") for n in ([256] if quick else [256, 65536, 1048576])]
    cases += [
        (case, n, "float32")
        for case in ("jacobi20", "gradient20")
        for n in ([64] if quick else [512, 2048])
    ]
    if solvers_only:
        cases = [
            (case, n, dtype)
            for case in ("jacobi_solve", "sd_solve", "cg_solve")
            for n, dtype in (
                [(64, "float32")]
                if quick
                else [(512, "float32"), (2048, "float32"), (1024, "float64")]
            )
        ]
    report = {
        "schema_version": 2,
        "numerical_precision": precision_metadata(),
        "diagnostics_timed": False,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "seed": 47,
        "repeats": repeats,
        "tf32": False,
        "threads": threadpool_info(),
        "unavailable": errors,
        "versions": {},
        "rows": rows,
    }
    for package in ("numpy", "pyopencl", "vulkan", "cupy-cuda13x", "nvmath-python"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report["devices"] = {name: engine.device for name, engine in engines.items()}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    for case, size, dtype in cases:
        inputs = make_inputs(case, size, dtype)
        with threadpool_limits(limits=1):
            expected = Runner("reference", case, inputs).compute(inputs)
            if case.endswith("_solve"):
                diagnostic_reference = solver_inputs(
                    Case(case.removesuffix("_solve"), (size, size), dtype), 47
                )[1].astype(np.float64)
                reference_label = "known-generating-solution; RHS rounded to input dtype"
            else:
                diagnostic_inputs = tuple(a.astype(np.float64) for a in inputs)
                diagnostic_runner = Runner("reference", case, diagnostic_inputs)
                diagnostic_runner.rate = Runner("original", case, inputs).rate
                diagnostic_reference = diagnostic_runner.compute(diagnostic_inputs)
                reference_label = "numpy-float64"

        def diagnose(
            row,
            host,
            diagnostic_reference=diagnostic_reference,
            reference_label=reference_label,
            case=case,
            inputs=inputs,
        ):
            record_error(row, host, diagnostic_reference, reference=reference_label)
            if case.endswith("_solve"):
                row["solver_error"] = solver_metrics(
                    *inputs,
                    host,
                    reference_solution=diagnostic_reference,
                    reference=reference_label,
                )

        specs = [(f"numpy-{t}t", None, None, False, t) for t in (1, 8, 16)]
        specs += [
            (
                name + ("-solver" if solvers_only else ("-compiled" if compiled else "-eager")),
                engine,
                None,
                compiled,
                1,
            )
            for name, engine in engines.items()
            for compiled in ((False,) if solvers_only else (False, True))
        ]
        specs += [
            (ref.name, None, ref, False, 1)
            for ref in references
            if case == "matmul" or ref.name == "cupy"
        ]
        active = []
        for name, engine, ref, compiled, threads in specs:
            runner = Runner(name, case, inputs, engine, ref, compiled)
            row = {
                "case": case,
                "size": size,
                "dtype": dtype,
                "implementation": name,
                "resident_seconds": [],
                "end_to_end_seconds": [],
                "threads": threads,
            }
            rows.append(row)
            try:
                with threadpool_limits(limits=threads):
                    row["upload_seconds"], resident = measure(runner, runner.upload)
                    row["planning_seconds"], _ = measure(runner, partial(runner.setup, resident))
                    row["first_execution_seconds"], first = measure(
                        runner, partial(runner.compute, resident)
                    )
                    row["download_seconds"], host = measure(runner, partial(runner.host, first))
                    diagnose(row, host)
                    row["max_abs_error"] = check(host, expected, dtype)
                    runner.compute(resident)
                    runner.sync()
                    check(runner.end_to_end(), expected, dtype)
                    if solvers_only:
                        row["iterations"] = runner.iterations
                if compiled:
                    args = (
                        resident
                        if case in ("matmul", "energy")
                        else (
                            (*resident, runner.rate)
                            if case == "updates100"
                            else (*resident[:3], resident[3] if case == "jacobi20" else runner.rate)
                        )
                    )
                    row["plan"] = runner.fn.explain(*args)
                active.append((runner, resident, row, threads))
            except Exception as exc:  # noqa: BLE001 - preserve failed measurements in the report
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED {case}/{size}/{dtype}/{name}: {row['error']}", flush=True)
        order = random.Random(47)
        for _ in range(repeats):
            order.shuffle(active)
            for runner, resident, row, threads in active:
                try:
                    with threadpool_limits(limits=threads):
                        elapsed, actual = measure(runner, partial(runner.compute, resident))
                        row["resident_seconds"].append(elapsed)
                        resident_host = runner.host(actual)
                        diagnose(row, resident_host)
                        check(resident_host, expected, dtype)
                        elapsed, host = measure(runner, runner.end_to_end)
                        row["end_to_end_seconds"].append(elapsed)
                        diagnose(row, host)
                        check(host, expected, dtype)
                        if solvers_only:
                            row.setdefault("iterations_per_round", []).append(runner.iterations)
                            a, b = inputs
                            residual = float(np.linalg.norm(b - a @ host))
                            tolerance = (1e-4 if dtype == "float32" else 1e-7) * float(
                                np.linalg.norm(b)
                            )
                            if residual > tolerance * 1.02:
                                raise AssertionError(f"Residual {residual} exceeds {tolerance}")
                            row["residual_norm"] = residual
                except Exception as exc:  # noqa: BLE001 - preserve failed measurements in the report
                    row["error"] = f"{type(exc).__name__}: {exc}"
        for _, _, row, _ in active:
            for mode in ("resident", "end_to_end"):
                samples = row[f"{mode}_seconds"]
                row[f"{mode}_median_seconds"] = statistics.median(samples) if samples else None
        save()
        print(
            f"Measured {case} size={size} {dtype}: {len(active)}/{len(specs)} implementations",
            flush=True,
        )
    for ref in references:
        ref.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--solvers-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    report = run(args.output, args.repeats, args.quick, args.solvers_only)
    raise SystemExit(1 if any("error" in row for row in report["rows"]) else 0)
