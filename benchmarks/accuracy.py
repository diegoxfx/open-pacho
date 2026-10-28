"""Bounded adversarial accuracy suite; Decimal80 references and untimed errors.

Run with --backends numpy,pacho-numpy for CPU-only inspection. CUDA BLAS adapters
cover GEMM only; CuPy covers all operations. Timings are warm resident wall times,
not transferable speed rankings for these intentionally small diagnostic inputs.
Native cuda-cublas/cuda-custom also cover scaled norms and the external solvers.
"""

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.numerical_metrics import number, precision_metadata, record_error, solver_metrics
from examples import solvers


@dataclass
class Case:
    name: str
    operation: str
    inputs: tuple
    method: str = ""


def decimal_array(value):
    value = np.asarray(value)
    return np.array([Decimal.from_float(float(x)) for x in value.flat], dtype=object).reshape(
        value.shape
    )


def decimal_solve(a, b):
    """Partial-pivoted elimination of the actual rounded input system."""
    a, b = a.copy(), b.copy()
    n = len(b)
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(a[row, column]))
        if not a[pivot, column]:
            raise ValueError("Singular Decimal reference system")
        a[[column, pivot]], b[[column, pivot]] = a[[pivot, column]], b[[pivot, column]]
        for row in range(column + 1, n):
            factor = a[row, column] / a[column, column]
            a[row, column:] -= factor * a[column, column:]
            b[row] -= factor * b[column]
    x = np.empty(n, dtype=object)
    for row in reversed(range(n)):
        x[row] = (b[row] - sum(a[row, row + 1 :] * x[row + 1 :], Decimal(0))) / a[row, row]
    return x


def reference(case):
    with localcontext() as context:
        context.prec = 80
        values = tuple(decimal_array(v) for v in case.inputs)
        if case.operation == "norm":
            return np.asarray(sum((x * x for x in values[0].flat), Decimal(0)).sqrt())
        if case.operation == "sum":
            return np.asarray(sum(values[0].flat, Decimal(0)))
        if case.operation == "add":
            return values[0] + values[1]
        if case.operation in ("gemm", "gemv", "dot"):
            return np.asarray(values[0] @ values[1])
        if case.operation == "energy":
            y = Decimal(2) * (values[0] * values[1] + values[2])
            return np.sum(y * y, axis=-1)
        return decimal_solve(*values)


def cases():
    result = []
    for dtype in (np.float32, np.float64):
        rng = np.random.default_rng(4701)
        big = 1e8 if dtype == np.float32 else 1e16
        low, high = (1e-25, 1e25) if dtype == np.float32 else (1e-200, 1e200)
        cancellation = np.tile(np.array([big, 1, -big], dtype=dtype), 23)
        result += [
            Case("cancellation-sum", "sum", (cancellation,)),
            Case("cancellation-dot", "dot", (cancellation, np.ones_like(cancellation))),
            Case("norm-underflow", "norm", (np.full(17, low, dtype=dtype),)),
            Case("norm-overflow", "norm", (np.full(17, high, dtype=dtype),)),
            Case(
                "subnormal-add",
                "add",
                (np.full(17, np.finfo(dtype).smallest_subnormal, dtype=dtype),) * 2,
            ),
            Case(
                "gemm-irregular",
                "gemm",
                tuple(rng.uniform(-1, 1, shape).astype(dtype) for shape in ((27, 31), (31, 25))),
            ),
            Case(
                "gemv-irregular",
                "gemv",
                tuple(rng.uniform(-1, 1, shape).astype(dtype) for shape in ((17, 33), (33,))),
            ),
            Case(
                "gemm-register-borders",
                "gemm",
                tuple(rng.uniform(-1, 1, shape).astype(dtype) for shape in ((65, 47), (47, 67))),
            ),
            Case(
                "gemm-cancellation",
                "gemm",
                (np.tile(cancellation, (32, 1)), np.ones((cancellation.size, 33), dtype=dtype)),
            ),
            Case(
                "energy-fusion",
                "energy",
                tuple(rng.uniform(-1, 1, (17, 31)).astype(dtype) for _ in range(3)),
            ),
        ]
        dense = np.full((12, 12), 0.025, dtype=dtype) + np.eye(12, dtype=dtype)
        exact = rng.uniform(-1, 1, 12).astype(dtype)
        for scale, label in ((1, "ordinary"), (low, "tiny-rhs"), (high, "large-rhs")):
            b = (dense @ exact) * np.asarray(scale, dtype=dtype)
            for method in ("cg", "jacobi", "sd"):
                result.append(Case(f"solver-{label}", "solve", (dense, b), method))
        # SPD Hilbert systems deliberately expose the residual/forward-error gap.
        for n in (4, 8):
            hilbert = np.asarray(
                [[1 / (i + j + 1) for j in range(n)] for i in range(n)], dtype=dtype
            )
            b = hilbert @ np.ones(n, dtype=dtype)
            result.append(Case(f"solver-hilbert-{n}", "solve", (hilbert, b), "cg"))
    return result


class Adapter:
    def __init__(self, name):
        self.name, self.backend, self.reference_adapter = name, None, None
        if name in ("numpy", "pacho-numpy"):
            self.xp = np if name == "numpy" else p
            self.backend = p.get_backend("numpy") if name == "pacho-numpy" else None
        elif name in ("cuda-cublas", "cuda-custom"):
            self.xp = p
            self.backend = p.get_backend("cuda", blas=name.split("-", 1)[1])
        elif name in ("cublas", "nvmath"):
            from benchmarks.cuda_references import CuBLASReference, NVMATHReference

            self.reference_adapter = (CuBLASReference if name == "cublas" else NVMATHReference)()
            self.xp = self.reference_adapter.cp
        else:
            from benchmarks.gaps import Adapter as GPUAdapter

            self.gpu = GPUAdapter(name)
            self.xp, self.backend = self.gpu.xp, self.gpu.engine

    def sync(self):
        if self.backend:
            self.backend.synchronize()
        elif self.xp is not np:
            self.xp.cuda.get_current_stream().synchronize()

    def upload(self, values):
        if self.reference_adapter:
            return self.reference_adapter.upload(values)
        return tuple(
            p.asarray(x, backend=self.backend) if self.backend else self.xp.asarray(x)
            for x in values
        )

    def host(self, value):
        return (
            p.asnumpy(value)
            if self.backend
            else np.asarray(value)
            if self.xp is np
            else self.xp.asnumpy(value)
        )

    def compute(self, case, values, plan=None):
        xp = self.xp
        if self.reference_adapter:
            return self.reference_adapter.compute(values, plan=plan), None
        if case.operation == "solve":
            rtol = 1e-5 if case.inputs[0].dtype == np.float32 else 1e-10
            if self.backend:
                fn = {
                    "cg": solvers.conjugate_gradient,
                    "jacobi": solvers.jacobi,
                    "sd": solvers.steepest_descent,
                }[case.method]
                result = fn(*values, rtol=rtol, maxiter=400)
                return result.x, {
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "reason": result.reason,
                    "reported_residual_norm": number(result.residual_norm),
                    "rtol": rtol,
                    "atol": 0.0,
                    "maxiter": 400,
                }
            from benchmarks.suite import native_solver

            value, iterations = native_solver(xp, case.method, *values, rtol=rtol, diagnostics=True)
            return value, {
                "iterations": iterations,
                "converged": True,
                "rtol": rtol,
                "atol": 0.0,
                "maxiter": 400,
            }
        if case.operation == "norm":
            return xp.linalg.norm(values[0]), None
        if case.operation == "sum":
            return xp.sum(values[0]), None
        if case.operation == "add":
            return values[0] + values[1], None
        if case.operation in ("dot", "gemm", "gemv"):
            return values[0] @ values[1], None
        a, b, c = values
        if self.backend and self.backend.name != "numpy":
            from benchmarks.execution import energy

            if not hasattr(self, "energy"):
                self.energy = p.compile(energy)
            return self.energy(a, b, c), None
        y = 2 * (a * b + c)
        return xp.sum(y * y, axis=-1), None


def run(
    output,
    backends=(
        "numpy",
        "pacho-numpy",
        "opencl",
        "opencl-clblast",
        "vulkan",
        "cupy",
        "cublas",
        "nvmath",
    ),
    repeats=3,
):
    report = {
        "schema_version": 1,
        "platform": platform.platform(),
        "origin": str(Path(p.__file__).resolve()),
        "precision": precision_metadata(),
        "repeats": repeats,
        "seed": 4701,
        "cpu_threads": 1,
        "diagnostics_timed": False,
        "timing_scope": "warm resident; intentionally small accuracy cases",
        "reference": "Decimal80 from exactly represented input floats; no assumption of wider longdouble",
        "reference_scopes": {
            "cublas": "gemm only",
            "nvmath": "gemm only",
            "cuda-cublas": "array operations, scaled norms and external solver examples",
            "cuda-custom": "array operations, scaled norms and external solver examples",
        },
        "diagnostic_threshold_is_correctness_contract": False,
        "numerical_error_aggregation": "worst over timed outputs, checked outside timers",
        "solver_residual_aggregation": "last timed output",
        "backend_errors": {},
        "unsupported_cases": [],
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    adapters = []
    for name in backends:
        try:
            adapters.append(Adapter(name))
        except Exception as exc:  # noqa: BLE001 - all requested failures are visible
            report["backend_errors"][name] = f"{type(exc).__name__}: {exc}"
    with threadpool_limits(limits=1):
        for case in cases():
            expected = reference(case)
            for adapter in adapters:
                if adapter.reference_adapter and case.operation != "gemm":
                    continue
                row = {
                    "case": case.name,
                    "operation": case.operation,
                    "method": case.method,
                    "dtype": str(case.inputs[0].dtype),
                    "shapes": [a.shape for a in case.inputs],
                    "implementation": adapter.name,
                    "resident_seconds": [],
                }
                report["rows"].append(row)
                if case.operation == "solve":
                    row["condition_number_2_float64"] = number(
                        np.linalg.cond(case.inputs[0].astype(float))
                    )
                try:
                    values = adapter.upload(case.inputs)
                    plan = (
                        adapter.reference_adapter.setup(values)
                        if adapter.reference_adapter
                        else None
                    )
                    adapter.compute(case, values, plan)
                    adapter.sync()
                    for _ in range(repeats):
                        adapter.sync()
                        start = time.perf_counter()
                        value, details = adapter.compute(case, values, plan)
                        adapter.sync()
                        row["resident_seconds"].append(time.perf_counter() - start)
                        host = adapter.host(value)
                        record_error(
                            row,
                            host,
                            expected,
                            reference="Decimal exact-input oracle",
                            precision="decimal80",
                        )
                    row["resident_median_seconds"] = statistics.median(row["resident_seconds"])
                    threshold = 2e-4 if row["dtype"] == "float32" else 2e-9
                    row["diagnostic_relative_error_threshold"] = threshold
                    error = row["numerical_error"]
                    row["output_dtype_matches_input"] = host.dtype == case.inputs[0].dtype
                    row["quality_finding"] = (
                        error["status"] != "finite"
                        or float(error["relative_l2_error"]) > threshold
                        or not row["output_dtype_matches_input"]
                    )
                    if details:
                        row["solver"] = details
                        row["solver_error"] = solver_metrics(*case.inputs, host)
                        with localcontext() as context:
                            context.prec = 80
                            a, b, x = (decimal_array(v) for v in (*case.inputs, host))
                            r = b - a @ x
                            residual = sum((v * v for v in r), Decimal(0)).sqrt()
                            bnorm = sum((v * v for v in b), Decimal(0)).sqrt()
                            row["decimal_true_residual_norm"] = number(residual)
                            row["decimal_relative_residual"] = (
                                number(residual / bnorm) if bnorm else 0.0
                            )
                        row["quality_finding"] |= not details["converged"]
                except Exception as exc:  # noqa: BLE001 - keep diagnostics and failing cases
                    row["error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"accuracy {adapter.name} {case.name} {case.method} {row['dtype']}: {row.get('error', 'finding' if row.get('quality_finding') else 'ok')}",
                    flush=True,
                )
            output.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    for adapter in adapters:
        if adapter.reference_adapter:
            adapter.reference_adapter.close()
    write_table(report, output.with_suffix(".md"))
    return report


def write_table(report, output):
    lines = [
        "# Numerical error and timing",
        "",
        "Decimal80 references for the inputs actually represented. Small diagnostic cases; timings do not represent performance at scale. Solver relative error measures forward error against the solution of the rounded system. Exceptions and cases outside the threshold are preserved.",
        "",
        "| Case / method / dtype | Implementation | ms | Max abs. error | Relative L2 error | Decimal relative residual | Backward error | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def fmt(value):
        return (
            "—"
            if value is None
            else f"{value:.3e}"
            if isinstance(value, (float, int))
            else str(value)
        )

    for row in report["rows"]:
        metrics, solver = row.get("numerical_error", {}), row.get("solver_error", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['case']} / {row['method']} / {row['dtype']}",
                    row["implementation"],
                    fmt(row.get("resident_median_seconds", 0) * 1000),
                    fmt(metrics.get("max_abs_error")),
                    fmt(metrics.get("relative_l2_error")),
                    fmt(row.get("decimal_relative_residual")),
                    fmt(solver.get("backward_error")),
                    row.get(
                        "error",
                        "outside diagnostic threshold" if row.get("quality_finding") else "ok",
                    ).replace("|", "/"),
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends", default="numpy,pacho-numpy,opencl,opencl-clblast,vulkan,cupy,cublas,nvmath"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    result = run(args.output, tuple(args.backends.split(",")), args.repeats)
    raise SystemExit(
        bool(result["backend_errors"])
        or any("error" in r or r.get("quality_finding") for r in result["rows"])
    )
