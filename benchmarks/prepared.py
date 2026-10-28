"""Amortization of explicit solver preparation, with synchronized paired rounds.

Resident solve excludes input transfers; preparation measures a matrix snapshot
and validation separately. One-shot includes upload, preparation, solve and result
download. No convergence checks or tolerances are changed for a timed variant.
"""

import argparse
import json
import platform
import random
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.execution import make_inputs
from benchmarks.numerical_metrics import precision_metadata, solver_metrics
from benchmarks.suite import Case
from benchmarks.suite import make_inputs as solver_inputs
from examples import solvers


def run(output, repeats=7):
    report = {
        "origin": str(Path(p.__file__).resolve()),
        "numerical_precision": precision_metadata(),
        "diagnostics_timed": False,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_threads": 1,
        "solver_rtol": {"float32": 1e-5, "float64": 1e-8},
        "solver_atol": 0.0,
        "solver_maxiter": 1000,
        "seed": 47,
        "devices": {},
        "repeats": repeats,
        "rows": [],
        "errors": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in ("numpy", "opencl", "opencl-clblast", "vulkan"):
        try:
            backend = (
                p.backends.get_backend("opencl", blas="clblast")
                if name == "opencl-clblast"
                else p.backends.get_backend(name)
            )
        except Exception as exc:  # noqa: BLE001
            report["errors"][name] = str(exc)
            continue
        report["devices"][name] = backend.device
        for size in (512, 2048):
            for dtype in ("float32", "float64"):
                for method, function in (
                    ("cg", solvers.conjugate_gradient),
                    ("jacobi", solvers.jacobi),
                    ("sd", solvers.steepest_descent),
                ):
                    case = {"cg": "cg_solve", "jacobi": "jacobi_solve", "sd": "sd_solve"}[method]
                    host_a, host_b = make_inputs(case, size, np.dtype(dtype))
                    known_solution = solver_inputs(Case(method, (size, size), dtype), 47)[1]
                    A, b = (p.asarray(value, backend=backend) for value in (host_a, host_b))
                    kwargs = {"rtol": 1e-5 if dtype == "float32" else 1e-8, "maxiter": 1000}
                    row = {
                        "backend": name,
                        "size": size,
                        "dtype": dtype,
                        "method": method,
                        "numerical_error_by_mode": {},
                        "samples_seconds": {
                            key: []
                            for key in (
                                "ordinary",
                                "prepared",
                                "preparation",
                                "ordinary_one_shot",
                                "prepared_one_shot",
                            )
                        },
                    }
                    report["rows"].append(row)

                    def measure(fn, backend=backend):
                        backend.synchronize()
                        start = time.perf_counter()
                        value = fn()
                        backend.synchronize()
                        return time.perf_counter() - start, value

                    def check(
                        result,
                        ordinary,
                        mode="prepared",
                        host_a=host_a,
                        host_b=host_b,
                        kwargs=kwargs,
                        dtype=dtype,
                        row=row,
                        known_solution=known_solution,
                    ):
                        value = p.asnumpy(result.x)
                        row["numerical_error_by_mode"][mode] = solver_metrics(
                            host_a,
                            host_b,
                            value,
                            reference_solution=known_solution,
                            reference="known-generating-solution; RHS rounded to input dtype",
                        )
                        assert (
                            result.converged
                            and value.dtype == host_b.dtype
                            and value.shape == host_b.shape
                        )
                        with threadpool_limits(limits=1):
                            residual = np.linalg.norm(
                                host_b.astype(np.float64)
                                - host_a.astype(np.float64) @ value.astype(np.float64)
                            )
                        assert (
                            residual
                            <= kwargs["rtol"] * np.linalg.norm(host_b.astype(np.float64)) * 1.1
                        )
                        assert result.iterations == ordinary.iterations
                        np.testing.assert_allclose(
                            result.residual_history,
                            ordinary.residual_history,
                            rtol=1e-5,
                            atol=1e-6 if dtype == "float32" else 1e-12,
                        )

                    def one_shot(
                        prepared,
                        backend=backend,
                        host_a=host_a,
                        host_b=host_b,
                        method=method,
                        function=function,
                        kwargs=kwargs,
                    ):
                        da, db = (p.asarray(value, backend=backend) for value in (host_a, host_b))
                        result = (
                            solvers.prepare(da, method=method).solve(db, **kwargs)
                            if prepared
                            else function(da, db, **kwargs)
                        )
                        return result, p.asnumpy(result.x)

                    try:
                        with threadpool_limits(limits=1):
                            ordinary = function(A, b, **kwargs)
                            system = solvers.prepare(A, method=method)
                            check(system.solve(b, **kwargs), ordinary)
                            row["iterations"] = ordinary.iterations
                            actions = {
                                "ordinary": partial(function, A, b, **kwargs),
                                "prepared": partial(system.solve, b, **kwargs),
                                "preparation": partial(solvers.prepare, A, method=method),
                                "ordinary_one_shot": partial(one_shot, False),
                                "prepared_one_shot": partial(one_shot, True),
                            }
                            for fn in actions.values():
                                measure(fn)
                            order = list(actions)
                            rng = random.Random(47)
                            for _ in range(repeats):
                                rng.shuffle(order)
                                for mode in order:
                                    elapsed, result = measure(actions[mode])
                                    row["samples_seconds"][mode].append(elapsed)
                                    if mode != "preparation":
                                        check(
                                            result[0] if mode.endswith("one_shot") else result,
                                            ordinary,
                                            mode,
                                        )
                                    del result
                            row["median_seconds"] = {
                                mode: statistics.median(values)
                                for mode, values in row["samples_seconds"].items()
                            }
                            row["validated"] = True
                    except Exception as exc:  # noqa: BLE001 - preserve failed cases
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    print(f"{name} {method} {size} {dtype}: {row.get('error', 'ok')}", flush=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    result = run(args.output, args.repeats)
    raise SystemExit(bool(result["errors"]) or any("error" in row for row in result["rows"]))
