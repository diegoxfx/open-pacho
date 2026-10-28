"""One-backend dense-network benchmark using the public application functions.

Run each backend in a separate process, for example ``python -m
benchmarks.dense_network --backend numpy --profile quick --output report.json``.
Resident samples include fresh output/intermediate allocations and synchronization.
Complete samples upload all inputs, execute a reusable prepared plan where selected,
and download every declared output. Plan preparation is reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import openpacho as p
from benchmarks.applications import (
    CuPyAdapter,
    NumpyAdapter,
    PachoAdapter,
    _measure,
    _openpacho_engine,
    _stats,
)
from benchmarks.numerical_metrics import error_metrics
from examples import dense_network as network

BACKENDS = ("numpy", "opencl", "opencl-clblast", "vulkan", "cuda-custom", "cuda-cublas", "cupy")
SEED, RATE = 47, 0.1
TOLERANCES = {"float32": (2e-4, 2e-6), "float64": (2e-10, 2e-12)}


@dataclass(frozen=True)
class Case:
    workload: str
    n: int
    d: int
    h: int
    o: int
    dtype: str

    @property
    def id(self):
        return f"{self.workload}-{self.n}x{self.d}x{self.h}x{self.o}-{self.dtype}"


QUICK_CASES = tuple(Case(w, 31, 7, 11, 3, dt) for w in ("inference", "sgd") for dt in TOLERANCES)
ALL_CASES = QUICK_CASES + tuple(
    Case(w, *shape, dt)
    for shape in ((4096, 128, 256, 32), (4095, 127, 255, 31))
    for w in ("inference", "sgd")
    for dt in TOLERANCES
)


def make_inputs(case, seed=SEED):
    """Return already rounded X, Y, W1, b1, W2, b2 from one fixed stream."""
    rng = np.random.default_rng(seed)
    shapes = (
        (case.n, case.d),
        (case.n, case.o),
        (case.d, case.h),
        (case.h,),
        (case.h, case.o),
        (case.o,),
    )
    scales = (1 / np.sqrt(case.d), 0.2, 1 / np.sqrt(case.d), 0.05, 1 / np.sqrt(case.h), 0.05)
    return tuple(
        (rng.standard_normal(shape) * scale).astype(case.dtype)
        for shape, scale in zip(shapes, scales)
    )


def runtime_inputs(case, inputs):
    return (inputs[0], *inputs[2:]) if case.workload == "inference" else inputs


def arguments(case, inputs):
    return inputs if case.workload == "inference" else (*inputs, RATE)


def function(case):
    return network.predict if case.workload == "inference" else network.sgd_step


def output_names(case):
    return ("prediction",) if case.workload == "inference" else ("loss", "w1", "b1", "w2", "b2")


def outputs(value):
    return value if isinstance(value, tuple) else (value,)


def reference(case, inputs):
    # Cast precisely the supplied rounded inputs; never regenerate higher precision data.
    values = tuple(a.astype(np.float64) for a in inputs)
    result = outputs(function(case)(*arguments(case, runtime_inputs(case, values))))
    if case.workload == "inference":
        return tuple(map(np.asarray, result)), None, None
    gradients = network.loss_and_gradients(*values)
    prediction = network.predict(values[0], *values[2:])
    return tuple(map(np.asarray, result)), tuple(map(np.asarray, gradients)), prediction


@dataclass
class Implementation:
    adapter: object
    name: str
    compiled: bool = False

    def prepare(self, case, resident):
        selected = function(case)
        return (
            p.compile(selected).prepare(*arguments(case, resident)) if self.compiled else selected
        )

    def calculate(self, plan, case, resident):
        # Every call receives the original parameters, including every SGD repetition.
        return plan(*arguments(case, resident))

    def download(self, value):
        # Snapshot before replay; array plans must not hide overwritten outputs.
        return tuple(np.array(self.adapter.host(a), copy=True) for a in outputs(value))


def implementations(backend):
    if backend == "numpy":
        return [Implementation(NumpyAdapter(t), f"numpy-{t}t") for t in (1, 8)]
    if backend == "cupy":
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        import cupy

        return [Implementation(CuPyAdapter(cupy), "cupy-eager")]
    adapter = PachoAdapter(backend, _openpacho_engine(backend))
    return [
        Implementation(adapter, backend + "-" + mode, mode == "compiled")
        for mode in ("eager", "compiled")
    ]


def balanced_orders(count, repeats, seed):
    rng = random.Random(seed)
    result = []
    for _ in range(repeats // 2):
        order = list(range(count))
        rng.shuffle(order)
        result.extend((order, order[::-1]))
    return result


def _failure(exc, stage):
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def diagnose(row, names, actual, expected, case, stage):
    """Record diagnostics without suppressing subsequent timing on numerical failures."""
    rtol, atol = TOLERANCES[case.dtype]
    label = "sameprecision" if case.dtype == "float64" else "numpy-float64"
    if len(actual) != len(expected):
        row["validation_failures"].append({"stage": stage, "message": "output count mismatch"})
    for name, got, want in zip(names, actual, expected):
        key = f"{stage}:{name}"
        try:
            metric = error_metrics(got, want, reference=label)
            accepted = (
                got.shape == want.shape
                and got.dtype == np.dtype(case.dtype)
                and bool(np.all(np.isfinite(got)))
                and bool(np.allclose(got, want, rtol=rtol, atol=atol))
            )
            metric.update(accepted=accepted, shape=list(got.shape), rtol=rtol, atol=atol)
            row["diagnostics"][key] = metric
            if not accepted:
                row["validation_failures"].append(
                    {"stage": key, "message": "numerical or dtype failure"}
                )
        except Exception as exc:  # noqa: BLE001 - preserve validation evidence.
            row["validation_failures"].append(_failure(exc, key))


def _row(case, impl, inputs):
    digest = hashlib.sha256()
    for value in inputs:
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return {
        "case": asdict(case),
        "case_id": case.id,
        "implementation": impl.name,
        "mode": "compiled" if impl.compiled else "eager",
        "threads": impl.adapter.threads,
        "seed": SEED,
        "rate": RATE if case.workload == "sgd" else None,
        "input_sha256": digest.hexdigest(),
        "input_bytes": sum(a.nbytes for a in inputs),
        "output_bytes": None,
        "peak_memory": "not measured",
        "diagnostics": {},
        "validation_failures": [],
        "execution_complete": False,
        "numerically_accepted": False,
        "timings": {
            key: []
            for key in (
                "initial_upload",
                "prepare",
                "first_call",
                "upload",
                "download",
                "resident",
                "complete",
            )
        },
        "complete_scope": "upload + reusable prepared plan/eager call + all declared output downloads",
        "resident_scope": "allocating computation + synchronization; resident original inputs",
        "upload_scope": "host aliases; no device transfer"
        if isinstance(impl.adapter, NumpyAdapter)
        else "upload all declared inputs",
        "first_call_may_compile_device_kernels": impl.adapter.is_pacho
        or impl.name.startswith("cupy"),
        "declared_outputs": list(output_names(case)),
    }


def _initialize(case, impl, inputs, expected, gradient_ref, prediction_ref, row, warmup_seconds):
    adapter = impl.adapter
    stage = "initial_upload"
    try:
        elapsed, resident = _measure(adapter, lambda: adapter.upload(inputs))
        row["timings"][stage].append(elapsed)
        stage = "prepare"
        elapsed, plan = _measure(adapter, lambda: impl.prepare(case, resident))
        row["timings"][stage].append(elapsed)
        stage = "first_call"
        elapsed, value = _measure(adapter, lambda: impl.calculate(plan, case, resident))
        row["timings"][stage].append(elapsed)
        elapsed, snapshot = _measure(adapter, lambda: impl.download(value))
        row["timings"]["download"].append(elapsed)
        row["output_bytes"] = sum(a.nbytes for a in snapshot)
        diagnose(row, output_names(case), snapshot, expected, case, "first_call")
        if case.workload == "sgd":
            stage = "gradient_diagnostics"
            gradient_function = network.loss_and_gradients
            if impl.compiled:
                gradient_function = p.compile(gradient_function).prepare(*resident)
            gradient_value = gradient_function(*resident)
            adapter.sync()
            diagnose(
                row,
                ("loss", "dw1", "db1", "dw2", "db2"),
                impl.download(gradient_value),
                gradient_ref,
                case,
                stage,
            )
            prediction = network.predict(resident[0], *resident[2:])
            adapter.sync()
            diagnose(
                row,
                ("prediction",),
                impl.download(prediction),
                (prediction_ref,),
                case,
                "prediction",
            )
        stage = "warmup"
        start, count = time.perf_counter(), 0
        while count < 3 or time.perf_counter() - start < warmup_seconds:
            value = impl.calculate(plan, case, resident)
            adapter.sync()
            count += 1
        row["warmup"] = {"calls": count, "seconds": time.perf_counter() - start}
        diagnose(row, output_names(case), impl.download(value), expected, case, "warmup")
        return resident, plan
    except Exception as exc:  # noqa: BLE001 - preserve one failed row and continue.
        row["error"] = _failure(exc, stage)
        return None


def _round(case, impl, inputs, expected, row, state, repeat):
    adapter = impl.adapter
    resident, plan = state
    stage = "resident"
    try:
        elapsed, value = _measure(adapter, lambda: impl.calculate(plan, case, resident))
        row["timings"][stage].append(elapsed)
        diagnose(
            row, output_names(case), impl.download(value), expected, case, f"resident-{repeat}"
        )
        # Standalone upload decomposition is outside the complete sample clock.
        stage = "upload"
        elapsed, uploaded = _measure(adapter, lambda: adapter.upload(inputs))
        row["timings"][stage].append(elapsed)
        del uploaded
        stage = "complete"

        def complete():
            uploaded = adapter.upload(inputs)
            return impl.download(impl.calculate(plan, case, uploaded))

        elapsed, snapshot = _measure(adapter, complete)
        row["timings"][stage].append(elapsed)
        diagnose(row, output_names(case), snapshot, expected, case, f"complete-{repeat}")
        return True
    except Exception as exc:  # noqa: BLE001 - preserve samples preceding runtime failure.
        row["error"] = _failure(exc, stage)
        return False


def _hash_sources(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance(backend, impls):
    package_origin = Path(p.__file__).resolve()
    helper_sources = {
        name: Path(sys.modules[helper.__module__].__file__).resolve()
        for name, helper in (("applications", _measure), ("numerical_metrics", error_metrics))
    }
    versions = {}
    for name in (
        "numpy",
        "open-pacho",
        "threadpoolctl",
        "cupy-cuda12x",
        "cupy-cuda13x",
        "pyopencl",
        "vulkan",
        "cuda-bindings",
        "nvidia-cublas",
        "nvidia-cuda-nvrtc",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": versions,
        "threadpools": threadpool_info(),
        "runner_sha256": _hash_sources([Path(__file__)]),
        "example_sha256": _hash_sources([Path(network.__file__)]),
        "package_origin": str(package_origin),
        "runtime_sha256": _hash_sources(package_origin.parent.rglob("*.py")),
        "measurement_helpers": {
            name: {"source": str(path), "sha256": _hash_sources([path])}
            for name, path in helper_sources.items()
        },
        "source_hash_method": "SHA256 of sorted source files, each basename followed by file bytes",
        "devices": [],
        "peak_memory": "not measured",
    }
    if backend not in ("numpy", "cupy") and impls:
        result["devices"] = [
            asdict(info)
            for info in p.devices("cuda" if backend.startswith("cuda") else backend.split("-")[0])
        ]
    elif backend == "cupy" and impls:
        cp = impls[0].adapter.xp
        result["devices"] = [str(cp.cuda.runtime.getDeviceProperties(0))]
        result["cuda_driver_version"] = cp.cuda.runtime.driverGetVersion()
        result["cuda_runtime_version"] = cp.cuda.runtime.runtimeGetVersion()
    return result


def _close_adapters(impls, report):
    seen = set()
    for impl in impls:
        adapter = impl.adapter
        if id(adapter) in seen:
            continue
        seen.add(id(adapter))
        try:
            adapter.sync()
        except Exception as exc:  # noqa: BLE001 - cleanup must preserve the original failure.
            report.setdefault("cleanup_errors", []).append(_failure(exc, "synchronize"))
        try:
            engine = adapter.pacho_backend
            if engine is not None and hasattr(engine, "close"):
                engine.close()
        except Exception as exc:  # noqa: BLE001 - report optional backend cleanup failures.
            report.setdefault("cleanup_errors", []).append(_failure(exc, "close"))


def run(
    output,
    *,
    backend="numpy",
    profile="all",
    repeats=12,
    cases=None,
    selected_implementations=None,
    warmup_seconds=0.1,
):
    """Write incremental evidence, retaining row failures and closing each backend once."""
    if backend not in BACKENDS or profile not in ("quick", "all"):
        raise ValueError("invalid backend or profile")
    if repeats < 2 or repeats % 2:
        raise ValueError("repeats must be even and at least 2")
    if warmup_seconds < 0.1:
        raise ValueError("warmup_seconds must be at least 0.1")
    selected_cases = tuple(
        cases if cases is not None else QUICK_CASES if profile == "quick" else ALL_CASES
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "backend": backend,
        "profile": profile,
        "seed": SEED,
        "repeats": repeats,
        "rows": [],
        "attempts_complete": False,
        "execution_complete": False,
        "all_numerically_accepted": False,
    }
    impls = []
    # Reserve before backend initialization. An existing report is never overwritten.
    with path.open("x", encoding="utf-8") as stream:

        def save():
            stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.truncate()
            stream.flush()

        save()
        try:
            try:
                impls = (
                    list(selected_implementations)
                    if selected_implementations is not None
                    else implementations(backend)
                )
            except Exception as exc:  # noqa: BLE001 - unavailable backends remain explicit rows.
                report["backend_error"] = _failure(exc, "initialize_backend")
                modes = ["eager", "compiled"] if backend not in ("numpy", "cupy") else ["eager"]
                report["rows"] = [
                    {
                        "case": asdict(case),
                        "case_id": case.id,
                        "mode": mode,
                        "error": report["backend_error"],
                        "execution_complete": False,
                        "numerically_accepted": False,
                    }
                    for case in selected_cases
                    for mode in modes
                ]
            if impls:
                try:
                    report["provenance"] = provenance(backend, impls)
                except Exception as exc:  # noqa: BLE001 - discovery must not suppress workload evidence.
                    report["provenance_error"] = _failure(exc, "provenance")
                for case in selected_cases:
                    inputs = make_inputs(case)
                    with threadpool_limits(limits=1, user_api="blas"):
                        expected, gradient_ref, prediction_ref = reference(case, inputs)
                    inputs = runtime_inputs(case, inputs)
                    rows = [_row(case, impl, inputs) for impl in impls]
                    report["rows"].extend(rows)
                    states = []
                    for impl, row in zip(impls, rows):
                        with threadpool_limits(limits=impl.adapter.threads, user_api="blas"):
                            row["threadpools"] = threadpool_info()
                            states.append(
                                _initialize(
                                    case,
                                    impl,
                                    inputs,
                                    expected,
                                    gradient_ref,
                                    prediction_ref,
                                    row,
                                    warmup_seconds,
                                )
                            )
                        save()
                    orders = balanced_orders(len(impls), repeats, SEED)
                    for row in rows:
                        row["round_order"] = [[impls[i].name for i in order] for order in orders]
                    for repeat, order in enumerate(orders):
                        for index in order:
                            if states[index] is None:
                                continue
                            impl, row = impls[index], rows[index]
                            with threadpool_limits(limits=impl.adapter.threads, user_api="blas"):
                                if not _round(
                                    case, impl, inputs, expected, row, states[index], repeat
                                ):
                                    states[index] = None
                            save()
                    for row in rows:
                        row["execution_complete"] = "error" not in row
                        row["numerically_accepted"] = (
                            row["execution_complete"] and not row["validation_failures"]
                        )
                        row["timings"] = {
                            key: _stats(value) for key, value in row["timings"].items()
                        }
                    states.clear()
                    save()
            report["execution_complete"] = bool(report["rows"]) and all(
                r["execution_complete"] for r in report["rows"]
            )
            report["all_numerically_accepted"] = bool(report["rows"]) and all(
                r["numerically_accepted"] for r in report["rows"]
            )
            report["attempts_complete"] = True
        except BaseException as exc:
            report["fatal_error"] = _failure(exc, "run")
            raise
        finally:
            preserving_exception = sys.exc_info()[0] is not None
            _close_adapters(impls, report)
            try:
                save()
            except Exception:
                if not preserving_exception:
                    raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--profile", choices=("quick", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repeats < 2 or args.repeats % 2:
        parser.error("--repeats must be even and at least 2")
    report = run(args.output, backend=args.backend, profile=args.profile, repeats=args.repeats)
    return int(
        not report["execution_complete"]
        or not report["all_numerically_accepted"]
        or bool(report.get("cleanup_errors"))
    )


if __name__ == "__main__":
    raise SystemExit(main())
