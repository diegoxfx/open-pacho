"""Bounded application workloads for NumPy-compatible array backends.

This module measures two deliberately small, reproducible workloads:

* ``covariance`` computes ``center = X - mean(X, axis=0, keepdims=True)`` and
  returns ``center.T @ center / (n - 1)``.
* ``ridge`` prepares ``Xt = X.T.copy()`` once, starts ``w`` at zero, and takes
  exactly twenty updates of ``w = w - rate * (Xt @ (X @ w - y) + lambda * w)``.

Inputs use seed 47.  ``X`` is standard normal data scaled by ``1 / sqrt(d)``;
ridge ``y`` is ``X @ known_w + noise`` generated from the same deterministic
stream.  The fixed step size is ``rate = 0.25 / n`` and the fixed ridge
regularization is ``0.1``.

The public workload functions operate on NumPy, CuPy, or OpenPacho arrays.  The
runner compares NumPy with one and eight BLAS threads, OpenPacho OpenCL custom
and CLBlast backends, Vulkan, optional native CUDA custom/cuBLAS, and CuPy.
OpenPacho rows are emitted in
both eager and ``p.compile`` modes; CuPy is an eager control.  Each row records
warm synchronized resident/calculation samples, upload, preparation (including
the explicit ridge transpose), calculation, download, end-to-end samples, and a
separate first-call sample that includes compilation when applicable.

The command line writes one JSON report and preserves unavailable backends and
per-row exceptions instead of dropping them::

    python -m benchmarks.applications --quick --backends "" \
        --output .artifacts/benchmarks/applications-quick.json
    python -m benchmarks.applications --repeats 7 \
        --output .artifacts/benchmarks/applications-full.json

Use ``--backends opencl,opencl-clblast,vulkan,cupy`` for the default GPU set;
append ``cuda-cublas,cuda-custom`` for native CUDA. Compiled CUDA plans fuse
supported elementwise/map-reduce groups; matmul and BLAS remain separate, while
bool accumulation and graphs are unsupported. ``--quick`` selects
``n=256, d=32, float32``;
the full run uses ``(4096, 128)`` and ``(16384, 256)`` for both float32 and float64.
Diagnostics are outside all timing clocks and use ``benchmarks.numerical_metrics``
against a NumPy float64 reference.  Float64 rows are labelled ``sameprecision``
because that reference has the same precision.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import openpacho as pacho
from benchmarks.numerical_metrics import error_metrics, precision_metadata, record_error

RIDGE_STEPS = 20
RIDGE_REGULARIZATION = 0.1
DEFAULT_SEED = 47
DEFAULT_REPEATS = 7
DEFAULT_CPU_THREADS = (1, 8)
DEFAULT_BACKENDS = ("opencl", "opencl-clblast", "vulkan", "cupy")
OPTIONAL_BACKENDS = (*DEFAULT_BACKENDS, "cuda-cublas", "cuda-custom")


@dataclass(frozen=True)
class ApplicationCase:
    """One bounded application benchmark case."""

    workload: str
    n: int
    d: int
    dtype: str

    @property
    def id(self) -> str:
        return f"{self.workload}-n{self.n}-d{self.d}-{self.dtype}"


# ``Case`` is a convenient discovery alias matching the other benchmark
# modules while keeping the more descriptive public dataclass name available.
Case = ApplicationCase

QUICK_CASES = (
    ApplicationCase("covariance", 256, 32, "float32"),
    ApplicationCase("ridge", 256, 32, "float32"),
)
FULL_CASES = tuple(
    ApplicationCase(workload, n, d, dtype)
    for workload in ("covariance", "ridge")
    for n, d in ((4096, 128), (16384, 256))
    for dtype in ("float32", "float64")
)
IRREGULAR_CASES = tuple(
    ApplicationCase(workload, n, d, dtype)
    for workload in ("covariance", "ridge")
    for n, d in ((4095, 127), (4097, 129))
    for dtype in ("float32", "float64")
)


def benchmark_cases(quick: bool = False) -> tuple[ApplicationCase, ...]:
    """Return the fixed quick or full case set in deterministic order."""

    return QUICK_CASES if quick else FULL_CASES


def sample_covariance(X: Any) -> Any:
    """Compute the sample covariance with the exact benchmark expression."""

    center = X - X.mean(axis=0, keepdims=True)
    return center.T @ center / (X.shape[0] - 1)


# Short aliases make the workload functions easy to discover in an interactive
# session and preserve the operation names used in report rows.
covariance = sample_covariance


def ridge_gradient_step(X: Any, Xt: Any, y: Any, w: Any, rate: float, regularization: float) -> Any:
    """Take one fixed ridge-gradient step using already prepared ``Xt``."""

    return w - rate * (Xt @ (X @ w - y) + regularization * w)


def ridge_step(X: Any, Xt: Any, y: Any, w: Any, rate: float, regularization: float) -> Any:
    """Alias for :func:`ridge_gradient_step`."""

    return ridge_gradient_step(X, Xt, y, w, rate, regularization)


def _infer_namespace(value: Any, explicit: Any | None = None) -> Any:
    if explicit is not None:
        return explicit
    if isinstance(value, pacho.ndarray):
        return pacho
    module = type(value).__module__
    if module.startswith("cupy"):
        import cupy as cp

        return cp
    return np


def _zeros_for(X: Any, xp: Any, d: int) -> Any:
    if xp is pacho:
        return pacho.zeros(d, dtype=X.dtype, backend=X._backend)
    return xp.zeros(d, dtype=X.dtype)


def prepare_ridge(X: Any, xp: Any | None = None) -> tuple[Any, Any]:
    """Materialize ``X.T.copy()`` once and allocate the zero starting vector."""

    xp = _infer_namespace(X, xp)
    return X.T.copy(), _zeros_for(X, xp, X.shape[1])


def ridge_regression(
    X: Any,
    y: Any,
    *,
    xp: Any | None = None,
    rate: float | None = None,
    regularization: float = RIDGE_REGULARIZATION,
    steps: int = RIDGE_STEPS,
    step: Callable[..., Any] = ridge_gradient_step,
) -> Any:
    """Run the fixed twenty-step ridge workload on a NumPy-like namespace.

    ``steps`` is accepted to make the helper convenient for callers inspecting
    the operation, but the benchmark workload is intentionally fixed at twenty
    and rejects another value.  ``step`` can be a backend-specific compiled
    function with the same six-argument signature.
    """

    if steps != RIDGE_STEPS:
        raise ValueError(f"ridge benchmark requires exactly {RIDGE_STEPS} steps")
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("ridge inputs must have shapes (n, d) and (n,)")
    xp = _infer_namespace(X, xp)
    rate = 0.25 / X.shape[0] if rate is None else rate
    Xt, w = prepare_ridge(X, xp=xp)
    for _ in range(RIDGE_STEPS):
        w = step(X, Xt, y, w, rate, regularization)
    return w


def ridge_objective(X: Any, y: Any, w: Any, regularization: float = RIDGE_REGULARIZATION) -> Any:
    """Return ``0.5*||Xw-y||² + 0.5*lambda*||w||²``."""

    residual = X @ w - y
    return 0.5 * (residual @ residual) + 0.5 * regularization * (w @ w)


def ridge_gradient(X: Any, y: Any, w: Any, regularization: float = RIDGE_REGULARIZATION) -> Any:
    """Return the gradient used by :func:`ridge_gradient_step`."""

    return X.T @ (X @ w - y) + regularization * w


def make_covariance_inputs(
    n: int, d: int, dtype: str | np.dtype = "float32", seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Create deterministic covariance input with the prescribed scaling."""

    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n, d)).astype(dtype)
    scale = np.asarray(1.0 / np.sqrt(d), dtype=dtype)
    return (values * scale).astype(dtype, copy=False)


def make_ridge_inputs(
    n: int,
    d: int,
    dtype: str | np.dtype = "float32",
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic ``X``, ``y`` and the hidden generating ``known_w``."""

    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(dtype)
    X *= np.asarray(1.0 / np.sqrt(d), dtype=dtype)
    known_w = rng.standard_normal(d).astype(dtype)
    noise = (rng.standard_normal(n) * 0.01).astype(dtype)
    y = (X @ known_w + noise).astype(dtype, copy=False)
    return X, y, known_w


def make_inputs(case: ApplicationCase, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, ...]:
    """Create the host inputs for one :class:`ApplicationCase`."""

    if case.workload == "covariance":
        return (make_covariance_inputs(case.n, case.d, case.dtype, seed),)
    if case.workload == "ridge":
        X, y, known_w = make_ridge_inputs(case.n, case.d, case.dtype, seed)
        return X, y, known_w
    raise ValueError(f"unknown workload {case.workload!r}")


def _runtime_inputs(
    case: ApplicationCase, inputs: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, ...]:
    return inputs if case.workload == "covariance" else inputs[:2]


def _rate(case: ApplicationCase) -> float:
    return 0.25 / case.n


def _reference(
    case: ApplicationCase, inputs: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-input-dtype output and the float64 diagnostic reference."""

    runtime = _runtime_inputs(case, inputs)
    if case.workload == "covariance":
        expected = np.asarray(sample_covariance(runtime[0]))
        diagnostic = np.asarray(sample_covariance(runtime[0].astype(np.float64)))
    else:
        expected = np.asarray(ridge_regression(runtime[0], runtime[1]))
        diagnostic = np.asarray(
            ridge_regression(runtime[0].astype(np.float64), runtime[1].astype(np.float64))
        )
    return expected, diagnostic


def _stats(samples: Iterable[float]) -> dict[str, Any]:
    values = [float(sample) for sample in samples]
    return {
        "samples_seconds": values,
        "median_seconds": float(statistics.median(values)) if values else None,
        "p10_seconds": float(np.percentile(values, 10)) if values else None,
        "p90_seconds": float(np.percentile(values, 90)) if values else None,
        "min_seconds": min(values) if values else None,
        "max_seconds": max(values) if values else None,
    }


def _measure(adapter: Adapter, function: Callable[[], Any]) -> tuple[float, Any]:
    adapter.sync()
    started = time.perf_counter_ns()
    value = function()
    adapter.sync()
    return (time.perf_counter_ns() - started) * 1e-9, value


class Adapter:
    """Small common surface for CPU, OpenPacho, and CuPy execution."""

    name: str
    xp: Any
    threads: int = 1
    pacho_backend: Any | None = None
    is_pacho: bool = False

    def upload(self, values: Sequence[np.ndarray]) -> tuple[Any, ...]:
        raise NotImplementedError

    def host(self, value: Any) -> np.ndarray:
        raise NotImplementedError

    def sync(self) -> None:
        raise NotImplementedError


class NumpyAdapter(Adapter):
    def __init__(self, threads: int):
        self.name = f"numpy-{threads}t"
        self.threads = threads
        self.xp = np

    def upload(self, values: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
        return tuple(values)

    def host(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def sync(self) -> None:
        return None


class PachoAdapter(Adapter):
    def __init__(self, name: str, engine: Any):
        self.name = name
        self.pacho_backend = engine
        self.is_pacho = True
        self.xp = pacho

    def upload(self, values: Sequence[np.ndarray]) -> tuple[Any, ...]:
        return tuple(pacho.asarray(value, backend=self.pacho_backend) for value in values)

    def host(self, value: Any) -> np.ndarray:
        return pacho.asnumpy(value)

    def sync(self) -> None:
        self.pacho_backend.synchronize()


class CuPyAdapter(Adapter):
    def __init__(self, cp: Any):
        self.name = "cupy"
        self.xp = cp
        cp.cuda.Device(0).use()
        cp.cuda.get_current_stream().synchronize()

    def upload(self, values: Sequence[np.ndarray]) -> tuple[Any, ...]:
        return tuple(self.xp.asarray(value) for value in values)

    def host(self, value: Any) -> np.ndarray:
        return self.xp.asnumpy(value)

    def sync(self) -> None:
        self.xp.cuda.get_current_stream().synchronize()


@dataclass
class Implementation:
    """One backend execution mode used for every case."""

    adapter: Adapter
    label: str
    compiled: bool = False

    def __post_init__(self) -> None:
        if self.label.startswith("numpy-"):
            self.name = self.label
        elif self.label == "cupy":
            self.name = "cupy"
        else:
            self.name = f"{self.label}-compiled" if self.compiled else f"{self.label}-eager"
        if self.compiled and self.adapter.is_pacho:
            self.covariance_function = pacho.compile(sample_covariance)
            self.ridge_step_function = pacho.compile(ridge_gradient_step)
        else:
            self.covariance_function = sample_covariance
            self.ridge_step_function = ridge_gradient_step

    @property
    def threads(self) -> int:
        return self.adapter.threads

    def upload(self, values: Sequence[np.ndarray]) -> tuple[Any, ...]:
        return self.adapter.upload(values)

    def host(self, value: Any) -> np.ndarray:
        return self.adapter.host(value)

    def sync(self) -> None:
        self.adapter.sync()

    def prepare(self, case: ApplicationCase, resident: tuple[Any, ...]) -> Any:
        if case.workload == "covariance":
            return resident
        X, y = resident
        Xt, w = prepare_ridge(X, xp=self.adapter.xp)
        return X, Xt, y, w

    def calculate(self, case: ApplicationCase, prepared: Any) -> Any:
        if case.workload == "covariance":
            return self.covariance_function(prepared[0])
        X, Xt, y, w = prepared
        rate = _rate(case)
        for _ in range(RIDGE_STEPS):
            w = self.ridge_step_function(X, Xt, y, w, rate, RIDGE_REGULARIZATION)
        return w

    def close(self) -> None:
        self.sync()


def _backend_failure(name: str, exc: BaseException) -> dict[str, str]:
    return {"backend": name, "type": type(exc).__name__, "message": str(exc)}


def _openpacho_engine(name: str) -> Any:
    if name in ("cuda-cublas", "cuda-custom"):
        return pacho.get_backend("cuda", blas=name.split("-", 1)[1])
    if name == "opencl":
        return pacho.get_backend("opencl", blas="custom")
    if name == "opencl-clblast":
        options: dict[str, Any] = {"blas": "clblast"}
        library = os.environ.get("OPENPACHO_CLBLAST_LIBRARY")
        if library:
            options["clblast_library"] = library
        return pacho.get_backend("opencl", **options)
    if name == "vulkan":
        return pacho.get_backend("vulkan")
    raise ValueError(f"not an OpenPacho backend: {name!r}")


def _build_implementations(
    backend_names: Sequence[str], cpu_threads: Sequence[int]
) -> tuple[list[Implementation], list[dict[str, str]]]:
    implementations = [
        Implementation(NumpyAdapter(threads), f"numpy-{threads}t", False) for threads in cpu_threads
    ]
    failures: list[dict[str, str]] = []
    for backend_name in backend_names:
        try:
            if backend_name == "cupy":
                os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
                import cupy as cp

                implementations.append(Implementation(CuPyAdapter(cp), "cupy", False))
                continue
            engine = _openpacho_engine(backend_name)
            label = {
                "opencl": "opencl-custom",
                "opencl-clblast": "opencl-clblast",
                "vulkan": "vulkan",
                "cuda-cublas": "cuda-cublas",
                "cuda-custom": "cuda-custom",
            }[backend_name]
            adapter = PachoAdapter(label, engine)
            implementations.extend(
                (Implementation(adapter, label, True), Implementation(adapter, label, False))
            )
        except Exception as exc:  # noqa: BLE001 - preserve optional backend failures.
            failures.append(_backend_failure(backend_name, exc))
    return implementations, failures


def _validate(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"dtype mismatch: {actual.dtype} != {expected.dtype}")
    tolerance = 5e-3 if actual.dtype == np.dtype("float32") else 2e-10
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def _ridge_diagnostics(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    X64, y64, w64 = X.astype(np.float64), y.astype(np.float64), w.astype(np.float64)
    objective = ridge_objective(X64, y64, w64)
    gradient = ridge_gradient(X64, y64, w64)
    return float(objective), float(np.linalg.norm(gradient))


def _diagnose(
    row: dict[str, Any],
    case: ApplicationCase,
    inputs: tuple[np.ndarray, ...],
    actual: np.ndarray,
    expected: np.ndarray,
    diagnostic_reference: np.ndarray,
) -> None:
    reference_label = "sameprecision" if case.dtype == "float64" else "numpy-float64"
    record_error(
        row,
        actual,
        diagnostic_reference,
        reference=reference_label,
        precision="float64",
    )
    if case.workload == "ridge":
        X, y = inputs[:2]
        actual_objective, actual_gradient_norm = _ridge_diagnostics(X, y, actual)
        expected_objective, expected_gradient_norm = _ridge_diagnostics(X, y, diagnostic_reference)
        row["diagnostics"] = {
            "final_objective": actual_objective,
            "gradient_norm": actual_gradient_norm,
            "reference_final_objective": expected_objective,
            "reference_gradient_norm": expected_gradient_norm,
            "reference": reference_label,
            "numerical_error": {
                "final_objective": error_metrics(
                    np.asarray(actual_objective),
                    np.asarray(expected_objective),
                    reference=reference_label,
                ),
                "gradient_norm": error_metrics(
                    np.asarray(actual_gradient_norm),
                    np.asarray(expected_gradient_norm),
                    reference=reference_label,
                ),
            },
        }
        # Keep these convenient for consumers that do not want to descend into
        # the diagnostics object.
        row["final_objective"] = actual_objective
        row["gradient_norm"] = actual_gradient_norm


def _empty_timing_row(case: ApplicationCase, implementation: Implementation) -> dict[str, Any]:
    return {
        "case": asdict(case),
        "case_id": case.id,
        "workload": case.workload,
        "n": case.n,
        "d": case.d,
        "dtype": case.dtype,
        "backend": implementation.name,
        "compiled": implementation.compiled,
        "compiler_lowering": "backend-plan" if implementation.compiled else None,
        "mode": "compiled" if implementation.compiled else "eager",
        "threads": implementation.threads,
        "steps": RIDGE_STEPS if case.workload == "ridge" else None,
        "rate": _rate(case) if case.workload == "ridge" else None,
        "regularization": RIDGE_REGULARIZATION if case.workload == "ridge" else None,
        "input_bytes": 0,
        "resident_seconds": [],
        "calculate_seconds": [],
        "upload_seconds": [],
        "preparation_seconds": [],
        "download_seconds": [],
        "end_to_end_seconds": [],
        "first_execution_seconds": None,
        "first_call_includes_compile": bool(
            implementation.compiled and implementation.adapter.is_pacho
        ),
        "first_call_may_compile_device_kernels": bool(
            implementation.adapter.is_pacho or implementation.label == "cupy"
        ),
        "resident_is_calculate": True,
    }


@dataclass
class _CaseState:
    case: ApplicationCase
    inputs: tuple[np.ndarray, ...]
    expected: np.ndarray
    diagnostic_reference: np.ndarray
    implementation: Implementation
    row: dict[str, Any]
    runtime_inputs: tuple[np.ndarray, ...]
    prepared: Any = None
    active: bool = True


def _finish_row(row: dict[str, Any]) -> None:
    for field in (
        "resident_seconds",
        "calculate_seconds",
        "upload_seconds",
        "preparation_seconds",
        "download_seconds",
        "end_to_end_seconds",
    ):
        row[f"{field.removesuffix('_seconds')}_stats"] = _stats(row[field])


def _initialize_state(
    case: ApplicationCase,
    inputs: tuple[np.ndarray, ...],
    expected: np.ndarray,
    diagnostic_reference: np.ndarray,
    implementation: Implementation,
) -> _CaseState:
    row = _empty_timing_row(case, implementation)
    runtime_inputs = _runtime_inputs(case, inputs)
    row["input_bytes"] = sum(value.nbytes for value in runtime_inputs)
    state = _CaseState(
        case,
        inputs,
        expected,
        diagnostic_reference,
        implementation,
        row,
        runtime_inputs,
    )
    with threadpool_limits(limits=implementation.threads, user_api="blas"):
        try:
            upload_seconds, resident = _measure(
                implementation, lambda: implementation.upload(runtime_inputs)
            )
            row["initial_upload_seconds"] = upload_seconds
            preparation_seconds, state.prepared = _measure(
                implementation,
                lambda resident=resident: implementation.prepare(case, resident),
            )
            row["initial_preparation_seconds"] = preparation_seconds
            first_seconds, first_result = _measure(
                implementation,
                lambda: implementation.calculate(case, state.prepared),
            )
            row["first_execution_seconds"] = first_seconds
            download_seconds, first_host = _measure(
                implementation,
                lambda first_result=first_result: implementation.host(first_result),
            )
            row["initial_download_seconds"] = download_seconds
            _validate(first_host, expected)
            _diagnose(row, case, inputs, first_host, expected, diagnostic_reference)
            del resident, first_result, first_host
        except Exception as exc:  # noqa: BLE001 - preserve per-row failures.
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            state.active = False
    return state


def _run_state_round(state: _CaseState) -> None:
    if not state.active:
        return
    case, implementation = state.case, state.implementation
    row = state.row
    with threadpool_limits(limits=implementation.threads, user_api="blas"):
        try:
            calculate_seconds, result = _measure(
                implementation,
                lambda: implementation.calculate(case, state.prepared),
            )
            row["calculate_seconds"].append(calculate_seconds)
            row["resident_seconds"].append(calculate_seconds)
            host_seconds, host_result = _measure(
                implementation,
                lambda result=result: implementation.host(result),
            )
            row["download_seconds"].append(host_seconds)
            _validate(host_result, state.expected)
            _diagnose(
                row,
                case,
                state.inputs,
                host_result,
                state.expected,
                state.diagnostic_reference,
            )

            upload_seconds, fresh_resident = _measure(
                implementation,
                lambda: implementation.upload(state.runtime_inputs),
            )
            row["upload_seconds"].append(upload_seconds)
            preparation_seconds, _ = _measure(
                implementation,
                lambda fresh_resident=fresh_resident: implementation.prepare(case, fresh_resident),
            )
            row["preparation_seconds"].append(preparation_seconds)
            del fresh_resident, result, host_result

            def end_to_end() -> np.ndarray:
                fresh = implementation.upload(state.runtime_inputs)
                ready = implementation.prepare(case, fresh)
                value = implementation.calculate(case, ready)
                return implementation.host(value)

            e2e_seconds, e2e_host = _measure(implementation, end_to_end)
            row["end_to_end_seconds"].append(e2e_seconds)
            _validate(e2e_host, state.expected)
            _diagnose(
                row,
                case,
                state.inputs,
                e2e_host,
                state.expected,
                state.diagnostic_reference,
            )
            del e2e_host
        except Exception as exc:  # noqa: BLE001 - preserve timing-row failures.
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            state.active = False


def _run_case(
    case: ApplicationCase,
    inputs: tuple[np.ndarray, ...],
    expected: np.ndarray,
    diagnostic_reference: np.ndarray,
    implementation: Implementation,
    repeats: int,
) -> dict[str, Any]:
    """Run one implementation; retained as a useful narrow helper for callers."""

    state = _initialize_state(case, inputs, expected, diagnostic_reference, implementation)
    for _ in range(repeats):
        _run_state_round(state)
    _finish_row(state.row)
    return state.row


def _normalize_backends(backends: Sequence[str] | str | None) -> tuple[str, ...]:
    if backends is None:
        return DEFAULT_BACKENDS
    if isinstance(backends, str):
        values = tuple(part.strip().lower() for part in backends.split(",") if part.strip())
    else:
        values = tuple(str(part).strip().lower() for part in backends if str(part).strip())
    unknown = sorted(set(values) - set(OPTIONAL_BACKENDS))
    if unknown:
        raise ValueError(f"unknown backend(s): {', '.join(unknown)}")
    if len(values) != len(set(values)):
        raise ValueError("backends must not contain duplicates")
    return values


def run(
    output: str | Path,
    repeats: int = DEFAULT_REPEATS,
    quick: bool = False,
    backends: Sequence[str] | str | None = None,
    *,
    irregular: bool = False,
    seed: int = DEFAULT_SEED,
    cpu_threads: Sequence[int] = DEFAULT_CPU_THREADS,
) -> dict[str, Any]:
    """Run the bounded application benchmark and write a JSON report."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    cpu_threads = tuple(int(value) for value in cpu_threads)
    if not cpu_threads or min(cpu_threads) < 1:
        raise ValueError("cpu_threads must contain positive values")
    selected_backends = _normalize_backends(backends)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    implementations, failures = _build_implementations(selected_backends, cpu_threads)
    if quick and irregular:
        raise ValueError("quick and irregular select different case sets")
    selected_cases = IRREGULAR_CASES if irregular else benchmark_cases(quick)
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "applications",
        "seed": seed,
        "repeats": repeats,
        "quick": quick,
        "cases": [asdict(case) | {"id": case.id} for case in selected_cases],
        "requested_backends": list(selected_backends),
        "cpu_threads": list(cpu_threads),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {"numpy": np.__version__},
        "threads": threadpool_info(),
        "numerical_precision": precision_metadata(),
        "diagnostics_timed": False,
        "tf32": False,
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE", "unset"),
        "timing_policy": {
            "resident": "warm synchronized calculation with resident backend inputs",
            "upload": "warm synchronized host-to-backend transfer of runtime inputs",
            "preparation": "warm synchronized preparation; ridge includes Xt=X.T.copy() and w=zeros(d)",
            "calculate": "warm synchronized workload calculation; resident_seconds aliases these samples",
            "download": "warm synchronized backend-to-host result transfer",
            "end_to_end": "warm synchronized upload + preparation + calculation + download",
            "first_call": "separate synchronized first calculation; includes p.compile plan capture/build when applicable",
            "diagnostics": "outside all timing clocks",
        },
        "unavailable": failures,
        "rows": [],
    }
    try:
        report["versions"]["open-pacho"] = importlib.metadata.version("open-pacho")
    except importlib.metadata.PackageNotFoundError:
        report["versions"]["open-pacho"] = getattr(pacho, "__version__", "unknown")
    for package in (
        "pyopencl",
        "vulkan",
        "cupy-cuda13x",
        "cuda-bindings",
        "nvidia-cuda-nvrtc",
        "nvidia-cublas",
        "nvmath-python",
    ):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report["cuda_contract"] = (
        {
            "matmul_math": "cuBLAS PEDANTIC_MATH or custom NVRTC with fmad=false",
            "nvrtc_ftz": False,
            "fused_lowering": True,
            "scope": "float32/float64 vectors, matrices and broadcast batches; scaled norms; fused elementwise/map-reduce lowering; matmul/BLAS remain provider boundaries; bool accumulation and graphs remain unsupported",
        }
        if any(name.startswith("cuda-") for name in selected_backends)
        else None
    )
    for case in selected_cases:
        inputs = make_inputs(case, seed)
        with threadpool_limits(limits=1, user_api="blas"):
            expected, diagnostic_reference = _reference(case, inputs)
        states = [
            _initialize_state(case, inputs, expected, diagnostic_reference, implementation)
            for implementation in implementations
        ]
        active_states = [state for state in states if state.active]
        order = random.Random(seed + len(report["rows"]))
        for _ in range(repeats):
            order.shuffle(active_states)
            for state in active_states:
                _run_state_round(state)
            active_states = [state for state in active_states if state.active]
        for state in states:
            _finish_row(state.row)
            state.row["seed"] = seed
            report["rows"].append(state.row)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for implementation in implementations:
        try:
            implementation.close()
        except Exception as exc:  # noqa: BLE001 - preserve cleanup failures as report errors.
            report.setdefault("cleanup_errors", []).append(
                _backend_failure(implementation.name, exc)
            )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _parse_backends(value: str) -> tuple[str, ...]:
    try:
        return _normalize_backends(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--irregular", action="store_true", help="Use odd rectangular 4095x127 and 4097x129 cases"
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--backends",
        type=_parse_backends,
        default=DEFAULT_BACKENDS,
        help="comma-separated optional backends; use an empty string for CPU only",
    )
    parser.add_argument("--cpu-threads", default="1,8")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    try:
        cpu_threads = tuple(int(value) for value in args.cpu_threads.split(",") if value.strip())
        report = run(
            args.output,
            repeats=args.repeats,
            quick=args.quick,
            irregular=args.irregular,
            backends=args.backends,
            cpu_threads=cpu_threads,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    return int(
        bool(report["unavailable"])
        or bool(report.get("cleanup_errors"))
        or any("error" in row for row in report["rows"])
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BACKENDS",
    "DEFAULT_CPU_THREADS",
    "DEFAULT_REPEATS",
    "DEFAULT_SEED",
    "FULL_CASES",
    "QUICK_CASES",
    "RIDGE_REGULARIZATION",
    "RIDGE_STEPS",
    "ApplicationCase",
    "Case",
    "benchmark_cases",
    "build_parser",
    "covariance",
    "main",
    "make_covariance_inputs",
    "make_inputs",
    "make_ridge_inputs",
    "prepare_ridge",
    "ridge_gradient",
    "ridge_gradient_step",
    "ridge_objective",
    "ridge_regression",
    "ridge_step",
    "run",
    "sample_covariance",
]
