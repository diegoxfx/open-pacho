"""Bounded dense coupled-oscillator benchmark, one explicitly selected backend.

Run ``python -m benchmarks.oscillators --backend numpy --profile quick
--output report.json``. A compiled implementation prepares one Verlet step;
the integration loop remains Python. Every integration starts from the original
q/v and computes its initial acceleration inside the sample (65 GEMVs total).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import openpacho as p
from benchmarks import dense_network as shared
from benchmarks.applications import NumpyAdapter, _measure, _stats
from benchmarks.numerical_metrics import error_metrics
from examples import oscillators

BACKENDS = shared.BACKENDS
TOLERANCES = shared.TOLERANCES
SEED, DT, STEPS = 47, 1 / 32, 64
OUTPUT_NAMES = ("q", "v", "a")


@dataclass(frozen=True)
class Case:
    n: int
    dtype: str

    @property
    def id(self):
        return f"oscillators-n{self.n}-{self.dtype}"


QUICK_CASES = tuple(Case(31, dtype) for dtype in TOLERANCES)
ALL_CASES = QUICK_CASES + tuple(Case(n, dtype) for n in (257, 2049) for dtype in TOLERANCES)


def make_inputs(case, seed=SEED, *, certificate=None):
    """Generate dense Gram-plus-identity inputs; verify rounded K in reference.

    The optional certificate records construction evidence and the matrix hash,
    rather than claiming a rigorous floating-point positivity certificate.
    """
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((case.n, case.n))
    original = r.T @ r / case.n + np.eye(case.n)
    original = (original + original.T) * 0.5
    k = original.astype(case.dtype)
    perturbation_bound = float(np.linalg.norm(k.astype(np.float64) - original, ord=np.inf))
    if certificate is not None:
        certificate.update(
            construction="symmetrized float64 R.T @ R / n + I, then dtype cast",
            dtype_cast_perturbation_infinity_norm=perturbation_bound,
            rounded_matrix_sha256=hashlib.sha256(k.tobytes()).hexdigest(),
        )
    q, v = (rng.standard_normal(case.n) / np.sqrt(case.n) for _ in range(2))
    return k, q.astype(case.dtype), v.astype(case.dtype)


def energy(k, q, v):
    """Host float64 diagnostic, outside every integration timing clock."""
    k, q, v = (np.asarray(a, dtype=np.float64) for a in (k, q, v))
    return float(0.5 * (v @ v + q @ (k @ q)))


def relative_energy_change(initial, final):
    if not np.isfinite(initial) or not np.isfinite(final):
        return None
    if initial == 0:
        return 0.0 if final == 0 else None
    return float((final - initial) / initial)


def reference(case, inputs, certificate):
    """Discrete float64 reference and an independent small-case modal solution."""
    k, q, v = (a.astype(np.float64) for a in inputs)
    symmetric = bool(np.array_equal(k, k.T))
    certificate_matches = (
        certificate.get("rounded_matrix_sha256") == hashlib.sha256(inputs[0].tobytes()).hexdigest()
    )
    omega_bound = float(np.sqrt(np.linalg.norm(k, ord=np.inf)))
    stability = {
        **certificate,
        "generation_hash_matches_supplied_matrix": certificate_matches,
        "supplied_matrix_symmetric": symmetric,
        "dt_sqrt_infinity_norm": DT * omega_bound,
        "verlet_stability_bound_satisfied": DT * omega_bound < 2,
    }
    if not all(
        stability[key]
        for key in (
            "generation_hash_matches_supplied_matrix",
            "supplied_matrix_symmetric",
            "verlet_stability_bound_satisfied",
        )
    ):
        raise ValueError("rounded input fails generation hash, symmetry, or Verlet stability guard")
    try:
        np.linalg.cholesky(k)
    except np.linalg.LinAlgError as exc:
        raise ValueError("rounded input fails float64 numerical Cholesky positivity check") from exc
    stability["numerical_cholesky_succeeded"] = True
    stability["positivity_check"] = (
        "float64 numerical Cholesky of exact supplied rounded K; not a rigorous certificate"
    )
    discrete = tuple(np.asarray(a) for a in oscillators.integrate(k, q, v, DT, STEPS))
    initial_energy = energy(k, q, v)
    result = {
        "discrete": discrete,
        "stability": stability,
        "initial_energy": initial_energy,
        "discrete_final_energy": energy(k, *discrete[:2]),
        "continuous": None,
    }
    if case.n <= 257:
        eigenvalues, modes = np.linalg.eigh(k)
        if eigenvalues[0] <= 0:
            raise ValueError("rounded input fails float64 numerical modal positivity check")
        omega, t = np.sqrt(eigenvalues), DT * STEPS
        qmode, vmode = modes.T @ q, modes.T @ v
        cosine, sine = np.cos(omega * t), np.sin(omega * t)
        qc = modes @ (cosine * qmode + sine / omega * vmode)
        vc = modes @ (-omega * sine * qmode + cosine * vmode)
        result["continuous"] = (qc, vc, -(k @ qc))
        stability["modal_minimum_eigenvalue"] = float(eigenvalues[0])
    return result


@dataclass
class Implementation:
    adapter: object
    name: str
    compiled: bool = False

    def prepare(self, case, resident, acceleration_example):
        return (
            p.compile(oscillators.verlet_step).prepare(*resident, acceleration_example, DT)
            if self.compiled
            else oscillators.verlet_step
        )

    def calculate(self, plan, case, resident):
        return oscillators.integrate(*resident, DT, STEPS, step=plan)

    def download(self, value):
        return tuple(np.array(self.adapter.host(a), copy=True) for a in value)


def implementations(backend):
    # Share provider discovery, engine choice and CPU thread controls.
    return [Implementation(i.adapter, i.name, i.compiled) for i in shared.implementations(backend)]


def diagnose(row, actual, expected, case, stage):
    """Backend roundoff is judged against the discrete, rounded-input reference."""
    row["validation_calls"] += 1
    if len(actual) != len(OUTPUT_NAMES):
        row["validation_failures"].append({"stage": stage, "message": "output count mismatch"})
    rtol, atol = TOLERANCES[case.dtype]
    for name, got, want in zip(OUTPUT_NAMES, actual, expected):
        key = f"{stage}:{name}"
        try:
            metric = error_metrics(
                got,
                want,
                reference=("sameprecision" if case.dtype == "float64" else "numpy-float64"),
            )
            accepted = (
                got.shape == want.shape
                and got.dtype == np.dtype(case.dtype)
                and bool(np.all(np.isfinite(got)))
                and bool(np.allclose(got, want, rtol=rtol, atol=atol))
            )
            metric.update(accepted=accepted, shape=list(got.shape), rtol=rtol, atol=atol)
            # Warmup validates every call; keep its latest metrics to bound report size.
            row["diagnostics"][key] = metric
            if not accepted:
                row["validation_failures"].append(
                    {"stage": key, "message": "numerical or dtype failure"}
                )
        except Exception as exc:  # noqa: BLE001 - retain validation evidence, including shape errors.
            row["validation_failures"].append(shared._failure(exc, key))


def _row(case, impl, inputs, ref):
    digest = hashlib.sha256()
    for value in inputs:
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    continuous = ref["continuous"]
    return {
        "case": asdict(case),
        "case_id": case.id,
        "implementation": impl.name,
        "mode": "compiled" if impl.compiled else "eager",
        "threads": impl.adapter.threads,
        "seed": SEED,
        "dt": DT,
        "steps": STEPS,
        "final_time": DT * STEPS,
        "gemvs_per_integration": STEPS + 1,
        "input_sha256": digest.hexdigest(),
        "input_bytes": sum(a.nbytes for a in inputs),
        "input_bytes_by_name": dict(zip(("K", "q", "v"), (a.nbytes for a in inputs))),
        "output_bytes": None,
        "peak_memory": "not measured",
        "stability": ref["stability"],
        "initial_energy": ref["initial_energy"],
        "reference_final_relative_energy_change": relative_energy_change(
            ref["initial_energy"], ref["discrete_final_energy"]
        ),
        "continuous_reference": "float64 modal eigendecomposition of supplied rounded K"
        if continuous is not None
        else "not computed: n > 257; discrete float64 reference only",
        "integration_time_error": {
            name: error_metrics(got, want, reference="continuous-modal-float64")
            for name, got, want in zip(OUTPUT_NAMES, ref["discrete"], continuous)
        }
        if continuous is not None
        else None,
        "diagnostics": {},
        "validation_failures": [],
        "validation_calls": 0,
        "execution_complete": False,
        "numerically_accepted": False,
        "timings": {
            key: []
            for key in (
                "initial_upload",
                "acceleration_example",
                "prepare",
                "first_call",
                "upload",
                "download",
                "resident",
                "complete",
            )
        },
        "complete_scope": "upload K/q/v + integration + download q/v/a + synchronization",
        "resident_scope": "original resident K/q/v; initial acceleration + 64 allocating steps + synchronization",
        "upload_scope": "host aliases; no device transfer"
        if isinstance(impl.adapter, NumpyAdapter)
        else "upload K/q/v",
        "prepare_scope": "one Verlet step; independent acceleration example; excludes integration loop",
        "first_call_may_compile_device_kernels": impl.adapter.is_pacho
        or impl.name.startswith("cupy"),
        "declared_outputs": list(OUTPUT_NAMES),
    }


def _energy_replay(impl, plan, case, resident, inputs, ref, row):
    """Untimed replay; host transfers intentionally occur only in this diagnostic."""
    if case.n > 257:
        row["energy_replay"] = {"performed": False, "reason": "n > 257"}
        return
    k, q, v = resident
    a = -(k @ q)
    initial = ref["initial_energy"]
    excursions = [0.0]
    for _ in range(STEPS):
        q, v, a = plan(k, q, v, a, DT)
        impl.adapter.sync()
        host = impl.download((q, v, a))
        change = relative_energy_change(initial, energy(inputs[0], *host[:2]))
        excursions.append(abs(change) if change is not None else None)
    diagnose(row, host, ref["discrete"], case, "energy_replay_final")
    finite = [x for x in excursions if x is not None and np.isfinite(x)]
    row["energy_replay"] = {
        "performed": True,
        "samples_including_t0": len(excursions),
        "max_absolute_relative_excursion": max(finite) if len(finite) == len(excursions) else None,
        "scope": "untimed replay, float64 host energy after each step",
    }


def _initialize(case, impl, inputs, ref, row, warmup_seconds):
    adapter, stage = impl.adapter, "initial_upload"
    try:
        elapsed, resident = _measure(adapter, lambda: adapter.upload(inputs))
        row["timings"][stage].append(elapsed)
        stage = "acceleration_example"
        elapsed, a_example = _measure(adapter, lambda: -(resident[0] @ resident[1]))
        row["timings"][stage].append(elapsed)
        stage = "prepare"
        elapsed, plan = _measure(adapter, lambda: impl.prepare(case, resident, a_example))
        row["timings"][stage].append(elapsed)
        stage = "first_call"
        elapsed, value = _measure(adapter, lambda: impl.calculate(plan, case, resident))
        row["timings"][stage].append(elapsed)
        elapsed, host = _measure(adapter, lambda: impl.download(value))
        row["timings"]["download"].append(elapsed)
        row["output_bytes"] = sum(a.nbytes for a in host)
        diagnose(row, host, ref["discrete"], case, stage)
        row["final_relative_energy_change"] = relative_energy_change(
            ref["initial_energy"], energy(inputs[0], *host[:2])
        )
        stage = "warmup"
        start, count = time.perf_counter(), 0
        while count < 3 or time.perf_counter() - start < warmup_seconds:
            value = impl.calculate(plan, case, resident)
            adapter.sync()
            diagnose(row, impl.download(value), ref["discrete"], case, "warmup_latest")
            count += 1
        row["warmup"] = {
            "calls": count,
            "seconds": time.perf_counter() - start,
            "includes_untimed_validation": True,
        }
        stage = "energy_replay"
        _energy_replay(impl, plan, case, resident, inputs, ref, row)
        return resident, plan
    except Exception as exc:  # noqa: BLE001 - preserve failed rows while other implementations run.
        row["error"] = shared._failure(exc, stage)
        return None


def _round(case, impl, inputs, ref, row, state, repeat):
    adapter, (resident, plan), stage = impl.adapter, state, "resident"
    try:
        elapsed, value = _measure(adapter, lambda: impl.calculate(plan, case, resident))
        row["timings"][stage].append(elapsed)
        diagnose(row, impl.download(value), ref["discrete"], case, f"resident-{repeat}")
        stage = "upload"
        elapsed, uploaded = _measure(adapter, lambda: adapter.upload(inputs))
        row["timings"][stage].append(elapsed)
        del uploaded
        stage = "complete"

        def complete():
            uploaded = adapter.upload(inputs)
            return impl.download(impl.calculate(plan, case, uploaded))

        elapsed, host = _measure(adapter, complete)
        row["timings"][stage].append(elapsed)
        diagnose(row, host, ref["discrete"], case, f"complete-{repeat}")
        return True
    except Exception as exc:  # noqa: BLE001 - retain timing samples preceding backend failures.
        row["error"] = shared._failure(exc, stage)
        return False


def provenance(backend, impls):
    result = shared.provenance(backend, impls)
    result["runner_sha256"] = shared._hash_sources([Path(__file__)])
    result["example_sha256"] = shared._hash_sources([Path(oscillators.__file__)])
    path = Path(shared.__file__).resolve()
    result["measurement_helpers"]["dense_network"] = {
        "source": str(path),
        "sha256": shared._hash_sources([path]),
    }
    return result


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
    """Reserve an exclusive report and preserve incremental failures and timings."""
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
        "cases": [],
        "attempts_complete": False,
        "execution_complete": False,
        "all_numerically_accepted": False,
    }
    impls = []
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
            except Exception as exc:  # noqa: BLE001 - missing providers must have explicit rows.
                report["backend_error"] = shared._failure(exc, "initialize_backend")
                modes = ("eager",) if backend in ("numpy", "cupy") else ("eager", "compiled")
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
                except Exception as exc:  # noqa: BLE001 - provenance discovery must not suppress execution.
                    report["provenance_error"] = shared._failure(exc, "provenance")
                for case in selected_cases:
                    certificate = {}
                    with threadpool_limits(limits=1, user_api="blas"):
                        start = time.perf_counter()
                        inputs = make_inputs(case, certificate=certificate)
                        generation_seconds = time.perf_counter() - start
                        start = time.perf_counter()
                        ref = reference(case, inputs, certificate)
                        reference_seconds = time.perf_counter() - start
                    report["cases"].append(
                        {
                            "case_id": case.id,
                            "generation_seconds": generation_seconds,
                            "reference_seconds": reference_seconds,
                            "cpu_blas_threads": 1,
                        }
                    )
                    rows = [_row(case, impl, inputs, ref) for impl in impls]
                    report["rows"].extend(rows)
                    states = []
                    for impl, row in zip(impls, rows):
                        with threadpool_limits(limits=impl.adapter.threads, user_api="blas"):
                            row["threadpools"] = threadpool_info()
                            states.append(_initialize(case, impl, inputs, ref, row, warmup_seconds))
                        save()
                    orders = shared.balanced_orders(len(impls), repeats, SEED)
                    for row in rows:
                        row["round_order"] = [[impls[i].name for i in order] for order in orders]
                    for repeat, order in enumerate(orders):
                        for index in order:
                            if states[index] is None:
                                continue
                            impl, row = impls[index], rows[index]
                            with threadpool_limits(limits=impl.adapter.threads, user_api="blas"):
                                if not _round(case, impl, inputs, ref, row, states[index], repeat):
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
            report["fatal_error"] = shared._failure(exc, "run")
            raise
        finally:
            preserving_exception = sys.exc_info()[0] is not None
            shared._close_adapters(impls, report)
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
