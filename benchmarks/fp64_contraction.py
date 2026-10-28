"""Bounded benchmark-only FP64 contraction experiment.

Run ONE worker after an external exclusive-device readiness gate, with no other
GPU work: ``python -m benchmarks.fp64_contraction --backend cuda-custom
--profile all --output NEW.json``. Repeat with OpenCL in a separate process.
Import does not initialize a GPU. No runtime flags, dispatch, or caches change.
Timings describe preallocated resident kernels, never application performance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks import fp64_contraction_kernels as kernels
from benchmarks import fp64_contraction_oracle as oracle
from benchmarks.cuda_fp64_gemm import MODES, measure, mode_orders
from benchmarks.gaps import _sample_stats

SEED = 1069
PERFORMANCE_SHAPES = ((4096, 128, 256), (128, 4096, 256), (256, 4096, 32))
FLAG_NAMES = ("blas", "fp64", "_register_gemm", "_gemm_tile", "_architecture")
UNSAFE_OPENCL_FLAGS = (
    "-cl-fast-relaxed-math",
    "-cl-unsafe-math-optimizations",
    "-cl-finite-math-only",
    "-cl-denorms-are-zero",
    "-cl-no-signed-zeros",
    "-cl-mad-enable",
    "-ffast-math",
    "-Ofast",
    "-ffp-contract=fast",
    "-funsafe-math-optimizations",
    "-fno-signed-zeros",
    "-fdenormal-fp-math=positive-zero",
    "-fdenormal-fp-math=preserve-sign",
)


def provenance():
    package = Path(p.__file__).resolve().parent
    helpers = (
        Path(__file__),
        Path(kernels.__file__),
        Path(oracle.__file__),
        Path(__file__).with_name("cuda_fp64_gemm.py"),
        Path(__file__).with_name("cuda_fp64_tile.py"),
        Path(__file__).with_name("kernel_inspection.py"),
        Path(__file__).with_name("gaps.py"),
    )
    versions = {}
    for name in ("numpy", "threadpoolctl", "pyopencl", "cuda-bindings", "nvidia-cuda-nvrtc"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_origin": str(Path(p.__file__).resolve()),
        "versions": versions,
        "runtime_sources": {
            str(path.relative_to(package)): digest(path.read_bytes())
            for path in sorted(package.rglob("*.py"))
        },
        "helpers": {str(path.resolve()): digest(path.read_bytes()) for path in helpers},
        "build_environment": {
            key: os.environ.get(key)
            for key in ("PYOPENCL_BUILD_OPTIONS", "PYOPENCL_NO_CACHE", "POCL_EXTRA_BUILD_FLAGS")
        },
    }


def digest(data):
    return hashlib.sha256(data).hexdigest()


def runtime_state(backend):
    """Audit identities privately; persisted hashes do not serialize driver handles."""
    cache = getattr(backend, "_kernels", {})
    return {
        "cache_identity": id(cache),
        "cache_entries": [
            {"source_sha256": digest(str(key).encode()), "object_identity": id(value)}
            for key, value in cache.items()
        ],
        "flags": {name: getattr(backend, name, None) for name in FLAG_NAMES},
    }


def policy_evidence(metadata, backend_name, variant):
    """Infer only an unmixed compute-entry instruction policy, never opaque IR."""
    evidence = metadata.get("f64_instruction_evidence")
    observed = None
    if evidence is not None:
        fma, mul, add = (evidence.get(name + "_f64", 0) for name in ("fma", "mul", "add"))
        if fma > 0 and mul == 0 and add == 0:
            observed = "fused"
        elif fma == 0 and mul > 0 and add > 0:
            observed = "separate"
    requested = (
        "fused"
        if variant == "contracted"
        else "separate"
        if variant in ("separate", "no_contraction")
        else None
    )
    # Some implementations append environment flags without returning them in
    # build.OPTIONS. Preserve the distinction between visible instruction policy
    # and a compiler configuration supporting a contraction-only comparison.
    build_text = (
        str(metadata.get("effective_build_options", ""))
        + " "
        + " ".join(
            str(value or "")
            for key, value in metadata.get("build_environment", {}).items()
            if key in ("PYOPENCL_BUILD_OPTIONS", "POCL_EXTRA_BUILD_FLAGS")
        )
    )
    unsafe = (
        [flag for flag in UNSAFE_OPENCL_FLAGS if flag in build_text]
        if backend_name == "opencl"
        else []
    )
    sufficient = (
        observed is not None and (requested is None or observed == requested) and not unsafe
    )
    return {
        "observed_policy": observed,
        "requested_policy": requested,
        "evidence_sufficient": sufficient,
        "unsafe_build_flags": unsafe,
        "status": "unsafe_opencl_build_flags"
        if unsafe
        else "verified_compute_entry_ir"
        if sufficient
        else "insufficient_or_mixed_instruction_evidence",
        "scope": "static compiler IR, not loaded machine code",
        "backend": backend_name,
    }


def poison_output(backend, out, host):
    """Overwrite the existing buffer outside clocks without compiling fill kernels."""
    if backend.name == "cuda":
        from openpacho._cuda import _check

        backend.synchronize()
        with backend._activate():
            _check(backend.driver.cuMemcpyHtoD(out._data.ptr, host.ctypes.data, host.nbytes))
            _check(backend.driver.cuStreamSynchronize(0))
    elif backend.name == "opencl":
        backend.cl.enqueue_copy(backend.queue, out._data, host, is_blocking=True)
    elif backend.name == "numpy":  # CPU protocol tests only; never a CLI fallback.
        np.copyto(out._data.reshape(out.shape), host)
    else:
        raise ValueError("Unsupported buffer poison backend")


def _error(exc):
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


def _inputs(profile, seed):
    for name, host in oracle.cases():
        yield name, "ieee", host
    shapes = PERFORMANCE_SHAPES if profile == "all" else ((33, 31, 65),)
    # Shape-specific deterministic generators keep operands identical across
    # providers, geometries and process rounds, regardless of critical-case count.
    for m, k, n in shapes:
        rng = np.random.default_rng(np.random.SeedSequence([seed, m, k, n]))
        yield (
            f"performance-{m}x{k}x{n}",
            "performance",
            tuple(rng.uniform(-1, 1, shape).astype(np.float64) for shape in ((m, k), (k, n))),
        )


def _validation(actual, references, evidence, numpy_reference):
    diagnostics = {
        policy: oracle.evaluate(actual, references, policy) for policy in ("separate", "fused")
    }
    policy = evidence["observed_policy"] if evidence["evidence_sufficient"] else None
    tolerance = (
        None
        if numpy_reference is None
        else bool(np.allclose(actual, numpy_reference, rtol=2e-10, atol=2e-12, equal_nan=False))
    )
    # An opaque binary can produce useful numbers without establishing causality.
    policy_ok = diagnostics[policy]["policy_equivalent"] if policy else None
    accepted = tolerance is not False and (
        policy_ok if policy else any(value["policy_equivalent"] for value in diagnostics.values())
    )
    return {
        "accepted": bool(accepted),
        "numpy_float64_allclose": tolerance,
        "numpy_tolerance": {"rtol": 2e-10, "atol": 2e-12},
        "policy_compliance_verified": bool(policy is not None and policy_ok),
        "policy_diagnostics": diagnostics,
    }


def run_row(backend, backend_name, geometry, kind, host, row, folder, repeats, seed, save):
    """Own every executable until cleanup, including preparation/launch failures."""
    owners = []
    try:
        a, b = (p.asarray(value, backend=backend) for value in host)
        shape = (a.shape[0], b.shape[1])
        row["shape_mkn"] = [a.shape[0], a.shape[1], b.shape[1]]
        row["input_sha256"] = [digest(value.tobytes()) for value in host]
        row["runtime_before_native"] = runtime_state(backend)
        # Authoritative production route runs before any owned raw samples.
        elapsed, native = measure(backend, lambda: a @ b)
        authoritative = np.array(p.asnumpy(native), copy=True)
        row["native_first_call_seconds"] = elapsed
        row["native_output_sha256"] = digest(authoritative.tobytes())
        row["runtime_after_native"] = runtime_state(backend)
        descriptor = kernels.capture(backend, a, b, shape, geometry)
        row["descriptor"] = descriptor.metadata()
        coordinates = (
            None
            if kind == "ieee"
            else sorted(
                {
                    (0, 0),
                    (shape[0] - 1, shape[1] - 1),
                    (shape[0] // 2, shape[1] // 2),
                    (min(16, shape[0] - 1), min(31, shape[1] - 1)),
                    (shape[0] // 3, shape[1] // 3),
                }
            )
        )
        references = oracle.sampled_references(
            *host, coordinates=coordinates, padded_k=descriptor.accumulation_steps
        )
        row["sampled_references"] = references
        numpy_reference = None
        if kind == "performance":
            with threadpool_limits(limits=1, user_api="blas"):
                numpy_reference = host[0] @ host[1]
        variants = (
            ("separate", "contracted")
            if backend_name == "cuda-custom"
            else ("native", "no_contraction")
        )
        # Independent bracketing controls share an executable, never an output.
        # Finish buffer setup and preflight both poison uploads before any owner
        # snapshots the runtime cache. These are copies, never fill kernels.
        output = {mode: p.empty(shape, dtype=np.float64, backend=backend) for mode in MODES}
        if len({id(value._data) for value in output.values()}) != 3:
            raise AssertionError("Three distinct preallocated buffers are required")
        poison = np.empty(shape, dtype=np.float64)
        poison.flat[::2], poison.flat[1::2] = np.nan, np.inf
        alternate_poison = np.where(np.isnan(poison), np.inf, np.nan)
        for out in output.values():
            poison_output(backend, out, poison)
            poison_output(backend, out, alternate_poison)
        row["poison"] = "alternating NaN/+Inf before every checked/sample call, outside clocks"
        row["poison_setup"] = "both patterns copied to all outputs before any owned compilation"
        row["runtime_after_setup"] = runtime_state(backend)
        if row["runtime_after_setup"] != row["runtime_after_native"]:
            raise AssertionError("Runtime flags/cache changed during capture/buffer/poison setup")
        prepared, row["variants"] = {}, {}
        for variant in variants:
            start = time.perf_counter()
            owner = kernels.prepare(backend, descriptor, variant, folder / variant)
            owners.append(owner)
            prepared[variant] = owner
            row["variants"][variant] = {
                "build_seconds": time.perf_counter() - start,
                "metadata": owner.metadata,
                "policy_evidence": policy_evidence(owner.metadata, backend_name, variant),
            }
            save()
        variant_for = {mode: variants[1] if mode == "on" else variants[0] for mode in MODES}
        expected, row["modes"] = {}, {}
        stable_runtime = runtime_state(backend)
        row["runtime_after_prepare"] = stable_runtime
        if stable_runtime != row["runtime_after_setup"]:
            raise AssertionError("Runtime flags/cache changed during owned preparation")

        def launch(mode):
            return prepared[variant_for[mode]].call(a, b, output[mode])

        def checked_call(mode, stage, timed=True):
            state = row["modes"][mode]
            poison_output(
                backend, output[mode], alternate_poison if len(state["checks"]) % 2 else poison
            )
            elapsed, value = measure(backend, lambda: launch(mode))
            actual = np.array(p.asnumpy(value), copy=True)
            if mode not in expected:
                expected[mode] = actual
                evidence = row["variants"][variant_for[mode]]["policy_evidence"]
                state["validation"] = _validation(actual, references, evidence, numpy_reference)
                if mode != "on":
                    state["native_reproduction"] = oracle.compare(actual, authoritative)
                state["output_sha256"] = digest(actual.tobytes())
            check = oracle.compare(actual, expected[mode])
            state["checks"].append({"stage": stage, **check})
            if timed:
                state["samples_seconds"].append(elapsed)
            return elapsed

        for mode in MODES:
            row["modes"][mode] = {"variant": variant_for[mode], "samples_seconds": [], "checks": []}
            row["modes"][mode]["first_call_seconds"] = checked_call(mode, "first", False)
            save()
        if kind == "performance":
            for mode in MODES:
                start, count = time.monotonic(), 0
                while count < 3 or time.monotonic() - start < 0.10:
                    checked_call(mode, "warmup", False)
                    count += 1
                row["modes"][mode]["warmup"] = {"calls": count, "seconds": time.monotonic() - start}
            row["orders"] = mode_orders(repeats, seed)
            for index, order in enumerate(row["orders"]):
                for mode in order:
                    checked_call(mode, f"sample-{index}")
                save()
            for state in row["modes"].values():
                state["timing"] = _sample_stats(state["samples_seconds"])
            medians = {
                mode: float(np.median(state["samples_seconds"]))
                for mode, state in row["modes"].items()
            }
            row["candidate_speedup_vs_bracketing_controls"] = (
                medians["off-a"] + medians["off-b"]
            ) / (2 * medians["on"])
            row["control_b_over_a"] = medians["off-b"] / medians["off-a"]
        else:
            for mode in MODES:
                checked_call(mode, "replay", False)
        row["runtime_after_owned_samples"] = runtime_state(backend)
        row["runtime_unchanged_during_owned_work"] = stable_runtime == runtime_state(backend)
        if not row["runtime_unchanged_during_owned_work"]:
            raise AssertionError("Runtime flags/cache changed during owned kernel work")
        row["execution_complete"] = True
        row["evidence_sufficient"] = all(
            state["policy_evidence"]["evidence_sufficient"] for state in row["variants"].values()
        )
        row["numerically_accepted"] = all(
            state["validation"]["accepted"]
            and state.get("native_reproduction", {"equivalent": True})["equivalent"]
            and all(check["equivalent"] for check in state["checks"])
            for state in row["modes"].values()
        )
    finally:
        failures = []
        for owner in reversed(owners):
            try:
                owner.close()
            except Exception as exc:  # noqa: BLE001 - retain all owner cleanup failures.
                failures.append(_error(exc))
        if failures:
            row["cleanup_errors"] = failures
        row["owned_resources_closed"] = all(owner.closed for owner in owners)
        save()


def run(output, backend, *, profile="all", repeats=24, seed=SEED):
    mode_orders(repeats, seed)
    if backend not in ("cuda-custom", "opencl") or profile not in ("quick", "all"):
        raise ValueError("Select cuda-custom/opencl and quick/all")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "openpacho.fp64-contraction.v1",
        "backend": backend,
        "profile": profile,
        "repeats": repeats,
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance(),
        "rows": [],
        "attempts_complete": False,
        "execution_complete": False,
        "numerically_accepted": False,
        "evidence_sufficient": False,
        "requirements": "External exclusive-device readiness gate; one worker and no concurrent GPU work",
        "timing_scope": "synchronized resident preallocated kernel wall time, including Python dispatch; no allocation/upload/download in clocks; no application performance claim",
        "warmup_policy": ">=3 calls and >=0.10 seconds per mode after separate first call",
        "comparison_policy": "Independent controls bracket candidate; each variant compares to its own verified sequential policy; exact mathematical accuracy is separate; numerical failures remain visible and timed",
    }
    device = None
    # Reserve before GPU initialization and preserve partial results on failures.
    with path.open("x", encoding="utf-8") as stream:

        def save():
            stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.truncate()
            stream.flush()

        save()
        try:
            artifact_root = path.with_name(path.name + ".artifacts")
            artifact_root.mkdir(exist_ok=False)
            report["artifact_directory"] = str(artifact_root.resolve())
            device = p.get_backend("cuda" if backend == "cuda-custom" else "opencl", blas="custom")
            report["device"] = str(device.device)
            report["initial_runtime_state"] = runtime_state(device)
            geometries = ("direct", "tile16") if backend == "cuda-custom" else ("native",)
            for name, kind, host in _inputs(profile, seed):
                for geometry in geometries:
                    row = {
                        "name": name,
                        "kind": kind,
                        "geometry": geometry,
                        "execution_complete": False,
                        "numerically_accepted": False,
                        "evidence_sufficient": False,
                    }
                    report["rows"].append(row)
                    print(f"{backend}: {name}/{geometry}", flush=True)
                    save()
                    try:
                        run_row(
                            device,
                            backend,
                            geometry,
                            kind,
                            host,
                            row,
                            artifact_root / f"{name}-{geometry}",
                            repeats,
                            seed,
                            save,
                        )
                    except Exception as exc:  # noqa: BLE001 - continue independent cases.
                        row["error"] = _error(exc)
                    save()
            report["attempts_complete"] = True
            for flag in ("execution_complete", "numerically_accepted", "evidence_sufficient"):
                report[flag] = bool(report["rows"]) and all(row[flag] for row in report["rows"])
            report["causal_conclusion_allowed"] = (
                report["evidence_sufficient"] and report["numerically_accepted"]
            )
        except Exception as exc:  # noqa: BLE001 - persist initialization/setup evidence.
            report["error"] = _error(exc)
        finally:
            if device is not None:
                try:
                    device.synchronize()
                    report["final_runtime_state"] = runtime_state(device)
                    close = getattr(device, "close", None)
                    if close is not None:
                        close()
                    report["backend_cleanup"] = (
                        "closed" if close else "queue synchronized; backend exposes no close"
                    )
                except Exception as exc:  # noqa: BLE001 - preserve primary failure.
                    report["cleanup_error"] = _error(exc)
            save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cuda-custom", "opencl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("quick", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--seed", type=int, default=SEED)
    report = run(**vars(parser.parse_args(argv)))
    return (
        0
        if (
            report["attempts_complete"]
            and report["execution_complete"]
            and report["numerically_accepted"]
            and "cleanup_error" not in report
            and all(not row.get("cleanup_errors") for row in report["rows"])
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
