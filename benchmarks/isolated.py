"""Sequential benchmark workers with recorded idle gates between processes."""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks.stability import Policy, nvidia_snapshot, wait_ready


def run(
    output,
    backends,
    *,
    rounds=2,
    repeats=21,
    seed=941,
    gpu_id="0",
    policy=None,
    allow_unverified=False,
    worker="backend_sharing",
    control_wheel=None,
    control_component="methods",
):
    if output.exists() or rounds < 1 or repeats < 3:
        raise ValueError("Use a new directory, positive rounds and at least three samples")
    allowed = {
        "numpy-1t",
        "numpy-8t",
        "opencl",
        "opencl-clblast",
        "vulkan",
        "cuda-custom",
        "cuda-cublas",
        "cupy",
    }
    if not backends or len(set(backends)) != len(backends) or set(backends) - allowed:
        raise ValueError("Select distinct supported backends")
    if worker not in (
        "backend_sharing",
        "kernel_variants",
        "staging",
        "random_generation",
        "completion",
        "routing",
        "fusion",
        "cuda_dispatch",
        "three_actions",
        "padding",
        "kernel_inspection",
        "padding_copy_review",
        "padding_decision",
        "rectangular_gemm",
        "blas_vectors",
        "compiled_layouts",
    ):
        raise ValueError("Unknown worker")
    if worker in ("padding_decision", "rectangular_gemm", "blas_vectors", "compiled_layouts"):
        if repeats < 6 or repeats % 6:
            raise ValueError("Paired decision requires repeats to be a multiple of six")
        supported = {
            "padding_decision": {"opencl", "vulkan", "cuda-custom"},
            "rectangular_gemm": {"opencl", "cuda-custom"},
            "blas_vectors": {"cuda-cublas"},
            "compiled_layouts": {"opencl-clblast"},
        }
        if set(backends) - supported[worker]:
            raise ValueError("Paired decision does not support the selected backend")
    if control_wheel is not None and worker not in (
        "padding_copy_review",
        "backend_sharing",
        "staging",
        "completion",
        "routing",
        "fusion",
        "cuda_dispatch",
        "three_actions",
        "blas_vectors",
    ):
        raise ValueError("This benchmark worker does not support a historical wheel control")
    output.mkdir(parents=True)
    policy = policy or Policy()
    report = {
        "schema": "openpacho.isolated.v1",
        "policy": asdict(policy),
        "gpu_id": gpu_id,
        "seed": seed,
        "rounds": rounds,
        "repeats": repeats,
        "worker": worker,
        "runs": [],
        "complete": False,
        "scope": "process exit, observed idle gate, next worker; warmup outside sample clocks; no thermal steady-state guarantee",
        "device_association": "telemetry GPU is selected explicitly; backend-to-UUID mapping is not verified automatically",
    }
    manifest = output / "session.json"
    save = lambda: manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    baseline = identity = None
    probe = lambda: nvidia_snapshot(gpu_id)
    # Randomized first order, reverse in the second round to balance position.
    order = list(backends)
    random.Random(seed).shuffle(order)
    try:
        for round_index in range(rounds):
            for backend in order if round_index % 2 == 0 else list(reversed(order)):
                row = {"round": round_index + 1, "backend": backend}
                report["runs"].append(row)
                print(
                    f"Waiting for GPU readiness before {backend}, round {round_index + 1}",
                    flush=True,
                )
                row["before"] = wait_ready(
                    probe, policy, baseline_temperature=baseline, allow_unverified=allow_unverified
                )
                save()
                if not row["before"]["ready"]:
                    row["status"] = "not-run-readiness"
                    return report
                if baseline is None:
                    baseline = row["before"]["baseline_temperature"]
                    report["reference_idle_temperature"] = baseline
                observed = row["before"].get("uuid")
                if observed is not None:
                    if identity is not None and observed != identity:
                        row["status"] = "not-run-device-changed"
                        return report
                    identity = observed
                filename = f"{round_index + 1}-{backend}"
                result_path = output / (filename + ".json")
                command = [
                    sys.executable,
                    "-m",
                    f"benchmarks.{worker}",
                    "--backends",
                    backend,
                    "--repeats",
                    str(repeats),
                    "--seed",
                    str(seed + round_index),
                    "--output",
                    str(result_path.resolve()),
                ]
                row["command"] = command
                if control_wheel is not None:
                    command.extend(
                        [
                            "--control-wheel",
                            str(control_wheel.resolve()),
                            "--control-component",
                            control_component,
                        ]
                    )
                env = os.environ.copy()
                env["PACHO_BENCHMARK_GPU_ID"] = str(gpu_id)
                env.setdefault("CUPY_CACHE_DIR", str(Path.cwd() / ".artifacts/cupy-cache"))
                env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
                start = time.monotonic()
                with (output / (filename + ".log")).open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                row.update(
                    returncode=result.returncode,
                    process_seconds=time.monotonic() - start,
                    after=probe(),
                    status="ok" if result.returncode == 0 else "worker-failed",
                )
                if result.returncode == 0 and not result_path.exists():
                    row["status"] = "worker-missing-report"
                save()
                print(f"Completed {backend}, round {round_index + 1}: {row['status']}", flush=True)
                if row["status"] != "ok":
                    return report
        report["complete"] = True
        return report
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="opencl,vulkan,cuda-custom")
    parser.add_argument(
        "--worker",
        choices=(
            "backend_sharing",
            "kernel_variants",
            "staging",
            "random_generation",
            "completion",
            "routing",
            "fusion",
            "cuda_dispatch",
            "three_actions",
            "padding",
            "kernel_inspection",
            "padding_copy_review",
            "padding_decision",
            "rectangular_gemm",
            "blas_vectors",
            "compiled_layouts",
        ),
        default="backend_sharing",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--control-wheel", type=Path)
    parser.add_argument("--control-component", choices=("methods", "gemm"), default="methods")
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="NVIDIA UUID/index to observe; select matching devices in each backend",
    )
    parser.add_argument("--idle-seconds", type=float, default=10)
    parser.add_argument("--max-wait-seconds", type=float, default=120)
    parser.add_argument("--max-temperature", type=float)
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Explicit fixed pause when no telemetry exists; never bypass hot/busy telemetry",
    )
    args = parser.parse_args()
    try:
        policy = Policy(
            min_idle_seconds=args.idle_seconds,
            max_wait_seconds=args.max_wait_seconds,
            max_temperature=args.max_temperature,
        )
        report = run(
            args.output,
            args.backends.split(","),
            rounds=args.rounds,
            repeats=args.repeats,
            seed=args.seed,
            gpu_id=args.gpu_id,
            policy=policy,
            worker=args.worker,
            allow_unverified=args.allow_unverified,
            control_wheel=args.control_wheel,
            control_component=args.control_component,
        )
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
