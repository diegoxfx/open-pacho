"""Compare fusion/custom/CLBlast and optionally the previous wheel in one session."""

import argparse
import hashlib
import importlib.util
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np

from benchmarks.suite import Adapter, Case, operation, run_case, source_hashes, telemetry


def load_legacy(wheel, output):
    target = output / "legacy"
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            parts = Path(name).parts
            if not parts or parts[0] != "openpacho" or ".." in parts or name.endswith("/"):
                continue
            destination = target.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    import sys

    spec = importlib.util.spec_from_file_location(
        "_pacho_legacy_benchmark",
        target / "openpacho" / "__init__.py",
        submodule_search_locations=[str(target / "openpacho")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LegacyAdapter(Adapter):
    def __init__(self, module):
        self.name, self.cpu_threads, self.xp = "opencl-legacy", None, module
        self.engine = module.get_backend("opencl")

    def upload(self, args):
        return tuple(self.xp.asarray(a, backend=self.engine) for a in args)

    def host(self, value):
        return self.xp.asnumpy(value)

    def compute(self, case, args, diagnostics=False):
        if case.operation in ("cg", "jacobi", "sd"):
            solver = getattr(
                self.xp.solvers,
                {"cg": "conjugate_gradient", "jacobi": "jacobi", "sd": "steepest_descent"}[
                    case.operation
                ],
            )
            result = solver(*args, rtol=1e-4 if case.dtype == "float32" else 1e-7, maxiter=400)
            if not result.converged:
                raise ValueError(result.reason)
            return (result.x, result.iterations) if diagnostics else result.x
        return operation(case, self.xp, args, diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-wheel", type=Path)
    parser.add_argument("--clblast", action="store_true")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--only", help="comma-separated operation names")
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("at least 3 rounds required")
    if args.output.exists():
        parser.error("output must be a new directory")
    args.output.mkdir(parents=True)
    adapters = [Adapter(f"numpy-{n}") for n in (1, 8, 16)]
    adapters += [Adapter("opencl"), Adapter("opencl-fused")]
    if args.clblast:
        adapters.append(Adapter("opencl-clblast"))
    if args.legacy_wheel:
        adapters.append(LegacyAdapter(load_legacy(args.legacy_wheel, args.output)))
    cases = []
    for dtype in ("float32", "float64"):
        cases.extend(Case("transcendental", (n,), dtype) for n in (1024, 1048576, 4194304))
        cases.append(Case("reuse", (1048576,), dtype, 100))
        cases.extend(Case("matmul", (n, n, n), dtype) for n in (256, 1024, 2048))
        cases.extend(Case(op, (n, n), dtype) for n in (1024, 4096) for op in ("cg", "jacobi", "sd"))
    if args.only:
        cases = [case for case in cases if case.operation in args.only.split(",")]
        if not cases:
            parser.error("no matching cases")
    random.Random(4701).shuffle(cases)
    metadata = {
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_order": [case.id for case in cases],
        "source_sha256": source_hashes(),
        "telemetry_start": telemetry(),
        "rounds": args.rounds,
        "seed": 4701,
        "numpy_version": np.__version__,
        "legacy_wheel_sha256": hashlib.sha256(args.legacy_wheel.read_bytes()).hexdigest()
        if args.legacy_wheel
        else None,
        "policy": "randomized synchronized wall time; warm runtime; native allocations; "
        "same inputs/dtypes; best measured NumPy BLAS1/8/16; each call includes result release; "
        "CPU host input/output; end_to_end includes upload/compute/download; "
        "legacy is unmodified archived wheel, loaded alongside current package",
    }
    result = {"metadata": metadata, "rows": [], "errors": []}
    start = time.perf_counter()
    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {case.id}", flush=True)
        selected = [
            a
            for a in adapters
            if not (
                a.name == "opencl-fused"
                and case.operation not in ("reuse", "transcendental")
                or a.name == "opencl-clblast"
                and case.operation in ("reuse", "transcendental")
            )
        ]
        try:
            result["rows"].extend(run_case(case, selected, args.rounds, 4701 + i, 0.002, 8))
        except Exception as exc:  # noqa: BLE001 -- persist failed cases and exit nonzero
            result["errors"].append({"case": case.id, "error": repr(exc)})
            print(f"ERROR: {exc}", flush=True)
        (args.output / "checkpoint.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    metadata["elapsed_s"] = time.perf_counter() - start
    metadata["source_unchanged"] = source_hashes() == metadata["source_sha256"]
    metadata["telemetry_end"] = telemetry()
    (args.output / "raw.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Optimización: comparación en la misma sesión",
        "",
        metadata["policy"],
        "",
        "Medianas end-to-end en ms (CPU sin copias artificiales).",
        "",
        "| Caso | NumPy | Anterior | Custom | Fusionado | CLBlast |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in sorted(cases, key=lambda c: c.id):
        rows = [r for r in result["rows"] if r["case_id"] == case.id]
        cpu = [r["median_s"] for r in rows if r["mode"] == "cpu"]
        if not cpu:
            continue
        values = [f"{min(cpu) * 1000:.3f}"]
        for backend in ("opencl-legacy", "opencl", "opencl-fused", "opencl-clblast"):
            matches = [r for r in rows if r["backend"] == backend and r["mode"] == "end_to_end"]
            values.append(f"{matches[0]['median_s'] * 1000:.3f}" if matches else "—")
        lines.append(f"| {case.id} | {' | '.join(values)} |")
    lines += [
        "",
        f"Errores: {result['errors']}",
        f"Fuentes sin cambios: {metadata['source_unchanged']}",
        "",
        (
            "Solvers: matrices SPD tridiagonales almacenadas y procesadas densas. "
            "No compara con un solver disperso especializado. Convergencia y residuo "
            "se validan; las iteraciones pueden variar por redondeo."
        ),
        "",
        "Muestras, validaciones, tiempos residentes y hashes: [raw.json](raw.json).",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Done: {len(result['rows'])} rows, {len(result['errors'])} errors", flush=True)
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
