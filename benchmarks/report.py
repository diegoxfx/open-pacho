"""Generate transparent CSV summaries, static charts and a Spanish report from raw data."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"opencl": "OpenPacho / OpenCL", "cuda": "CuPy / CUDA"}
COLORS = {"opencl": "#167d8d", "cuda": "#bf5b20"}


def summarize(report):
    grouped = defaultdict(list)
    for row in report["rows"]:
        grouped[row["case_id"]].append(row)
    comparisons = []
    for case_id, rows in grouped.items():
        cpu = min((r for r in rows if r["mode"] == "cpu"), key=lambda r: r["median_s"])
        for row in rows:
            if row["mode"] not in ("resident", "end_to_end"):
                continue
            seed = int(
                hashlib.sha256((case_id + row["backend"] + row["mode"]).encode()).hexdigest()[:8],
                16,
            )
            rng = np.random.default_rng(seed)
            c, g = np.array(cpu["samples_s"]), np.array(row["samples_s"])
            indices = rng.integers(0, len(c), (2000, len(c)))
            ratios = np.median(c[indices], axis=1) / np.median(g[indices], axis=1)
            low, high = np.quantile(ratios, [0.025, 0.975])
            ratio = cpu["median_s"] / row["median_s"]
            classification = (
                "faster"
                if ratio >= 1.1 and low > 1
                else "slower"
                if ratio <= 1 / 1.1 and high < 1
                else "unclear"
            )
            comparisons.append(
                {
                    "case_id": case_id,
                    "operation": row["operation"],
                    "shape": row["shape"],
                    "dtype": row["dtype"],
                    "steps": row["steps"],
                    "backend": row["backend"],
                    "mode": row["mode"],
                    "cpu_backend": cpu["backend"],
                    "cpu_s": cpu["median_s"],
                    "gpu_s": row["median_s"],
                    "speedup": ratio,
                    "bootstrap_low": float(low),
                    "bootstrap_high": float(high),
                    "classification": classification,
                    "valid": row["validation"]["valid"],
                }
            )
    return comparisons


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_matmul(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for i, dtype in enumerate(("float32", "float64")):
        for j, mode in enumerate(("resident", "end_to_end")):
            ax = axes[i, j]
            for backend, label in LABELS.items():
                selected = sorted(
                    (
                        r
                        for r in rows
                        if r["operation"] == "matmul"
                        and len(set(r["shape"])) == 1
                        and r["dtype"] == dtype
                        and r["mode"] == mode
                        and r["backend"] == backend
                    ),
                    key=lambda r: r["shape"][0],
                )
                if not selected:
                    continue
                x = [r["shape"][0] for r in selected]
                y = [r["speedup"] for r in selected]
                ax.plot(x, y, "o-", color=COLORS[backend], label=label)
                ax.fill_between(
                    x,
                    [r["bootstrap_low"] for r in selected],
                    [r["bootstrap_high"] for r in selected],
                    color=COLORS[backend],
                    alpha=0.12,
                )
            ax.axhline(1, color="#555555", ls="--", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.set_title(
                f"{dtype} · {'datos residentes' if mode == 'resident' else 'incluye subida y descarga'}"
            )
            ax.set_ylabel("Tiempo NumPy / tiempo GPU")
            ax.set_xlabel("Lado de la matriz cuadrada")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Producto matricial: más de 1× significa que gana GPU\nBaseline: NumPy más rápido medido entre 1, 8 y 16 hilos",
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, output, "matmul-speedup")


def plot_workloads(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, op, title in zip(
        axes.flat,
        ("add", "transcendental", "distance", "reuse"),
        (
            "Suma: poco cálculo por byte",
            "sin / exp / cos: más cálculo",
            "Distancia: salida escalar",
            "100 pasos, una subida y descarga",
        ),
    ):
        for backend, label in LABELS.items():
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["operation"] == op
                    and r["dtype"] == "float32"
                    and r["mode"] == "end_to_end"
                    and r["backend"] == backend
                    and (op != "reuse" or r["steps"] == 100)
                ),
                key=lambda r: r["shape"][0],
            )
            ax.plot(
                [r["shape"][0] for r in selected],
                [r["speedup"] for r in selected],
                "o-",
                color=COLORS[backend],
                label=label,
            )
        ax.axhline(1, ls="--", color="#555555", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.set_xlabel("Número de elementos")
        ax.set_ylabel("Tiempo NumPy / tiempo GPU")
        ax.set_title(title)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Coste completo con transferencias · float32", fontsize=14)
    fig.tight_layout()
    save_figure(fig, output, "workload-speedup")


def plot_transfers(report, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode, title in zip(axes, ("h2d", "d2h"), ("Host → GPU", "GPU → host")):
        for backend, label in LABELS.items():
            selected = sorted(
                (r for r in report["transport"] if r["backend"] == backend and r["mode"] == mode),
                key=lambda r: r["bytes"],
            )
            ax.plot(
                [r["bytes"] / 2**20 for r in selected],
                [r["GB_per_s"] for r in selected],
                "o-",
                color=COLORS[backend],
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Tamaño de la copia (MiB)")
        ax.set_ylabel("GB/s efectivos")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Copias con buffers ya asignados y memoria host pageable", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output, "transfer-bandwidth")


def plot_solvers(rows, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for i, dtype in enumerate(("float32", "float64")):
        for j, op in enumerate(("cg", "jacobi", "sd")):
            ax = axes[i, j]
            for backend, label in LABELS.items():
                selected = sorted(
                    (
                        r
                        for r in rows
                        if r["operation"] == op
                        and r["dtype"] == dtype
                        and r["mode"] == "end_to_end"
                        and r["backend"] == backend
                    ),
                    key=lambda r: r["shape"][0],
                )
                ax.plot(
                    [r["shape"][0] for r in selected],
                    [r["speedup"] for r in selected],
                    "o-",
                    label=label,
                    color=COLORS[backend],
                )
            ax.axhline(1, ls="--", color="#555555", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            name = {
                "cg": "Gradiente conjugado",
                "jacobi": "Jacobi",
                "sd": "Descenso por gradiente",
            }[op]
            ax.set_title(f"{name} · {dtype}")
            ax.set_xlabel("Número de variables")
            ax.set_ylabel("Tiempo NumPy / tiempo GPU")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Solvers con subida y descarga: el tamaño cambia el resultado\nMisma recurrencia y tolerancia; más de 1× significa que gana GPU",
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, output, "solver-speedup")


def table_row(case_id, comparisons):
    found = {
        r["backend"]: r
        for r in comparisons
        if r["case_id"] == case_id and r["mode"] == "end_to_end"
    }
    if not found:
        return None
    first = next(iter(found.values()))
    values = [f"`{case_id}`", f"{first['cpu_s'] * 1000:.4f} ({first['cpu_backend']})"]
    for backend in LABELS:
        row = found.get(backend)
        values.append(f"{row['gpu_s'] * 1000:.4f} / {row['speedup']:.2f}×" if row else "—")
    return "| " + " | ".join(values) + " |"


def build_report(raw, comparisons, output, cold=None):
    metadata = raw["metadata"]
    lines = [
        "# Benchmarks completos de OpenPacho, NumPy y CUDA",
        "",
        "Resultados locales de la implementación existente. No se modificaron kernels para esta comparación.",
        "",
        "## Hipótesis y criterio",
        "",
        (
            "La hipótesis sometida a prueba es que OpenPacho no puede ganar a NumPy por el coste de transferencia, ni con CUDA. "
            "Se separa la afirmación de rendimiento de su explicación causal. CuPy/CUDA es un control independiente con "
            "kernels, bibliotecas y asignador diferentes; OpenPacho sigue usando OpenCL."
        ),
        "",
        (
            "La referencia de cada configuración es **el NumPy más rápido medido entre 1, 8 y 16 hilos**. "
            "GPU/CPU no se ejecutaron a la vez. Se compara tiempo de pared sincronizado, con el mismo dtype y resultados validados. "
            "Un factor mayor que 1 significa que GPU tarda menos."
        ),
        "",
        "## Resumen de la cuadrícula",
        "",
        "| Motor | Escenario | Gana claramente | Pierde claramente | Diferencia pequeña/incierta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for backend, label in LABELS.items():
        for mode in ("resident", "end_to_end"):
            counts = Counter(
                r["classification"]
                for r in comparisons
                if r["backend"] == backend and r["mode"] == mode
            )
            lines.append(
                f"| {label} | {mode} | {counts['faster']} | {counts['slower']} | {counts['unclear']} |"
            )
    lines.extend(
        [
            "",
            (
                "“Claramente” exige una diferencia mediana de al menos 10% y que el intervalo bootstrap "
                "descriptivo de 95% no cruce 1. No es una probabilidad de ganar en aplicaciones reales: los recuentos "
                "dependen de esta cuadrícula de pruebas. El baseline CPU se selecciona con los mismos datos; "
                "esta selección favorece a CPU y los intervalos no corrigen la selección ni múltiples comparaciones."
            ),
            "",
            "## Ejemplos con coste completo",
            "",
            (
                "Incluyen subida, cálculo y descarga, con el runtime ya inicializado. Tiempos en milisegundos; "
                "la segunda cifra es tiempo de NumPy dividido por tiempo de GPU."
            ),
            "",
            "| Configuración | NumPy ms (hilos) | OpenPacho ms / factor | CuPy ms / factor |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    selected = [
        "add-16777216-float32-k1",
        "transcendental-16777216-float32-k1",
        "distance-16777216-float32-k1",
        "reuse-4194304-float32-k100",
        "matmul-1024x1024x1024-float32-k1",
        "matmul-2048x2048x2048-float32-k1",
        "matmul-2048x2048x2048-float64-k1",
        "matvec-4096x4096-float32-k1",
        "cg-1024x1024-float32-k1",
        "jacobi-1024x1024-float32-k1",
    ]
    for case in selected:
        row = table_row(case, comparisons)
        if row:
            lines.append(row)
    lines.extend(
        [
            "",
            "## Jacobi y descenso por gradiente",
            "",
            (
                "La tabla incluye tiempos residentes. Perder residente no puede atribuirse a subir y "
                "descargar la matriz completa en cada solución. Sí quedan sincronizaciones y pequeñas "
                "transferencias de escalares por iteración."
            ),
            "",
            "| Método y tamaño | NumPy ms | OpenCL residente / completo ms | CUDA residente / completo ms |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for dtype, n in (("float32", 1024), ("float32", 4096), ("float32", 8192), ("float64", 8192)):
        for method in ("cg", "jacobi", "sd"):
            cid = f"{method}-{n}x{n}-{dtype}-k1"
            selected_rows = [r for r in comparisons if r["case_id"] == cid]
            if not selected_rows:
                continue
            values = [f"{method} {n} {dtype}", f"{selected_rows[0]['cpu_s'] * 1000:.3f}"]
            for backend in LABELS:
                modes = {
                    r["mode"]: r["gpu_s"] * 1000 for r in selected_rows if r["backend"] == backend
                }
                values.append(f"{modes['resident']:.3f} / {modes['end_to_end']:.3f}")
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "![Escala de solvers](solver-speedup.png)", ""])
    lines.extend(
        [
            "",
            "![Producto matricial](matmul-speedup.png)",
            "",
            "![Distintas cargas](workload-speedup.png)",
            "",
            "## Separar coste de transferencia y coste de ejecución",
            "",
            (
                "`resident` conserva los arrays grandes en GPU. Incluye despacho Python, temporales, asignación/liberación "
                "y sincronización. OpenPacho también sube operandos escalares: el ensayo de 100 pasos incluye "
                "200 pequeñas subidas escalares. No es tiempo exclusivo del kernel."
            ),
            "",
            (
                "`upload` y `download` miden las APIs completas por separado. El upload de OpenPacho incluye una copia "
                "contigua de host y creación del buffer. La prueba de transporte usa buffers existentes y no mide lo mismo. "
                "No se asume que sumar medianas de fases reconstruya exactamente la mediana end-to-end."
            ),
            "",
            (
                "CuPy conserva su pool de memoria predeterminado; OpenPacho crea buffers OpenCL nuevos. Las diferencias "
                "son parte del coste de usar las librerías actuales, no una prueba de inferioridad del estándar OpenCL."
            ),
            "",
            "![Ancho de banda de copias](transfer-bandwidth.png)",
            "",
            "## Condiciones y reproducibilidad",
            "",
            f"- Sistema: `{metadata['platform']}`; Python `{metadata['python']}`.",
            f"- CPU: `{metadata['processor']}`; {metadata['logical_cpus']} procesadores lógicos. Se controló BLAS con threadpoolctl.",
            f"- GPU OpenCL: `{metadata.get('selected_opencl_device', 'no seleccionado')}`.",
            f"- GPU CUDA: `{metadata.get('cuda_device', {})}`.",
            f"- Versiones: `{metadata['versions']}`.",
            f"- Runtime CUDA: `{metadata.get('cuda_runtime')}`; TF32 override: `{metadata.get('cuda_tf32_override')}`.",
            f"- Configuraciones: {len(metadata['case_order'])}; filas crudas: {len(raw['rows'])}; errores: {len(raw['errors'])}.",
            (
                f"- Rondas por fila: {sorted({len(r['samples_s']) for r in raw['rows']})}; bloques adaptativos hasta {metadata['arguments']['max_batch']} llamadas. "
                "Se sincroniza cada llamada, incluso dentro del bloque; cada muestra es su tiempo medio."
            ),
            "- Orden de configuraciones y filas aleatorizado con semilla registrada; validación fuera del reloj.",
            (
                "- Transporte: orden fijo ascendente de tamaños, OpenCL seguido de CUDA; útil como diagnóstico, "
                "sin atribuir significación estadística a pequeñas diferencias de ancho de banda."
            ),
            f"- Fuentes sin cambios durante el ensayo: `{metadata['source_unchanged']}`; hashes en raw.json.",
            f"- Duración de la suite: {metadata['elapsed_s']:.1f} s; telemetría GPU al inicio, durante y al final en raw.json.",
            "",
            (
                "Casos: suma, reducción, funciones trascendentes, distancia, secuencias de 1/10/100 pasos, "
                "matmul cuadrado/rectangular, matvec, transposición materializada y los tres solvers. "
                "Float32 y float64 se comparan por separado. El máximo vector tiene 16.777.216 elementos; "
                "matmul llega a 2048² y matvec/transposición a 4096². La extensión de solvers llega a 8192 variables "
                "(512 MiB por matriz float64) con 7 rondas y una llamada por muestra."
            ),
            "",
            (
                "Los solvers usan la misma recurrencia, verificaciones de finitud y criterio de residuo real. "
                "La matriz SPD tiene diagonal 2.5 y sub/superdiagonal -1, almacenada en formato denso. "
                "No representa todos los condicionamientos ni problemas dispersos. Iteraciones y residuos se guardan "
                "en las filas de solvers GPU residentes; las filas end-to-end también validan el resultado. "
                "NumPy es un solver iterativo equivalente, no solve directo cronometrado."
            ),
            "",
            "## Inicialización y primera llamada",
            "",
        ]
    )
    if cold:
        lines.extend(
            [
                (
                    "Procesos nuevos; caché CuPy existente conservada y caché de disco OpenCL deshabilitada en este ensayo. La métrica suma imports/inicialización "
                    "y el primer matmul con transferencias; excluye generación de inputs y validación. "
                    "Son tres repeticiones y no deben interpretarse como tiempos estables de arranque del sistema."
                ),
                "",
                "| Motor | dtype | Inicialización mediana ms | Primera operación ms | Suma mediana ms |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for backend in ("numpy-1", "numpy-8", "numpy-16", "opencl", "cuda"):
            for dtype in ("float32", "float64"):
                rs = [r for r in cold["rows"] if r["backend"] == backend and r["dtype"] == dtype]
                med = lambda key, rows=rs: np.median([r[key] for r in rows]) * 1000
                lines.append(
                    f"| {backend} | {dtype} | {med('initialization_s'):.1f} | {med('first_end_to_end_s'):.1f} | {med('startup_plus_first_s'):.1f} |"
                )
    else:
        lines.append(
            "El primer uso dentro de la suite está registrado en first_call_s; no equivale a proceso frío."
        )
    lines.extend(
        [
            "",
            "## Archivos",
            "",
            "- `raw.json`: todas las muestras, metadatos, validaciones, hashes y errores.",
            "- `comparisons.csv`: cada GPU/escenario frente a la mejor configuración NumPy medida.",
            "- `timings.csv`: medianas, dispersión, bytes y fases de todas las filas.",
            "- Figuras PNG y SVG: resultados exportables.",
            "",
            (
                "El resultado solo describe este equipo, este código y estas entradas. No demuestra que todas las "
                "GPUs ganen ni que CUDA sea intrínsecamente superior a OpenCL. Para el diseño de temporización se consultó "
                "[la documentación oficial de CuPy](https://docs.cupy.dev/en/stable/user_guide/performance.html), "
                "especialmente sincronización, primera llamada y asignación de memoria."
            ),
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cold", type=Path)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.raw.resolve() == (args.output / "raw.json").resolve():
        parser.error("use a separate report directory to preserve the original raw measurements")
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    original_metadata = json.loads(json.dumps(raw["metadata"]))
    raw["metadata"]["runs"] = [{"path": str(args.raw), "metadata": original_metadata}]
    for extra_path in args.extra:
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        for path, digest in raw["metadata"]["source_sha256"].items():
            if path.startswith("src/") and extra["metadata"]["source_sha256"].get(path) != digest:
                raise ValueError(f"Runtime source differs between runs: {path}")
        raw["rows"].extend(extra["rows"])
        raw["errors"].extend(extra["errors"])
        raw["metadata"]["case_order"].extend(extra["metadata"]["case_order"])
        raw["metadata"]["elapsed_s"] += extra["metadata"]["elapsed_s"]
        raw["metadata"]["source_unchanged"] &= extra["metadata"]["source_unchanged"]
        raw["metadata"]["runs"].append({"path": str(extra_path), "metadata": extra["metadata"]})
    comparisons = summarize(raw)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.json").write_text(
        json.dumps(raw, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_csv(args.output / "comparisons.csv", comparisons)
    flattened = [
        {k: v for k, v in row.items() if k not in ("samples_s", "validation")}
        for row in raw["rows"]
    ]
    write_csv(args.output / "timings.csv", flattened)
    plot_matmul(comparisons, args.output)
    plot_workloads(comparisons, args.output)
    plot_transfers(raw, args.output)
    plot_solvers(comparisons, args.output)
    cold = json.loads(args.cold.read_text(encoding="utf-8")) if args.cold else None
    if cold:
        (args.output / "cold-start.json").write_text(json.dumps(cold, indent=2), encoding="utf-8")
    build_report(raw, comparisons, args.output, cold)
    print(
        json.dumps(
            {
                f"{b}/{m}": dict(
                    Counter(
                        r["classification"]
                        for r in comparisons
                        if r["backend"] == b and r["mode"] == m
                    )
                )
                for b in LABELS
                for m in ("resident", "end_to_end")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
