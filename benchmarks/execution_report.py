"""Render synchronized execution measurements into Markdown, CSV and a static plot."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(inputs, output, artifacts):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    rows = [row for report in reports for row in report["rows"]]
    if any(
        "error" in row or len(row["resident_seconds"]) != 7 or len(row["end_to_end_seconds"]) != 7
        for row in rows
    ):
        raise ValueError("This report requires complete, valid seven-round measurements")
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / "measurements.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "case",
                "size",
                "dtype",
                "implementation",
                "mode",
                "median_ms",
                "p25_ms",
                "p75_ms",
                "samples_ms",
            ]
        )
        for row in rows:
            for mode in ("resident", "end_to_end"):
                values = np.array(row[f"{mode}_seconds"]) * 1000
                writer.writerow(
                    [
                        row["case"],
                        row["size"],
                        row["dtype"],
                        row["implementation"],
                        mode,
                        *np.quantile(values, [0.5, 0.25, 0.75]),
                        json.dumps(values.tolist()),
                    ]
                )
    keys = list(dict.fromkeys((r["case"], r["size"], r["dtype"]) for r in rows))
    names = ["opencl", "opencl-clblast", "vulkan", "cupy", "cublas", "nvmath"]
    labels = ["OpenCL custom", "OpenCL CLBlast", "Vulkan", "CuPy", "cuBLAS", "nvmath"]

    def group(key):
        return {r["implementation"]: r for r in rows if (r["case"], r["size"], r["dtype"]) == key}

    def selected(key, name):
        suffix = "-solver" if key[0].endswith("_solve") else "-compiled"
        return group(key).get(name + suffix if name in names[:3] else name)

    def milliseconds(case, size, implementation=None, mode="resident"):
        data = group((case, size, "float32"))
        value = (
            min(
                row[f"{mode}_median_seconds"]
                for name, row in data.items()
                if name.startswith("numpy")
            )
            if implementation is None
            else data[implementation][f"{mode}_median_seconds"]
        )
        return f"{value * 1000:.3f}"

    m = milliseconds
    solver_count = sum(row["case"].endswith("_solve") for row in rows)
    lines = [
        "# Compilador, Vulkan y referencias CUDA: resultados locales",
        "",
        (
            f"Plataforma: {reports[0]['platform']}. Dispositivos: {reports[0]['devices']}. "
            "La sesión original del 2026-09-06 corresponde a Windows 11, RTX 5060 Ti "
            "16 GB (driver 616.64), Ryzen 7 5700X3D y 32 GB RAM. "
            "No se ha validado AMD/Intel."
        ),
        "",
        (
            f"**{len(rows)} configuraciones, siete rondas por modo, cero errores de validación.** "
            f"{len(rows) - solver_count} de operaciones/pasos y {solver_count} de solvers completos. "
            "Los JSON conservan todas las muestras; el CSV añade cuartiles."
        ),
        "",
        "## Qué cambia y qué sigue costando",
        "",
        (
            f"100 actualizaciones de 1 048 576 valores float32: {m('updates100', 1048576)} ms "
            f"en el mejor NumPy, {m('updates100', 1048576, 'opencl-eager')} ms en OpenCL directo, "
            f"{m('updates100', 1048576, 'opencl-compiled')} ms con compile y "
            f"{m('updates100', 1048576, 'cupy')} ms en CuPy. "
            f"Incluyendo transferencias: {m('updates100', 1048576, mode='end_to_end')} ms "
            f"NumPy y {m('updates100', 1048576, 'opencl-compiled', 'end_to_end')} ms OpenCL compilado."
        ),
        "",
        (
            f"Matmul 2048×2048 float32: {m('matmul', 2048)} ms en NumPy, "
            f"{m('matmul', 2048, 'opencl-clblast-compiled')} ms OpenCL+CLBlast, "
            f"{m('matmul', 2048, 'cublas')} ms cuBLAS y {m('matmul', 2048, 'nvmath')} ms nvmath. "
            "La ventaja principal aquí viene del proveedor matricial: compilar un matmul "
            "aislado no lo fusiona ni mejora necesariamente su kernel."
        ),
        "",
        (
            f"Jacobi completo a 2048 float32: {m('jacobi_solve', 2048)} ms NumPy, "
            f"{m('jacobi_solve', 2048, 'opencl-solver')} ms OpenCL custom y "
            f"{m('jacobi_solve', 2048, 'cupy')} ms CuPy. Descenso más pronunciado: "
            f"{m('sd_solve', 2048)} ms NumPy, {m('sd_solve', 2048, 'opencl-solver')} ms OpenCL y "
            f"{m('sd_solve', 2048, 'cupy')} ms CuPy. Son tiempos residentes: la transferencia "
            "no es la única causa de las diferencias. Hay GEMV/DOT, despacho, asignaciones "
            "y sincronización para convergencia. CLBlast por sí solo no resuelve el algoritmo."
        ),
        "",
        (
            f"Vulkan: matmul 2048 float32 residente {m('matmul', 2048, 'vulkan-compiled')} ms, "
            f"con transferencias {m('matmul', 2048, 'vulkan-compiled', 'end_to_end')} ms; "
            f"100 actualizaciones grandes, {m('updates100', 1048576, 'vulkan-compiled')} ms. "
            "Es funcional; faltan pool, fusión y menor coste de envío. Estos números "
            "no representan un límite de Vulkan como tecnología."
        ),
        "",
        (
            f"100 actualizaciones de solo 256 valores: {m('updates100', 256)} ms NumPy y "
            f"{m('updates100', 256, 'opencl-compiled')} ms OpenCL compilado. La corrección "
            "se conserva; no se prioriza hacer que GPU gane este caso."
        ),
        "",
        "## Método y límites de comparación",
        "",
        (
            "- Tiempo de pared sincronizado, con despacho Python y asignación de resultados. "
            "Rondas con orden aleatorio de implementaciones. No se usan tiempos de lanzamiento asincrónico como cómputo completo."
        ),
        (
            "- Residente: inputs ya cargados, planes preparados y kernels calentados. "
            "Cada repetición reinicia el estado lógico de las iteraciones. "
            "La validación y descarga del resultado quedan fuera de este tiempo."
        ),
        (
            "- Con transferencias: nuevos buffers de entrada, cómputo y descarga. "
            "Se reutilizan planes/programas; nvmath reenlaza operandos sin planificar otra vez. "
            "NumPy usa los arrays host existentes: no se le añade una copia artificial."
        ),
        (
            "- Los JSON separan subida, planificación, primera ejecución y descarga. "
            "La primera ejecución es de esa configuración dentro del proceso; "
            "puede reutilizar cachés de configuraciones anteriores. No mide un proceso "
            "frío independiente ni incluye inicialización del dispositivo. "
            "La captura de OpenPacho está incluida en su primera ejecución; la "
            "planificación explícita de nvmath figura separada."
        ),
        (
            "- NumPy se mide con BLAS limitado a 1, 8 y 16 hilos. Las tablas muestran "
            "la mejor mediana por caso y modo. Las operaciones sin BLAS no se "
            "paralelizan por cambiar ese límite; diferencias pequeñas ahí son ruido."
        ),
        (
            "- float32/float64 originales, TF32 deshabilitado; nvmath usa COMPUTE_32F/64F. "
            "Se comprueban forma, dtype y valores en cada ronda. No se exige identidad bit a bit."
        ),
        (
            "- `updates100`: x=(1-tasa)*x+tasa*b, cien pasos. `energy`: "
            "sum((2*(a*b+c))²,axis=-1), arrays tamaño×32. Jacobi20 y gradient20 usan "
            "20 pasos de una matriz SPD densa dominante; gradient20 usa tasa fija."
        ),
        (
            "- Los casos `_solve` ejecutan los solvers de examples/solvers.py hasta convergencia, "
            "con residuo real en cada iteración, rtol=1e-4/1e-7 para float32/64 y "
            "maxiter=400. El descenso más pronunciado usa búsqueda exacta de paso. "
            "NumPy/CuPy ejecutan la misma recurrencia mediante operaciones nativas "
            "de `benchmarks.suite.native_solver`. Los coeficientes escalares pueden "
            "tener rutas/precisión intermedia diferentes. Se valida además el residuo host "
            "contra la tolerancia (margen de 2 % por orden de reducción). "
            "Los inputs son sistemas tridiagonales almacenados densamente; no es una prueba sparse."
        ),
        (
            "- CUDA es la plataforma. La referencia concreta cuBLAS usa bindings directos; "
            "CuPy matmul también puede usar cuBLAS y nvmath utiliza cuBLASLt. "
            "No son tres dispositivos independientes. cuBLAS/nvmath se comparan solo "
            "en matmul; no se inventa un solver de alto nivel para esas interfaces."
        ),
        (
            "- No se midieron PyTorch/JAX, uso máximo de memoria, energía, todas las "
            "formas rectangulares ni portabilidad de rendimiento entre fabricantes. "
            "Esta batería cubre el alcance actual; no constituye todos los workloads posibles."
        ),
        "",
    ]
    lines += [
        "## Tiempos residentes",
        "",
        (
            "Milisegundos; menor es mejor. OpenPacho usa compile en operaciones/pasos; "
            "en `_solve`, el solver medido ya compila sus actualizaciones. "
            "Los modos eager y compilado completos están en el JSON/CSV."
        ),
        "",
    ]
    for mode in ("resident", "end_to_end"):
        if mode == "end_to_end":
            lines += [
                "## Carga + cómputo + descarga",
                "",
                "Milisegundos; planes y programas calientes.",
                "",
            ]
        lines += [
            "| Caso / tamaño / dtype | NumPy mejor | " + " | ".join(labels) + " |",
            "| --- | " + " | ".join(["---:"] * 7) + " |",
        ]
        for key in keys:
            data = group(key)
            cpu = min(
                row[f"{mode}_median_seconds"]
                for name, row in data.items()
                if name.startswith("numpy")
            )
            values = [f"{1000 * cpu:.3f}"]
            for name in names:
                row = selected(key, name)
                values.append(f"{1000 * row[f'{mode}_median_seconds']:.3f}" if row else "—")
            lines.append(f"| {key[0]} / {key[1]} / {key[2]} | " + " | ".join(values) + " |")
        lines.append("")
    lines += [
        "## Prioridad siguiente",
        "",
        (
            "1. Reducir sincronizaciones y coste de ejecución en solvers sin cambiar "
            "silenciosamente su frecuencia de comprobación o tolerancias; optimizar GEMV/DOT."
        ),
        (
            "2. Fusionar map-reduce y mejorar decisiones sobre temporales compartidos. "
            "Medir un ejecutor con menor overhead cuando domine Python."
        ),
        (
            "3. Pool y fusión Vulkan, reutilización de descriptores/comandos y kernels "
            "matriciales optimizados. Verificar un segundo fabricante."
        ),
        (
            "4. Ampliar operaciones frecuentes, como matmul por lotes, según casos reales. "
            "Mantener CLBlast y kernels custom intercambiables por contrato; "
            "CUDA backend y dispersas siguen como trabajo futuro."
        ),
        "",
        "## Reproducir",
        "",
        "```powershell",
        "$env:PYOPENCL_NO_CACHE='1'",
        "$env:PACHO_VULKAN_COMPILER='<ruta a glslang o glslc>'",
        "$env:OPENPACHO_CLBLAST_LIBRARY='<ruta a CLBlast>'",
        "python -m benchmarks.execution --repeats 7 --output .artifacts/benchmarks/execution/raw.json",
        "python -m benchmarks.execution --solvers-only --repeats 7 --output .artifacts/benchmarks/execution-solvers/raw.json",
        "python -m benchmarks.execution_report --inputs .artifacts/benchmarks/execution/raw.json .artifacts/benchmarks/execution-solvers/raw.json --output docs/execution-results.md --artifacts .artifacts/benchmarks/execution-report",
        "```",
        "",
        (
            "Dependencias adicionales: extra benchmark, CuPy y nvmath-python con componentes "
            "CUDA compatibles, CLBlast nativo, bindings Vulkan y compilador de shaders. "
            "No forman parte de las dependencias mínimas CPU de OpenPacho."
        ),
        "",
        "Versiones medidas: "
        + ", ".join(f"{name} {version}" for name, version in reports[0]["versions"].items())
        + ".",
        "Los artefactos locales originales están en "
        + ", ".join(str(p.parent).replace("\\", "/") for p in inputs)
        + ".",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    # Full matrix of measured speedups; log colors expose both wins and losses.
    plot_names = names[:4]
    ratios = []
    for key in keys:
        data = group(key)
        best = min(
            row["resident_median_seconds"] for name, row in data.items() if name.startswith("numpy")
        )
        ratios.append(
            [best / selected(key, name)["resident_median_seconds"] for name in plot_names]
        )
    ratios = np.array(ratios)
    fig, ax = plt.subplots(figsize=(10, 12))
    chart = ax.imshow(np.log2(ratios), cmap="RdYlGn", vmin=-10, vmax=6, aspect="auto")
    ax.set_xticks(range(4), labels[:4], rotation=15)
    ax.set_yticks(range(len(keys)), [f"{c} · {n} · {dt}" for c, n, dt in keys], fontsize=9)
    for i in range(len(keys)):
        for j in range(4):
            ax.text(j, i, f"{ratios[i, j]:.3g}×", ha="center", va="center", fontsize=9)
    ax.set_title(
        "Velocidad residente respecto al mejor NumPy\n>1× gana; <1× pierde · 7 rondas · RTX 5060 Ti",
        pad=20,
    )
    fig.colorbar(chart, ax=ax, label="log₂(velocidad relativa a NumPy)", shrink=0.65)
    fig.tight_layout()
    fig.savefig(artifacts / "resident-speedups.png", dpi=160)
    fig.savefig(artifacts / "resident-speedups.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    render(args.inputs, args.output, args.artifacts)
