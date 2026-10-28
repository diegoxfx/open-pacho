# Resultados de benchmarks — 2026-09-06

**La hipótesis se cumple en muchos casos, pero no como afirmación universal.**
OpenPacho y CuPy ganaron en determinadas cargas aun incluyendo transferencias.
Jacobi y descenso por gradiente perdieron en tamaños menores y ganaron a 8192
variables en los sistemas de prueba.

Equipo: Ryzen 7 5700X3D (8 núcleos, 16 hilos), aproximadamente 32 GB RAM,
RTX 5060 Ti de 16 GB, Windows 11. NumPy 2.5.3/OpenBLAS 0.3.34.106.0,
PyOpenCL 2026.1.4, CuPy 14.2.0. CuPy usa CUDA como control independiente;
OpenPacho sigue ejecutando su backend OpenCL. TF32 deshabilitado.

Una comprobación posterior bajo copias repetidas de 64 MiB confirmó un enlace
activo **PCIe Gen3 ×8** en este equipo; las copias grandes medidas rondaron
6–7 GB/s. Esta configuración condiciona el coste de transferencia y no representa
todos los enlaces posibles de esta GPU. Evidencia local:
`.artifacts/benchmarks/report-2026-09-06/pcie-probe.json` (no distribuido).

Se probaron **140 configuraciones**, con **1540 series de tiempos**: 122 casos
generales de 11 rondas y 18 casos adicionales de solvers de 7 rondas. Todas las
configuraciones pasaron su validación numérica. Se comparó contra el mejor
NumPy medido entre 1, 8 y 16 hilos, sin cambiar el código de la librería.

## Tiempos con subida, cálculo y descarga

Medianas en milisegundos. Runtime inicializado; generación de datos y validación
fuera del reloj. La CPU ya recibe y devuelve arrays de host, así que no se añade
una copia artificial a su baseline.

| Caso | NumPy | OpenPacho/OpenCL | CuPy/CUDA |
| --- | ---: | ---: | ---: |
| Suma de 16.777.216 elementos, float32 | **14.11** | 94.28 | 43.52 |
| Secuencia sin/exp/cos de 16.777.216 elementos, float32 | 139.79 | 98.16 | **39.74** |
| Matmul 2048×2048, float32 | 29.81 | 48.26 | **13.13** |
| Matmul 2048×2048, float64 | **66.15** | 116.76 | 102.50 |
| Jacobi 1024 variables, float32 | **4.34** | 53.51 | 19.37 |
| Jacobi 8192 variables, float32 | 385.58 | 276.98 | **119.18** |
| Descenso por gradiente 1024 variables, float32 | **4.32** | 68.95 | 35.92 |
| Descenso por gradiente 8192 variables, float32 | 554.52 | 335.75 | **155.56** |
| Jacobi 8192 variables, float64 | 1440.83 | 751.98 | **299.05** |
| Descenso por gradiente 8192 variables, float64 | 2382.81 | 1130.29 | **438.74** |

Los métodos usaron idénticas recurrencias y tolerancias, y los solvers residentes
registraron el mismo número de iteraciones que el NumPy de referencia. Los datos
son una matriz SPD de diagonal 2.5 y sub/superdiagonal -1, **almacenada y procesada
como densa**. No se está comparando contra un solver tridiagonal o disperso
especializado; estos resultados no describen todos los sistemas o condicionamientos.

## Qué explica las diferencias

1. **Transferencias y preparación.** En una suma grande hay poco cálculo por byte.
   La GPU puede calcular rápido y perder la comparación completa al sumar copias,
   asignaciones y preparación de arrays.
2. **Despacho y control por iteración.** Jacobi de 1024 variables en float32 tarda
   48.41 ms en OpenPacho y 19.02 ms en CuPy incluso con datos residentes, frente a
   4.34 ms de NumPy. La subida de la matriz no explica esa pérdida. Siguen existiendo
   kernels pequeños, temporales y sincronizaciones para extraer escalares.
3. **Amortización.** Al crecer los sistemas de prueba, el trabajo sobre los mismos
   datos compensa esos costes. A 8192 variables la ventaja persiste al incluir las
   transferencias. Esto refuta “nunca puede ganar”, sin prometer ganancias generales.
4. **Implementación.** CuPy usa bibliotecas y kernels optimizados y un pool de
   memoria. OpenPacho usa kernels sencillos y buffers nuevos. La diferencia entre
   ellos no demuestra una superioridad intrínseca de CUDA sobre OpenCL.

En la cuadrícula completa, OpenPacho mostró una ventaja end-to-end clara en
**15/140** configuraciones; CuPy en **33/140**. “Clara” exige al menos 10% de ventaja
mediana e intervalo bootstrap descriptivo de 95% sin cruzar 1. Los recuentos dependen
de los casos elegidos, no son probabilidades de ganar en aplicaciones reales.

## Arranque

En procesos nuevos, para una única multiplicación 1024×1024 float32, la suma de
imports/inicialización y primera operación costó aproximadamente **0.233 s en
NumPy**, **1.694 s en OpenPacho** y **1.216 s en CuPy**. Aquí NumPy ganó claramente.
Son tres procesos por motor y dtype, con las cachés de disco existentes conservadas
en CuPy y caché de disco OpenCL deshabilitada por restricciones del entorno.

## Reproducir y revisar

El diseño y los comandos están en [benchmarks/README.md](../benchmarks/README.md).
La salida local completa está en
`.artifacts/benchmarks/report-2026-09-06/REPORT.md`, junto a `raw.json`,
`comparisons.csv`, `timings.csv`, datos de arranque y gráficas PNG/SVG. Estos archivos
locales no forman parte del código distribuido.

Se guardaron snapshots del runner en cada carpeta de mediciones. Sus hashes
coinciden con los registrados: el runner se amplió con opciones para solvers
grandes después del primer ensayo. Los **nueve archivos del runtime de OpenPacho
permanecieron idénticos** entre ambos ensayos y durante cada ejecución.

El código del runner pasó 7 pruebas de semántica, validación y sincronización.
Los archivos crudos conservan cada réplica y los fallos (ninguno), así como versiones,
configuración BLAS, semilla, orden y telemetría. No se midieron otras GPUs,
solvers dispersos especializados, kernels fusionados ni todos los condicionamientos.
