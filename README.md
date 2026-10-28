# OpenPacho

**Arrays en GPU independientes del fabricante, con una API familiar para quien usa NumPy.**

OpenPacho retoma la idea de operaciones numéricas paralelas en GPU explorada en
PyPacho, con una implementación nueva. OpenCL y Vulkan son motores
de GPU seleccionables: no necesitan CUDA. NumPy es el
motor de CPU y la referencia de comportamiento. Versión **0.1.0a1**, experimental.
También existe un [backend CUDA inicial](docs/cuda.md) para NVIDIA, con alcance
menor que OpenCL/Vulkan y proveedores cuBLAS/custom seleccionables.

Consulta la [incorporación de normas estables CUDA](docs/cuda-norm-results.md), las
[tres iteraciones de kernels y layouts](docs/three-iterations-results.md) y el
[camino a la primera versión estable](docs/release-readiness.md), con criterios
de salida y límites de la validación actual.

## Instalar

Requiere Python 3.10 o posterior. Desde este repositorio:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install ".[opencl]"
pacho devices
```

El paquete todavía no está publicado en PyPI. Para CPU basta `python -m pip install .`.
Para contribuir: `python -m pip install -e ".[opencl,dev,benchmark]"`.
Los extras `opencl`, `vulkan` y `cuda` son independientes y combinables:
`python -m pip install ".[opencl,vulkan]"` no instala CUDA. La distribución se llama
`open-pacho` y el import `openpacho`; cuando se publique, el equivalente será
`python -m pip install "open-pacho[opencl]"`.

OpenCL requiere el controlador/runtime del fabricante además de PyOpenCL. Los
dispositivos disponibles dependen del sistema: NVIDIA, AMD, Intel y otros proveedores
pueden exponer OpenCL; no se presupone que todo equipo lo tenga. En dispositivos
sin doble precisión, usa `dtype=pacho.float32`: **nunca se reduce la precisión sin avisar**.
Consulta la [instalación de PyOpenCL](https://documen.tician.de/pyopencl/misc.html#installation).

Para Vulkan: `python -m pip install ".[vulkan]"`, un controlador Vulkan y un
compilador GLSL/SPIR-V; consulta [instalación y límites](docs/vulkan.md).

Para CUDA inicial: `python -m pip install ".[cuda]"` y un driver NVIDIA compatible
con CUDA 13. Usa `pacho devices --backend cuda`; consulta el [alcance CUDA](docs/cuda.md).

## Migrar una función NumPy

Puedes conservar NumPy dentro de funciones existentes. Los protocolos de despacho
de NumPy dirigen las operaciones compatibles a OpenPacho:

```python
import numpy as np
import openpacho as pacho


def distancias(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


with pacho.use_backend("opencl"):
    a = pacho.asarray([[1, 2, 3], [4, 5, 6]], dtype=pacho.float32)
    b = pacho.asarray([1, 1, 1], dtype=pacho.float32)
    resultado = distancias(a, b)  # las operaciones permanecen en GPU
    print(resultado.device)
    print(pacho.asnumpy(resultado))  # transferencia explícita al final
```

También puedes cambiar el import en código que use el subconjunto compatible:

```python
import openpacho as np

np.set_backend("opencl")
a = np.asarray([[4, 1], [1, 3]], dtype=np.float32)
b = np.asarray([1, 2], dtype=np.float32)
resultado = a @ b
print(np.asnumpy(resultado))
```

El backend inicial es `numpy`, sin inicialización de GPU al importar. `set_backend`
cambia dónde se crean los próximos arrays; `use_backend` restaura la selección al salir.
Los arrays existentes conservan su dispositivo. Para escoger uno concreto:

```python
print(pacho.devices())
pacho.set_backend("opencl", device="0:0")  # ID de plataforma:dispositivo
```

La selección OpenCL predeterminada exige una GPU. Un dispositivo OpenCL de CPU se
puede seleccionar explícitamente por ID. No hay fallback automático cuando falla GPU.

## Qué incluye

- Arrays de cualquier dimensión con broadcasting, operaciones escalares y promoción de tipos de NumPy 2.
- `+`, `-`, `*`, `/`, potencia flotante, comparaciones y funciones `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `abs`, `isfinite`.
- `where(condición, x, y)` con broadcasting de los tres argumentos; selecciona en el backend sin descargar. Consulta los [límites de selección](docs/compatibility.md#selección-condicional).
- `sum`, `mean`, `min`, `max`, reshape, transposición, índices enteros y slices, diagonales, `dot` para vectores/matrices y `@`/`matmul` por lotes con broadcasting de dimensiones iniciales.
- Tipos `float32`, `float64`, `int32`, `int64`, `uint32`, `uint64`, `bool` según la operación y las capacidades del dispositivo.
- Normas euclídeas/Frobenius escaladas para evitar cuadrados intermedios extremos.
- [Generación aleatoria residente](docs/random.md) con `pacho.random.default_rng(seed, backend=...)`: uniforme, normal y exponencial; semilla reproducible y parámetros escalares.
- CLI de dispositivos y benchmarks reproducibles con sincronización y validación contra NumPy.
- Compilación opcional con `pacho.compile(f)`, `@pacho.compile` y `@pacho.compile()`: planes cacheados/preparados, salidas múltiples y fusión elemento a elemento/map-reduce en OpenCL, Vulkan y CUDA para los grupos soportados; consulta [el contrato y ejemplos](docs/compilation.md) y los [resultados de fusión CUDA](docs/cuda-fusion-results.md).
- Memoria reutilizable OpenCL/Vulkan y CLBlast opcional con GEMM, GEMM batched, GEMV y DOT flotantes; consulta [ejecución y rendimiento](docs/performance.md).
- Menor coste de llamadas compiladas y GEMV/DOT custom cooperativos CUDA; consulta las [tres optimizaciones medidas](docs/three-actions-results.md), incluyendo límites y errores numéricos.
- [Padding opcional](docs/padding.md) con `@pacho.compile(padding=True)` o `pacho.matmul(a, b, padding=True)`: buffers alineados para GEMM custom float32 en OpenCL, Vulkan y CUDA, con coste de memoria y copias explícito; desactivado por defecto.

La [ronda de staging Vulkan y aleatorios](docs/staging-random-results.md) documenta
la selección de mapeos de lectura, el RNG residente y sus controles de tiempo/error.

El [análisis de gaps y optimización](docs/gaps-and-optimization.md) distingue
limitaciones del hardware, costes del runtime y oportunidades pendientes.
La [iteración de Vulkan, GEMM y estabilidad](docs/numerics-vulkan-results.md)
presenta mejoras medidas y límites pendientes. Los benchmarks publican
[error numérico y casos adversos](docs/numerical-accuracy.md), además de tiempos.
La [continuación de solvers y aplicaciones](docs/solver-applications-results.md)
amplía las mediciones y prepara el [camino CUDA](docs/cuda-backend-plan.md).
La [ronda CUDA inicial](docs/cuda-initial-results.md) implementa CUDA y mejora
la reutilización Vulkan cuando cambian las formas. La [ronda de dispatch](docs/vulkan-dispatch-results.md)
reduce conversiones Python/CFFI y publica diagnóstico GPU/host con un control alternado.
La [ampliación entre backends](docs/backend-sharing-results.md) incorpora push
descriptors Vulkan por capacidad, metadatos/índices OpenCL y lotes/tiles CUDA.
La [ronda de kernels y estabilidad](docs/kernel-stability-results.md) añade reposo
observado entre proveedores, controles de variantes y guardas especializadas OpenCL.
La [inspección de código generado](docs/kernel-inspection-results.md) corrige una
promoción accidental a float64 en GEMM OpenCL, revisa las copias y distingue
evidencia PTX/SPIR-V/SASS de hipótesis sobre el hardware.
La [reevaluación de padding](docs/padding-reassessment.md) conserva la opción por
ganancias concretas en productos rectangulares, manteniéndola desactivada por
defecto y documentando las formas donde pierde rendimiento.
La [comparación del compilador, Vulkan y referencias CUDA](docs/execution-results.md)
separa ejecución residente, transferencias y primera ejecución.
La [primera comparación de optimizaciones](docs/optimization-results.md) incluye
la versión anterior a esa etapa, NumPy y los caminos medidos en la misma sesión.

**No buscamos compatibilidad 100 % con NumPy.** Priorizamos cálculos matriciales
frecuentes con mejoras medibles y un coste de implementación razonable. La [tabla de compatibilidad](docs/compatibility.md)
documenta las diferencias: sin escritura por índice, `out=`, indexación avanzada,
arrays complejos ni álgebra dispersa. Las funciones no soportadas
fallan explícitamente. `np.asarray(array_pacho)` no copia silenciosamente a CPU;
usa `pacho.asnumpy` cuando una dependencia necesite un ndarray de NumPy.

## Aplicaciones de ejemplo

La librería ofrece operaciones matriciales, no solvers numéricos. CG, Jacobi,
descenso más pronunciado y sistemas preparados viven en [examples](examples/README.md),
fuera del paquete instalable. Desde el repositorio:

```python
import openpacho as pacho
from examples.solvers import conjugate_gradient

with pacho.use_backend("opencl"):
    A = pacho.asarray([[4, 1], [1, 3]], dtype=pacho.float32)
    b = pacho.asarray([1, 2], dtype=pacho.float32)
    result = conjugate_gradient(A, b, rtol=1e-5, maxiter=100)
    print(result.converged, result.iterations, result.residual_norm)
    print(pacho.asnumpy(result.x))
```

CG y descenso más pronunciado requieren una matriz simétrica definida positiva;
Jacobi requiere diagonal no nula y una iteración convergente. Los ejemplos no prueban
estas propiedades. El criterio de parada usa el residuo real
`||b - A @ x|| <= max(atol, rtol * ||b||)`. No converger devuelve `converged=False`
y una razón; no se presenta el último iterado como una solución garantizada.

## Verificar y medir

```sh
python -m pytest
python examples/numpy_migration.py --backend opencl
pacho benchmark --backend opencl --size 256 --repeat 5 --output benchmark.json
pacho benchmark --backend numpy --size 256 --repeat 5
python -m build
```

Las pruebas OpenCL se omiten si no existe el runtime/dispositivo. Para que la
ausencia de GPU sea un fallo de la verificación:

```powershell
$env:PACHO_REQUIRE_OPENCL = "1"
python -m pytest
```

En Linux: `PACHO_REQUIRE_OPENCL=1 python -m pytest`. `PACHO_OPENCL_DEVICE=0:0`
permite seleccionar un dispositivo de prueba concreto.

Los benchmarks separan transferencia inicial, primera ejecución (incluida compilación),
operaciones con datos residentes y descarga. No se promete una aceleración universal:
el tamaño del problema, la precisión, el controlador y el coste de despacho importan.
Consulta [arquitectura actual](docs/architecture.md), [decisiones de evolución](docs/architecture-evolution.md) y [prioridades](docs/roadmap.md).
Los resultados comprobados en este equipo están en [verificación local](docs/verification.md).
La comparación extensa CPU/OpenCL/CUDA, incluyendo transferencias y solvers hasta
8192 variables, está en [resultados de benchmarks](docs/benchmark-results.md).

## Origen y reconocimiento

El proyecto original es [azapat47/pypacho](https://github.com/azapat47/pypacho), de
Andrés Felipe Zapata Palacio, Esteban Echeverri Jaramillo, Diego Alejandro Cifuentes
y Juan David Arcila Moreno. Su publicación es
[PyPacho: A Python library that implements parallel basic operations on GPUs](https://ieeexplore.ieee.org/document/9623197).
PyPacho exploró operaciones básicas paralelas en GPU y comparaciones de métodos
iterativos por plataforma, tamaño y precisión. Ese trabajo es el antecedente
de la idea que desarrolla OpenPacho.

Esta implementación tiene código nuevo y licencia MIT. La
[nota de procedencia](docs/provenance.md) distingue las referencias inspeccionadas
de lo que se ha verificado. No afirma reproducir los resultados experimentales del paper.
