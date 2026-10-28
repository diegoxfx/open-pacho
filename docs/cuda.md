# CUDA inicial

Implementación inicial del 2026-09-12, ampliada con lotes, tiling, fusión y
GEMV/DOT cooperativos; actualización de normas estables: 2026-10-22.
CUDA se selecciona explícitamente y conserva el
API de arrays de OpenPacho. Es una primera base funcional; todavía tiene menos
operaciones y optimizaciones que OpenCL/Vulkan.

```powershell
python -m pip install ".[cuda]"
pacho devices --backend cuda
pacho benchmark --backend cuda --size 129 --repeat 7
```

La instalación anterior se ejecuta desde este repositorio; el paquete todavía
no se publica en PyPI. Requiere GPU NVIDIA, driver compatible con las bibliotecas CUDA 13 instaladas y
NVRTC. El extra incorpora CUDA bindings, NVRTC, los bindings de nvmath y cuBLAS.
La importación CPU no carga estos módulos. El runtime no depende de arrays ni
operaciones de CuPy; los comparadores CuPy y nvmath Matmul siguen siendo externos.
La validación física de esta etapa corresponde a Windows y RTX 5060 Ti; Linux
y otros dispositivos todavía requieren verificación.

```python
import numpy as np
import openpacho as pacho

gpu = pacho.get_backend("cuda", device="0", blas="cublas")
a = pacho.asarray(np.ones((33, 17), dtype=np.float32), backend=gpu)
b = pacho.asarray(np.ones((17, 65), dtype=np.float32), backend=gpu)
c = (a @ b) * 0.5 + 2
host = pacho.asnumpy(c)
```

`blas="custom"` selecciona productos propios mediante NVRTC. GEMM float32 elegible
reutiliza el generador con tiles y registros de OpenCL/Vulkan: 64 hilos por tile
de salida 32×32, con 4×4 acumuladores por hilo y 4 KiB compartidos. Los límites
del dispositivo y de índices se comprueban antes de seleccionarlo. Otros tamaños
y float64 conservan el kernel directo para GEMM. Los productos con vector derecho
y dimensión interna de al menos 256 usan una reducción cooperativa de 128 hilos
por fila, también en float64 y lotes contiguos. DOT de más de 4096 elementos usa
parciales de 1024 términos y una reducción posterior en GPU. Es la adaptación
del algoritmo probado en OpenCL; no depende de extensiones de una familia NVIDIA.
No hay autotuning. El proveedor por defecto es `cublas`; `clblast`
y `clblast_library` no son configuraciones válidas de CUDA. No se cambia a CPU
si falta el driver, una biblioteca o una operación.

GEMM custom añade dos dimensiones de tile menores para matrices con pocas filas:
en CUDA, 16×16 salidas y bloques K=32. La selección es interna y conservadora;
los casos no elegibles mantienen el kernel anterior. No introduce padding,
copias adicionales, tipos reducidos ni flags matemáticos. Ver el alcance y las
[mediciones de las tres iteraciones](three-iterations-results.md).

Con `blas="custom"`, `compile(..., padding=True)` o
`matmul(..., padding=True)` permiten [alinear GEMM float32](padding.md) usando
buffers privados y recorte en GPU. Se conserva `padding=False` por defecto;
cuBLAS y los demás casos mantienen su ruta. Las copias preservan los bits de
float32 y la opción no modifica `ftz`, `fmad` ni otros flags matemáticos.

## Alcance comprobable

| Operación | Alcance |
| --- | --- |
| float32/float64; bool para resultados y almacenamiento | Admitidos; enteros y otros dtypes producen error explícito |
| Construcción, subida, descarga, copia, reshape | Admitidos; arrays propios residentes |
| Aleatorios | [Philox4x32-10](random.md) propio: random/uniform/normal/exponential float32/64, sin cuRAND ni CuPy como dependencia |
| Aritmética, comparaciones y funciones elementwise del subconjunto común | Incluye `where(condition, x, y)` con broadcasting de tres argumentos, condición bool o numérica, sin descargas; dtype promovido desde las ramas |
| Slicing básico, transposición y diagonal | Gather GPU; las salidas copian según el contrato común |
| sum/mean por ejes, keepdims | Reducción por bloques con parciales para ejes largos; acumulación flotante |
| min/max (amin/amax) por ejes, keepdims | Conservan float32/64 o bool; NaN propagado y empates de cero deterministas; eager y compile/map-reduce |
| Dot, GEMV y GEMM, incluido matmul por lotes broadcast | cuBLAS o producto custom; formas rectangulares, impares y vacías |
| compile como función/decorador | Captura y reutiliza el plan; `CUDALowering` fusiona grupos elementwise/map-reduce admitidos, incluida la selección `where` |
| Normas estables | Norma euclídea/Frobenius escalada en float32/float64, eager y compilada; disponible con `blas="custom"` y `blas="cublas"` |
| Acumulador bool en sum | No admitido; sí se admite sumar entradas bool con acumulador float32/64 explícito. min/max sí conservan bool |
| Fusión CUDA elementwise/map-reduce | Admitida mediante `backend.create_lowering()` y NVRTC |
| Graphs, transferencias pinned/solapadas | Pendientes |

La suma booleana por defecto solicita int64 mediante el contrato común, que
esta etapa no soporta. Puede elegirse acumulación float32/64 explícitamente.
`compile` no habilita operaciones fuera de esta tabla; los benchmarks etiquetan
su ejecución CUDA compilada como `backend-plan`. CUDA ofrece una factoría propia
`create_lowering()` que proporciona `CUDALowering` para los grupos elementwise y
map-reduce admitidos. Matmul y BLAS permanecen fuera de esa fusión. Las variantes
de reducción usan 128 lanes y chunks de hasta 4096 términos, con NVRTC sin FMA ni
FTZ implícitos (`fmad=false`, `ftz=false`). Consulta los [resultados de fusión CUDA](cuda-fusion-results.md).
Las normas estables usan el lowering genérico cuando se compilan y están
disponibles con ambos proveedores BLAS; `blas` continúa eligiendo solo el
proveedor de productos. Consulta el [alcance de normas CUDA](cuda-norm-results.md).
Los solvers permanecen en [examples](../examples/README.md), fuera del paquete;
sus algoritmos no forman parte del runtime.

Los lotes regulares de cuBLAS usan `gemm_strided_batched` en una llamada, con
strides positivos en elementos. Broadcasting general reutiliza punteros de las
entradas y ejecuta GEMM por lote; no replica matrices ni usa strides cero fuera
del contrato documentado. Queda margen para un adaptador batched de punteros en
dispositivo. Custom calcula todos los lotes en el mismo lanzamiento. La promoción
de vectores, resultados vacíos y K=0 conserva la semántica común de matmul.
Ver [mediciones de la ampliación](backend-sharing-results.md).

Desde la ronda del 2026-09-24, dos vectores usan `sdot`/`ddot`, con salida escalar
residente. Si un operando es vector y el otro tiene batch explícito, se usa GEMV
por matriz con offsets contiguos; admite batch singleton y múltiples ejes.
Los productos matriz-vector/vector-matriz sin batch conservan GEMM: seleccionar
GEMV universalmente fue más lento en la suite medida. La ganancia batched incluye
el bucle host más sencillo y no prueba que todo kernel GEMV gane a GEMM.

## Memoria, ejecución y precisión

El backend retiene el contexto primario y usa un stream no bloqueante propio.
Cada llamada activa temporalmente ese contexto y restaura el del llamador; no
resetea el contexto primario compartido con otras bibliotecas. Un lock protege
estado, lanzamiento y argumentos. Las operaciones pendientes retienen sus buffers
hasta sincronizar; el límite inicial es 64 lanzamientos. El pool de CUDA reutiliza
tamaños exactos completados, expulsa entradas antiguas y conserva como máximo
128 buffers/128 MiB solicitados al driver, sin medir su redondeo físico interno.
No incorpora todavía la reutilización por capacidad de Vulkan.

El plan compilado de al menos dos pasos mantiene el contexto y lock durante su
ejecución. Los planes más cortos evitan ese coste adicional en host. Las
activaciones anidadas comprueban el contexto actual del hilo y evitan otro
push/pop cuando ya coincide. Si otra biblioteca activa uno distinto, se activa
temporalmente el propio y se restaura aquel. No se añade una espera GPU ni se
cambia el límite de operaciones pendientes. Ver el
[contrato de contexto del Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).

Cada módulo CUDA compilado conserva un vector de argumentos `ctypes` con
almacenamiento host estable para `kernelParams`. El vector se vuelve a enlazar en
cada despacho bajo el mismo lock y solo contiene copias de valores numéricos y
direcciones de dispositivo; la plantilla no retiene arrays ni escalares del
llamador. Las asignaciones de dispositivo siguen retenidas en `_pending` hasta
`synchronize()`. La LRU de 128 entradas y `close()`, después de sincronizar,
liberan el módulo y su vector de argumentos. Esta optimización solo cambia el
empaquetado de parámetros; no cambia kernels, matemática, contextos ni stream.
Ver [mediciones del despacho CUDA](cuda-dispatch-results.md).

La subida mediante Driver API termina su copia antes de retornar, por lo que
modificar el input NumPy después no modifica la GPU. La descarga espera el stream.
La subida completa explícitamente el stream por defecto después de HtoD:
memoria host paginable puede terminar de copiarse a staging antes de que termine
el DMA al dispositivo. Nuestro stream no bloqueante no hereda esa dependencia.
Es una frontera conservadora inicial, no una implementación de copias solapadas.
[Contrato de copias](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html),
[dependencias entre streams](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html).
`release_cached_memory()` sincroniza y libera solo memoria libre; las salidas
vivas permanecen válidas. `close()` libera recursos propios e invalida arrays de
ese backend con un error de backend cerrado. No resetea recursos de terceros.

cuBLAS usa `PEDANTIC_MATH`, punteros escalares host y el stream propio. DOT cambia
temporalmente el pointer mode a DEVICE para escribir el resultado en GPU; el
lock protege el cambio y un `finally` restaura HOST. Si restaurarlo falla, el
backend lo registra y debe recuperar HOST antes de cualquier nuevo producto
BLAS. Los buffers enviados se retienen incluso si la restauración falla. NVRTC
compila sin fast-math, con `fmad=false`, `ftz=false` y división/raíz precisas.
No se modifica globalmente la configuración matemática de otras bibliotecas.
El [experimento de contracción float64](fp64-contraction-results.md) compara
ejecutables aislados con y sin FMA, con oráculos distintos para cada política.
Sus flags no se aplican al backend: CUDA custom conserva `fmad=false`.
Los árboles de reducción pueden diferir de NumPy/cuBLAS: se comparan errores,
no se promete identidad bit a bit entre proveedores. La suma ordinaria puede
desbordarse o perder precisión por cancelación; todavía no sustituye una norma
escalada ni ofrece garantías generales de estabilidad.

Las cachés son locales al contexto/backend y no escriben kernels en el directorio
personal del usuario.

Fuentes primarias para los contratos utilizados:
[Driver API y NVRTC en CUDA bindings](https://nvidia.github.io/cuda-python/cuda-bindings/13.3.0/overview.html),
[modo matemático cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html).

## Comparación reproducible

La suite aplicada admite los dos proveedores CUDA como opciones adicionales:

```powershell
python -m benchmarks.applications --backends opencl,opencl-clblast,vulkan,cupy,cuda-cublas,cuda-custom --repeats 7 --output .artifacts/benchmarks/cuda-applications.json
python -m benchmarks.applications --irregular --backends opencl,opencl-clblast,vulkan,cupy,cuda-cublas,cuda-custom --repeats 7 --output .artifacts/benchmarks/cuda-irregular.json
```

La primera compilación, preparación y transferencias se separan de ejecución
residente. Ridge compara veinte pasos comunes, no tiempo hasta convergencia.
No se consideran los tiempos de esta primera implementación como el techo CUDA.
