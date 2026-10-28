# Compatibilidad con NumPy

El [backend CUDA](cuda.md) implementa un subconjunto menor: float32/64 y bool,
operaciones elementwise, gather, sum/mean flotantes y min/max float32/64 o bool, normas estables float32/64
(eager y compiladas) y matmul con vectores, matrices y broadcasting por lotes.
Enteros y acumuladores bool para sum en CUDA siguen pendientes; las entradas bool pueden
reducirse con acumulador float32/64 explícito. La fusión CUDA elementwise/map-reduce
está disponible para los grupos admitidos y matmul/BLAS
conservan su frontera. Las normas están disponibles con ambos proveedores BLAS.
Consulta los [resultados de fusión CUDA](cuda-fusion-results.md) y el [alcance de
normas CUDA](cuda-norm-results.md). Las tablas generales siguientes documentan
el contrato compartido, con diferencias por backend indicadas donde corresponde.

OpenPacho implementa un subconjunto explícito; no reexporta funciones de NumPy que
puedan ejecutar accidentalmente operaciones en CPU. `__array_ufunc__` y
`__array_function__` mantienen el backend de los arrays.

| Área | Soportado | Límites de esta versión |
| --- | --- | --- |
| Creación | array, asarray, empty, zeros, ones, full, zeros_like, ones_like, arange, linspace, eye | Orden C; full escalar. arange/linspace/eye se generan en host y se suben una vez. Sin archivos. |
| Aleatorios | random.default_rng: random, uniform, normal, exponential | [Generación residente](random.md), parámetros escalares float32/64. Vulkan normal/exponential solo float32. Sin paridad de secuencias NumPy, enteros ni captura. |
| Datos | float32/64, int32/64, uint32/64, bool | Sin float16, enteros de 8/16 bits, complejos, objetos o dtypes estructurados. float64 exige fp64. |
| Aritmética | +, -, *, /, potencia flotante, operadores inversos | Sin potencia entera, //, %, operadores de bits, `out=`, `ufunc where=` ni `ufunc.reduce`. |
| Funciones | sqrt, exp, log, sin, cos, tanh, abs, isfinite, comparaciones, `where(condition, x, y)` | `where` transmite los tres argumentos y acepta condición bool o numérica; el dtype de salida se promueve desde `x` e `y` según NumPy 2, con escalares Python débiles. No admite la forma de un argumento. |
| Broadcasting | Dimensiones arbitrarias, escalares Python y arrays 0-D | Combinar dispositivos/contextos exige una transferencia explícita. |
| Forma | shape, ndim, size, nbytes, dtype, reshape, transpose, T | Solo orden C; reshape puede compartir almacenamiento. |
| Índices | Enteros, slices con paso positivo/negativo, ellipsis, None/newaxis | Slices y transpose eager son copias en dispositivo, no vistas. La compilación puede omitir una transpuesta privada de matmul CLBlast según las condiciones siguientes. Sin máscaras booleanas ni indexación avanzada. |
| Escritura | Operaciones crean resultados nuevos | Sin setitem. `a += b` y similares no se admiten mediante out=; usa `a = a + b`. |
| Reducciones | sum, mean, min/max (amin/amax); axis=None/int/tuple y keepdims | sum promueve enteros pequeños a 64 bits. mean acumula en float32/64. min/max conservan dtype. Sin initial/where/out. |
| Álgebra | @/matmul con vectores, matrices y broadcasting por lotes; dot 0D/1D/2D; diag de una matriz; norma euclídea/Frobenius escalada | CUDA admite normas float32/64 en eager/compile con custom y cuBLAS; sin matmul booleano, dot de rango superior a 2, diag desde vector, orden espectral, solve directo, SVD/FFT o sparse. |
| Transferencias | asnumpy, to_host, to_backend, item | Sin conversión implícita con np.asarray. item sincroniza y copia un escalar. |

## Selección condicional

`where(condition, x, y)` evalúa la verdad de cada elemento de la condición y
selecciona entre `x` e `y` con broadcasting de los tres argumentos. Los arrays
deben pertenecer al mismo backend/dispositivo; la selección permanece en ese
backend y no descarga datos. La ruta eager está disponible junto con captura
`compile`/`fuse`, incluida su composición con los grupos elementwise y map-reduce
admitidos. Ambas ramas se evalúan antes de seleccionar; `where` no es una operación
de control de flujo perezosa. No se admite `where(condition)` (la forma que devuelve
índices), ni el argumento keyword `where=` de ufuncs o reducciones. Consulta los
[resultados de selección](selection-results.md).

## Extremos por ejes

`min`/`max`, sus alias `amin`/`amax` y los métodos `a.min()`/`a.max()` conservan
el dtype de entrada y permanecen residentes. Propagan NaN; los infinitos
participan en las comparaciones. Los enteros de 64 bits se comparan sin pasar
por flotantes. CUDA admite float32/64 y bool; sus enteros siguen fuera de alcance.

El signo del cero tiene una regla explícita y común a los backends: en un empate
entre `-0` y `+0`, min devuelve `-0` y max devuelve `+0`. Si solo aparece un signo,
lo conserva. NumPy puede escoger el signo según el orden/implementación de su
reducción; no se promete igualdad de bits con NumPy en ese empate.

Un eje reducido de longitud cero produce `ValueError`, incluso si la salida
también sería vacía. Una salida vacía con ejes reducidos no vacíos es válida.
`axis=()` conserva forma y valores. No se admite dtype de acumulación, `initial`,
`where` ni `out`. `compile` captura extremos y fusiona productores elegibles con
su reducción; `fuse` sigue limitado a elementwise. Ver [alcance y pruebas](extrema-results.md).

Los resultados escalares son arrays 0-D en el mismo dispositivo. Para control de
flujo Python, llama `.item()`. `if array:` siempre falla para evitar sincronizaciones
ocultas. El repr muestra metadatos y no descarga los valores.

Los inputs NumPy no contiguos se copian a orden C al subirlos. `asarray` puede
reutilizar un array Pacho en el mismo backend/dtype, pero los datos de host siempre
se copian a almacenamiento propio. `copy=False` al construir desde host no promete
compartir memoria con el objeto original.

Los resultados de punto flotante no tienen por qué coincidir bit a bit con NumPy:
el orden de las sumas y las funciones matemáticas del dispositivo puede variar.
La contracción producto-suma también depende de la ruta. CUDA custom desactiva
FMA en NVRTC; el GEMM custom OpenCL permite al compilador contraer, mientras sus
lowerers elementwise/map-reduce la desactivan. La [comprobación float64 local](fp64-contraction-results.md)
identifica las instrucciones generadas y los efectos numéricos de ambas políticas.
No existe todavía un selector público uniforme de contracción entre proveedores.
La conversión de flotantes a enteros fuera del rango representable no tiene
paridad garantizada entre NumPy y OpenCL; debe evitarse también en expresiones
fusionadas.
La norma usa suma de cuadrados escalada; evita formar directamente el cuadrado
de entradas muy pequeñas o grandes.
No se amplía el dtype: un resultado fuera de su rango sigue sin ser representable,
y los modos de aritmética subnormal del dispositivo siguen siendo relevantes.
Los solvers de [examples](../examples/README.md), fuera del paquete, mantienen
el chequeo de residuo en cada iteración. Un residuo pequeño
no certifica una solución precisa en sistemas mal condicionados; ver
[medición del error](numerical-accuracy.md).

## Ruta de migración

1. Ejecuta las pruebas del código original con NumPy y conserva resultados de referencia.
2. Convierte inputs con `pacho.asarray(..., dtype=...)` y selecciona OpenCL explícitamente.
3. Mantén llamadas NumPy compatibles, o cambia el import a `openpacho` para creadores y operaciones soportados.
4. Transfiere con `asnumpy` solo al entregar resultados a librerías que requieran CPU.
5. Compara precisión y tiempos después de una primera ejecución para compilar kernels.
6. Para cálculos repetidos, evalúa `@pacho.compile` según los
   [límites de captura y compilación](compilation.md). El backend sigue siendo explícito.

Una función de una dependencia que llama internamente `np.asarray` puede requerir
adaptación; soportar los protocolos de NumPy no implica compatibilidad universal
con SciPy, pandas, scikit-learn u otros paquetes.


## Compilación y capacidades por backend

La compatibilidad es deliberadamente parcial: se prioriza cálculo matricial
frecuente frente a replicar todo NumPy. [`compile`](compilation.md) añade
restricciones locales de pureza/captura y usa la misma semántica de operaciones.
La ejecución directa sigue disponible.

`compile(..., padding=True)` conserva el cuerpo NumPy intercambiable (`a @ b`)
y activa [padding interno](padding.md) solo para GEMM custom float32 elegible.
También existe `pacho.matmul(..., padding=True)` para ejecución directa; ese
keyword es propio de OpenPacho. El valor predeterminado es `False`. Las formas
lógicas y el proveedor elegido se conservan, incluidos los casos sin soporte.

Con OpenCL y CLBlast, `compile` puede sustituir una transpuesta completa 2D
privada por flags del GEMM float32/64, conservando el almacenamiento original
como entrada. Solo aplica si esa transpuesta tiene un único consumidor matmul y
no es una salida pública. Los temporales compartidos, salidas independientes,
dimensiones vacías y backends sin esa capacidad mantienen la copia; la ejecución
eager tampoco cambia. El padding explícito elegible tiene precedencia. El plan
no retiene las entradas de ejemplo ni introduce vistas públicas. Esta optimización
no garantiza ventaja en todas las formas ni igualdad bit a bit con el producto
de transpuestas materializadas; véase [compilación](compilation.md).

OpenCL y Vulkan se seleccionan explícitamente. Vulkan no admite trascendentes
float64 ni potencia float64; int64/uint64 requieren shaderInt64. Sus límites y
estado de optimización se detallan en [Vulkan](vulkan.md). El [backend CUDA](cuda.md)
es inicial: cuBLAS puede ser su proveedor; CuPy sigue siendo un comparador externo.
Los extras de instalación son independientes: OpenCL/Vulkan no requieren CUDA.
