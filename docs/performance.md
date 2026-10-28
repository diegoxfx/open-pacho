# Ejecución, fusión y CLBlast

La [ronda de tres iteraciones del 2026-09-24](three-iterations-results.md) incorpora
tiles rectangulares sin copias en OpenCL/CUDA, selección de primitivas vectoriales
cuBLAS y flags de transposición privada al compilar con CLBlast. Incluye controles
de la ruta anterior, dos rondas, transferencias y errores numéricos. Las cifras
de etapas anteriores son snapshots; en particular, la ventaja histórica de
padding no se transfiere automáticamente al GEMM directo más rápido.

El usuario elige el backend y el dispositivo. OpenPacho no cambia automáticamente
entre CPU y GPU ni mueve un array para decidir dónde ejecutar una operación.
Los backends implementados son `numpy`, `opencl`, `vulkan` y [CUDA inicial](cuda.md). La presencia de una GPU no implica que soporte cualquier backend.

Para código nuevo repetido se recomienda [`@pacho.compile`](compilation.md):
evita retrazar la función en cada llamada y admite reducciones, álgebra y varias
salidas. `fuse` se conserva como captura acotada por llamada. OpenCL y Vulkan
fusionan expresiones elemento a elemento y las integran con su reducción cuando
el intermedio no se devuelve ni tiene varios consumidores; CUDA aplica el mismo
esquema a los grupos elementwise/map-reduce admitidos.

El [análisis de gaps y optimización](gaps-and-optimization.md) distingue los costes
medidos y las oportunidades pendientes, sin presuponer paridad entre backends.

## Fusión explícita

```python
import numpy as np
import openpacho as pacho


@pacho.fuse
def actualizar(x, b, tasa=0.01):
    return (1 - tasa) * x + tasa * b


with pacho.use_backend("opencl"):
    x = pacho.asarray(np.ones(10000, dtype=np.float32))
    b = pacho.asarray(np.zeros(10000, dtype=np.float32))
    for _ in range(100):
        x = actualizar(x, b)
    resultado = pacho.asnumpy(x)
```

Cada llamada a `actualizar` ejecuta un kernel elemento a elemento y crea un único
buffer de salida. Los intermedios se calculan dentro del kernel. Los escalares se
pasan por valor, sin buffers ni transferencias de arrays escalares. El bucle de
Python sigue ejecutando 100 llamadas: esta versión no fusiona el bucle completo.

Se admiten las ufuncs elemento a elemento ya soportadas, broadcasting, constantes
y argumentos escalares, y un único resultado. No se admiten reducciones, `@`,
indexación, escritura, conversión a host o condiciones Python que dependan de
los datos trazados. La función debe ser pura: su cuerpo Python se ejecuta en cada
llamada. Los argumentos array deben ser OpenPacho; la conversión desde NumPy se
hace explícitamente antes de llamar. No se captura Python arbitrario.

La resolución de tipos y las conversiones intermedias siguen NumPy 2; no se activa
fast-math. La contracción automática de operaciones flotantes está deshabilitada
en los kernels fusionados. Esto no garantiza igualdad bit a bit en funciones
trascendentes. El backend NumPy evalúa el mismo grafo sin generar código nativo.
Los programas OpenCL, módulos NVRTC CUDA y pipelines Vulkan se almacenan en cachés
limitadas a 128 variantes por backend; formas, tipos y expresión determinan la
variante, los valores escalares no. En Vulkan, los límites de descriptores y push
constants del dispositivo siguen acotando el tamaño de una expresión fusionada.

Con `compile`, un cálculo como `np.sum((a - b) * (a - b), axis=1)` puede ejecutar
map-reduce en OpenCL, Vulkan y CUDA sin materializar todo el array transformado. Las
reducciones grandes conservan parciales en dispositivo y necesitan una etapa posterior.
En CUDA, `CUDALowering` compila los grupos admitidos con NVRTC; matmul y BLAS
conservan su frontera de proveedor. Consulta los [resultados de fusión CUDA](cuda-fusion-results.md).
Una expresión compartida por consumidores distintos o devuelta como salida conserva su
materialización; no se promete que cualquier grafo se convierta en un único kernel.

## Memoria y métodos iterativos

OpenCL reutiliza buffers mediante un `MemoryPool`. Las subidas de host ya contiguo
evitan una copia intermedia de CPU; la copia hacia GPU termina antes de retornar,
preservando el aislamiento si el usuario modifica después el input. Los arrays
no contiguos requieren preparación en host.

La cola en orden mantiene las dependencias entre operaciones y reutilizaciones.
El pool retiene memoria libre para llamadas posteriores. Para liberarla:
`pacho.get_backend("opencl").release_cached_memory()`; esto sincroniza la cola y
conserva los arrays vivos. Si se usa otra configuración de BLAS, debe llamarse
sobre ese objeto backend.

Vulkan reutiliza buffers de dispositivo y staging por tamaño exacto o capacidad
compatible acotada, con un límite
conjunto de 128 MiB de asignaciones nativas y 128 buffers libres. Los comandos
pendientes retienen sus buffers para impedir una reutilización prematura. Los
comandos y, en la ruta de sets, pools de descriptores completados también se reutilizan, con un máximo
de 128 recursos de cada tipo. `pacho.get_backend("vulkan").release_cached_memory()`
espera el trabajo pendiente y libera esas cachés, conservando los arrays vivos.
Las llamadas compiladas agrupan dispatches en un command buffer, con cortes por
sincronización o por 64 operaciones grabadas/pendientes; eager conserva envío por
operación. Se usan push descriptors cuando la capacidad y el número de bindings
lo permiten; los demás grupos reutilizan descriptor sets completados. Se prefiere
readback coherente con caché CPU. Se conservan plantillas de dispatch y estructuras
nativas de descriptores, barreras y envío. Los registros host de push se reutilizan
tras grabarlos; los buffers GPU se retienen hasta terminar su uso.
Ver [mediciones actuales](backend-sharing-results.md).
Esto no elimina la grabación de comandos ni el coste de muchas llamadas Python.

Los solvers de [examples](../examples/README.md) reutilizan planes compilados para actualizaciones vectoriales;
OpenCL y Vulkan las fusionan, incluido el patrón privado `b - A @ x` con GEMV
custom compatible. CG y descenso más pronunciado usan la norma escalada del
estado residual y direcciones normalizadas, mantienen coeficientes en GPU y descargan comprobaciones
de validez. Se sigue comprobando el residuo real **en cada iteración**, conservando
la información de convergencia y fallo. Todavía existen sincronizaciones para las
decisiones de parada; no son solvers completamente autónomos en dispositivo.
El orden y precisión de los cálculos de coeficientes puede cambiar ligeramente
el número de iteraciones frente a la versión anterior.

Los planes de avance agrupan actualización, residuo y norma; los de curvatura
agrupan GEMV y DOT (más normalización para descenso más pronunciado). La validación
host lee directamente la curvatura; evita un kernel booleano adicional y mantiene
el coeficiente en dispositivo para actualizar con el mismo dtype.

Para resolver repetidamente con la misma matriz, existe un contrato explícito:

```python
from examples import solvers  # requiere el repositorio, no viene en el wheel

sistema = solvers.prepare(A, method="cg")  # también "jacobi" o "sd"
primero = sistema.solve(b1, rtol=1e-5)
segundo = sistema.solve(b2, rtol=1e-5)
```

La preparación copia la matriz y valida su forma y finitud una vez; Jacobi también
guarda la diagonal validada. Las modificaciones posteriores de la matriz original
no alteran el sistema. Cada resolución valida b/x0, tolerancias y residuo real en
cada iteración. No se demuestra que la matriz sea SPD ni que Jacobi converja.
La precisión queda fijada por A (float64 para matrices no flotantes); un b que
exija promoverla se rechaza. Debe prepararse una matriz convertida explícitamente.
La copia añade memoria y tiempo de preparación: no se recomienda como aceleración
de una única resolución. Los pasos compilados de los solvers preparan sus bindings
una vez por resolución, sin cambiar el contrato público de convergencia.

## BLAS opcional, kernels propios conservados

```python
with pacho.use_backend(
    "opencl", device="0:0", blas="clblast", clblast_library="/ruta/a/la/biblioteca/CLBlast"
):
    A = pacho.asarray(np.eye(64, dtype=np.float32))
    y = A @ A
```

`clblast_library` es la ruta del binario nativo apropiado para el sistema (`.dll`,
`.so` o `.dylib`); también puede configurarse con `OPENPACHO_CLBLAST_LIBRARY`.
Sin ruta explícita se intenta encontrar la biblioteca instalada. CLBlast se
distribuye por separado: `pip install open-pacho[opencl]` no instala ese binario.

`blas="custom"` es la opción predeterminada y permite comparar los kernels propios.
`blas="clblast"` delega los productos flotantes admitidos: DOT/GEMV nativos para
operaciones vectoriales de rango 1/2 cuando la biblioteca expone sus símbolos y
GEMM para los demás casos. Si faltan los símbolos opcionales, permanece el camino
GEMM. `matmul` por lotes usa GEMM strided-batched nativo para offsets regulares,
o GEMM batched con arrays de offsets para broadcasting irregular, cuando están
disponibles. Los offsets apuntan a los buffers originales; no se duplican inputs
broadcast. Si faltan esos símbolos, conserva el bucle de GEMM del mismo proveedor.
Un error de ejecución nativa se propaga y no activa ese camino alternativo.
Los productos enteros y
las dimensiones vacías se resuelven en OpenPacho. Una biblioteca ausente o un
error de ejecución producen un error explícito; no provocan un cambio oculto de
motor. Configuraciones diferentes tienen contextos distintos; se elige la
configuración antes de crear los arrays.

Los kernels OpenCL propios cooperan entre work-items para GEMV con vector derecho
y DOT grandes. OpenCL y Vulkan usan una nueva ruta GEMM float32 para M/N≥32,
K≥16 y capacidad suficiente: cada grupo calcula 32×32 salidas y cada hilo guarda
4×4 acumuladores en registros, con 4 KiB de memoria compartida. Incluye broadcasting
por lotes y bordes. OpenCL conserva tiles locales de 16×16/8×8 o el kernel simple
para otros casos. En la ruta de registros, OpenCL elimina guardas cuando M/N son
múltiplos de 32 y K de 16; ver [control y resultados](kernel-stability-results.md).
Los casos parciales conservan las guardas con la configuración predeterminada.
El [padding opcional](padding.md) alinea GEMM custom float32 en OpenCL, Vulkan y
CUDA a cambio de buffers y copias; los tiempos públicos incluyen todo ese coste.
La [reevaluación](padding-reassessment.md) encuentra ganancias en algunas formas
rectangulares OpenCL/CUDA y pérdidas en la mayoría; se mantiene desactivado por
defecto, sin selección automática ni recomendación general para Vulkan.
Vulkan conserva su ruta de 8×8 para los casos no elegibles.
GEMM permite contracción
multiplicación-suma; la fusión elementwise conserva sus redondeos. Vulkan también
usa reducciones/GEMV cooperativos y parciales para DOT largo. Ver la
[investigación de CLBlast y controles de registros](clblast-techniques.md).
Falta medir y ajustar otras formas, tamaños y fabricantes.

La integración se limita a un adaptador de operaciones BLAS sobre buffers OpenCL.
Las expresiones, asignaciones y reducciones siguen bajo control de
OpenPacho. Una llamada BLAS es una frontera de fusión: no podemos insertar una
actualización arbitraria dentro de su kernel. Para operaciones donde esa frontera
cueste demasiado, se pueden desarrollar kernels propios y compararlos con CLBlast
sin cambiar la API pública de arrays. No se presupone qué implementación gana por
tamaño o dispositivo; no hay selección automática ni ajuste por hardware todavía.

Fuentes de las interfaces utilizadas:
[MemoryPool de PyOpenCL](https://documen.tician.de/pyopencl/tools.html),
[CLBlast](https://github.com/CNugteren/CLBlast).

Las matrices dispersas y los precondicionadores quedan como deuda futura.

## Diagnóstico de dispatch Vulkan

La [ronda de frontera CFFI](vulkan-dispatch-results.md) compara el binding generado
con comandos tipados del mismo backend. Se conservan conteos y resultados;
los tiempos GPU instrumentados y de pared se publican separados. Las mediciones
no atribuyen automáticamente el tiempo restante a Python ni a un límite físico.
