# Compilación opcional de funciones

Implementado en 0.1.0a1, 2026-09-06. La ejecución directa sigue siendo el valor
predeterminado. `compile` optimiza una función delimitada y mantiene el backend
de sus arrays. No crea un grafo lazy global ni cambia de CPU a GPU automáticamente.

```python
import numpy as np
import openpacho as pacho


@pacho.compile
def energia(a, b, c):
    x = a * b + c
    y = 2 * x
    return np.sum(y * y, axis=-1)


host = np.ones((1024, 32), dtype=np.float32)
original = energia(host, host, host)  # NumPy ejecuta la función original
# También "vulkan", "numpy" o "cuda"; CUDA fusiona los grupos admitidos.
with pacho.use_backend("opencl"):
    a = pacho.asarray(host)
    resultado = energia(a, a, a)
    print(energia.explain(a, a, a))
    print(energia.cache_info())
    host_resultado = pacho.asnumpy(resultado)
```

Las formas `pacho.compile(funcion)`, `@pacho.compile` y `@pacho.compile()` son
equivalentes. Las asignaciones `x` e `y` no cortan la captura. El `return` establece
las salidas del plan: un array o una tupla de arrays. El resultado público vuelve
a ser un ndarray ordinario residente; descargarlo requiere `asnumpy` o `item`.

## Especialización y escalares

La primera llamada con arrays OpenPacho captura el cuerpo Python, genera un plan
y ejecuta sus pasos. Las siguientes llamadas compatibles reenlazan buffers y
escalares, sin ejecutar otra vez el cuerpo Python. Los escalares calculados a
partir de argumentos escalares se recalculan en host por llamada, sin descargar
arrays. No se congela la tasa de la primera llamada:

```python
@pacho.compile()
def actualizar(x, b, tasa=0.01):
    return (1 - tasa) * x + tasa * b


x = actualizar(a, a, 0.01)
x = actualizar(x, a, 0.02)  # reutiliza el plan; cambia el valor de tasa
```

Las formas, ejes y otros parámetros que controlan la estructura se declaran
estáticos. Cambiar su valor genera otra variante:

```python
@pacho.compile(static_argnames=("eje",), cache_size=16)
def promedio(a, eje=-1):
    return np.mean(a, axis=eje)
```

Cada función tiene una LRU de hasta 32 planes por defecto. La clave incluye
forma/dtype y objeto backend de cada array, tipo de cada escalar y valores de los
argumentos estáticos. Todos los arrays públicos son contiguos; aún no existe una
clave de strides arbitrarios ni caché persistente de planes en disco.
`clear_cache()` libera los planes y reinicia estadísticas; los backends pueden
conservar sus programas y buffers libres. `explain(*args)` captura si hace falta,
sin ejecutar kernels, y muestra backend, BLAS, tipos, formas y pasos. No es un
perfilador de memoria ni mide sincronizaciones del controlador.

Las llamadas que suministran todos los parámetros por posición evitan reconstruir
el binding de la firma, si esta solo declara parámetros posicionales o
posicionales/por nombre. Siguen validándose los tipos y la especialización en cada
llamada; los escalares se reenlazan. Argumentos por nombre, defaults omitidos y
parámetros keyword-only conservan la ruta general. No requiere cambiar el código
del usuario ni llamar a `prepare`.
La captura de una variante nueva sí enlaza la firma; la ganancia repetida corresponde
a llamadas que reutilizan una especialización en caché.

Para repetir una especialización sin buscarla en la LRU en cada llamada:

```python
ejecutar = energia_compilada.prepare(a, b, c)
resultado = ejecutar(otro_a, otro_b, otro_c)
```

`prepare` devuelve un callable que valida forma, dtype y contexto de cada array,
tipo de escalares dinámicos y valores estáticos. Conserva los valores por defecto
y admite argumentos por nombre; no conserva los arrays de ejemplo ni congela
escalares dinámicos. Una incompatibilidad produce un error, no una recaptura.
Las salidas de llamadas sucesivas son independientes. Es un ejecutor Python
preparado, no código nativo ni un grafo de comandos GPU persistente.

## Operaciones y optimización actual

`compile(..., padding=True)` y `@compile(padding=True)` habilitan el
[padding opcional de GEMM](padding.md). `prepare` hereda esa política. Los pasos
admitidos usan buffers físicos alineados y recortan en GPU antes de continuar;
las formas lógicas no cambian. El valor predeterminado es `False`, y la ruta
habitual se conserva cuando backend, proveedor, dtype o forma no lo admiten.
`explain` muestra los pasos aplicados y su memoria privada solicitada.

Con OpenCL y `blas="clblast"`, un matmul compilado float32/64 puede consumir
directamente el almacenamiento original de una transpuesta completa 2D privada,
como en `a.T @ b`, `a @ b.T` o `a.T @ b.T`. Debe tener un único consumidor y no
ser una salida de la función. El backend prepara GEMM con los flags de
transposición y leading dimensions correspondientes; `explain` lo identifica
como `layout_matmul`. No hace falta cambiar el cuerpo NumPy de la función.
Las transpuestas compartidas o devueltas, los casos vacíos y las capacidades no
admitidas conservan su materialización. El padding explícito elegible se resuelve
primero. La ejecución eager conserva las copias habituales; esto no añade vistas
mutables ni retiene las entradas de captura. Los flags pueden cambiar la elección
de kernels y el redondeo del proveedor, por lo que no se promete una ganancia
general ni igualdad bit a bit con la ruta que materializa la transpuesta.

La captura reutiliza las reglas de ndarray: aritmética, broadcasting y promoción
de NumPy 2, `where(condition, x, y)`, reducciones `sum`/`mean`/`min`/`max`, matmul con
broadcasting por lotes, reshape, slicing, transposición, diagonales y norma
euclídea/Frobenius. `where` promueve solo `x` e `y` y admite condiciones bool o
numéricas. Ambas ramas se evalúan antes de seleccionar. Se admiten constructores
como `zeros_like`, `ones_like` y `pacho.full` con parámetros constantes compatibles.
La cobertura y dtypes también dependen del backend. En los pasos `reduce` y
`map_reduce`, `explain` indica `operation`: sum, min o max. Los extremos conservan
dtype, propagan NaN y usan la misma regla de empates de cero que eager; ver
[compatibilidad](compatibility.md#extremos-por-ejes).

OpenCL y CUDA también preparan `min/max` float32 sobre entradas materializadas
como `map_reduce`, aunque no haya un productor que fusionar. Las fronteras de
productores compartidos o devueltos se conservan. Vulkan, los otros dtypes y las
sumas mantienen su selección anterior. La [comparación de planes de extremos](extrema-planning-results.md)
documenta la ganancia medida y las regresiones que impiden extenderla a Vulkan.

El [generador aleatorio](random.md) tiene estado: sus llamadas se rechazan durante
`compile`/`fuse` antes de consumir contadores, incluso desde una closure. Genera
los inputs aleatorios fuera y pásalos como argumentos; no se congela una muestra
de la captura para reutilizarla en todas las ejecuciones.

- **OpenCL:** fusiona cadenas elemento a elemento y productores elementwise con
  `sum`/`min`/`max`, respetando conversiones intermedias y dtype de acumulación. `x*x` cuenta
  como un consumidor de x; no obliga a almacenar x. Un productor compartido entre
  consumidores diferentes o devuelto por la función se materializa. Map-reduce
  usa parciales compactos cuando la reducción es grande; un paso lógico no promete
  un solo kernel. Gather y matmul conservan sus fronteras salvo las transpuestas
  privadas CLBlast descritas arriba y el patrón privado `b - A @ x` con GEMV
  custom compatible. No hay eliminación general de expresiones equivalentes ni
  epílogos dentro del proveedor CLBlast.
- **Vulkan:** fusiona cadenas elemento a elemento con escalares por valor y
  productores con su reducción, con las mismas fronteras de consumidores y
  salidas públicas. Map-reduce usa acumulación tipada, grupos cooperativos y
  parciales compactos. Reutiliza pipelines, buffers y recursos de comandos. Agrupa el envío de
  comandos de cada invocación sincrónicamente en host y los ejecuta de forma
  asíncrona en GPU; no espera al resultado ni crea un grafo lazy global. Un grupo
  se divide si se requiere sincronizar o se llega al límite de recursos pendientes.
- **NumPy:** ejecuta primitivas NumPy. No genera código CPU nativo; envolver
  arrays OpenPacho CPU no promete acelerar NumPy. Con inputs NumPy nativos se
  ejecuta directamente la función original.

La selección `where` también puede combinarse en los grupos elementwise y
map-reduce admitidos por `compile`/`fuse`; no evalúa ramas de forma perezosa ni
admite la forma de un argumento o el keyword `where=` de ufuncs. Sus resultados
y límites están en [selección condicional](compatibility.md#selección-condicional)
y [resultados de selección](selection-results.md).

CLBlast sigue detrás de una operación matmul. Se escoge mediante el backend
OpenCL antes de crear arrays; no limita la extensión del IR. Elegir otro proveedor
BLAS todavía crea otro contexto, por lo que no es intercambiable sobre buffers
ya existentes sin transferencia explícita.

Los temporales vivos pertenecen a cada invocación. El plan no guarda los arrays
pasados durante la captura; deja de referenciar intermedios al agotar sus usos.
Los backends conservan recursos en vuelo mientras sean necesarios. La caché de
planes tiene lock y se verifican llamadas concurrentes con inputs distintos.

## Límites del contrato

La función debe ser pura. Este trazador por sobrecarga no puede detectar todo
efecto de Python: modificar una lista global o leer un global mutable es
responsabilidad del autor y queda fuera del contrato. Pasa datos cambiantes como
argumentos. No hay una reejecución silenciosa en CPU cuando falla la captura.

No se admiten mutación de arrays, `out=`, lecturas host, condiciones Python
dependientes de arrays/escalares dinámicos, llamadas compiladas anidadas, argumentos
array dentro de contenedores, `*args`/`**kwargs`, ni Python arbitrario dentro del
plan. Funciones auxiliares puras sin decorar sí pueden participar en la captura.
Las constantes array deben convertirse antes y pasarse explícitamente; no se
capturan arrays de una clausura. Los constructores `np.*` sin argumentos array
no pasan necesariamente por despacho NumPy y no son portables automáticamente.

Un bucle Python con límites estáticos se desenrolla durante la captura; no se
convierte en un bucle nativo de GPU. Para muchas iteraciones, compila un paso y
mantén el bucle fuera. Los solvers de `examples/solvers.py` ya reutilizan planes de actualización,
conservando las comprobaciones de validez y residuo real en cada iteración.
En GPU, el residual y su norma escalada se calculan mediante un plan común.
La norma es una operación propia del IR, conservando su estabilidad al compilar.
El patrón `b - A @ x` puede fusionar GEMV y resta si el producto es privado,
los dtypes flotantes coinciden y el proveedor es custom OpenCL o Vulkan. Un
producto compartido/devuelto o CLBlast conserva su frontera. CG y descenso más
pronunciado usan direcciones normalizadas y no forman el cuadrado de la norma
para sus coeficientes. No se relaja la tolerancia ni la frecuencia de chequeo.
Compilar pasos no elimina esas observaciones host ni garantiza ganar a NumPy.

`fuse` continúa disponible sobre el mismo IR para un único resultado elemento a
elemento. Retraza cada llamada; para trabajo repetido, utiliza `compile`.

La [arquitectura aceptada](execution-architecture-proposal.md) describe extensiones
futuras. El [análisis de gaps](gaps-and-optimization.md) cuantifica el alcance actual.

## CUDA inicial y aplicaciones externas

Internamente, el backend puede proporcionar `create_lowering()` y un contexto
`execution_scope()` para agrupar una ejecución. El plan enlaza estos métodos al
prepararse; no decide por nombres ni los renegocia en cada operación. Si no hay
lowerer usa las primitivas del backend elegido. Los errores de preparación o
ejecución se propagan, sin cambiar de backend ni relajar precisión.

CUDA reutiliza captura y planes. `backend.create_lowering()` enlaza su
`CUDALowering`, que compila con NVRTC los grupos elementwise/map-reduce admitidos.
Las reducciones fusionadas usan 128 lanes y chunks de hasta 4096 términos; NVRTC conserva
`fmad=false` y `ftz=false`. Matmul y BLAS siguen siendo pasos separados.
Cada módulo CUDA compilado conserva un vector de argumentos `ctypes` con
almacenamiento host estable. Cada llamada copia los escalares y las direcciones
de dispositivo en ese vector bajo el lock existente antes del enqueue; la plantilla
no retiene arrays ni escalares del llamador. Las asignaciones siguen en `_pending`
hasta `synchronize()`. La LRU de 128 variantes y `close()`, después de sincronizar,
liberan el módulo y el vector. Esta optimización de despacho no cambia kernels,
matemática, contextos ni stream; ver [resultados de despacho CUDA](cuda-dispatch-results.md).
`compile` no amplía los dtypes/operaciones del [contrato CUDA](cuda.md); el
acumulador bool en reducciones no está admitido, aunque las entradas bool sí
pueden reducirse con acumulador float32/64. Las normas estables float32/64 sí están
disponibles en CUDA eager y compilado, con ambos proveedores BLAS; graphs siguen
pendientes. Los solvers son [ejemplos externos](../examples/README.md), no una
extensión del compilador, y sus algoritmos no cambiaron al habilitar CUDA.
Consulta los [resultados de fusión CUDA](cuda-fusion-results.md). Los patrones
optimizados deben corresponder a operaciones genéricas y beneficiar funciones
equivalentes del usuario.
