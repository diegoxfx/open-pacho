# Arquitectura

La [arquitectura aceptada](execution-architecture-proposal.md) distingue el
objetivo futuro del subconjunto implementado aquí. Ver también
[compilación](compilation.md), [Vulkan](vulkan.md) y el
[análisis de gaps y optimización](gaps-and-optimization.md). La [dirección acordada](architecture-evolution.md)
separa decisiones futuras del código actual. Solvers y sistemas preparados viven
en `examples/solvers.py`, fuera del paquete de operaciones matriciales.

## Contrato

`src/openpacho/_array.py` valida formas y dtypes, implementa los protocolos NumPy y
delega operaciones a un backend. Cada ndarray tiene forma, dtype y almacenamiento
contiguo propio del backend. Sus metadatos no necesitan transferencias de GPU.

`backends.py` mantiene la selección mediante ContextVar, por lo que los contextos
anidados se restauran aun ante errores. No inicializa una GPU al importar. Los
backends se reutilizan por ID de dispositivo; arrays de contextos distintos no se
combinan implícitamente. `to_backend` es una copia explícita a través del host.

`random.py` reserva contadores de un generador ligado al backend; `_random.py`
comparte la aritmética Philox4x32-10 sin conocer nombres de backend. Los adaptadores
`_random_opencl.py`, `_random_vulkan.py` y `_random_cuda.py` poseen dialecto,
plantillas cacheadas y lanzamiento; cada backend enlaza su implementación. La consulta
`random_distributions(dtype)` declara soporte antes de reservar el stream.
La GPU genera directamente en la salida, con semilla/contador como escalares
dinámicos. El RNG con estado se rechaza durante captura; ver [contrato](random.md).

`_cuda.py` incorpora el backend CUDA inicial: retiene el contexto primario, usa
un stream propio, restaura el contexto del llamador y conserva buffers en vuelo.
NVRTC ejecuta primitivas propias; GEMM/GEMV/dot admiten cuBLAS o un kernel directo.
No usa arrays de CuPy ni fallback CPU. `compile` enlaza el `CUDALowering` propio
mediante `create_lowering()` para fusionar grupos elementwise/map-reduce. Admite
lotes broadcast: cuBLAS usa strided-batched para lotes
regulares y GEMM por lote para broadcasting general; custom ejecuta los lotes
en un kernel y comparte el generador tiled float32 de OpenCL/Vulkan.
En cuBLAS, DOT usa la primitiva nativa con resultado residente y un vector contra
matrices batched usa GEMV con offsets contiguos. Los productos vectoriales sin
batch conservan GEMM según la comparación medida. El pointer mode de DOT se
restaura bajo lock; un fallo de restauración exige recuperación antes de enviar
otro producto. No cambia el proveedor elegido por el usuario.
`_cuda_gemv.py` coopera entre 128 hilos por fila para productos con vector derecho
y K >= 256, incluidos lotes contiguos. DOT largo reduce parciales en dispositivo.
No cambia dtype ni flags matemáticos; el árbol de suma difiere del kernel serial.
El scope de ejecución mantiene contexto/lock durante planes de al menos dos pasos físicos conocidos y evita push/pop
anidados redundantes, comprobando el contexto real del hilo para respetar otras
bibliotecas. Las operaciones conservan sus fronteras de sincronización.
Cada módulo CUDA compilado conserva un vector de argumentos `ctypes` con
almacenamiento host estable. El backend lo vuelve a enlazar en cada despacho bajo
el lock existente y solo copia valores numéricos y direcciones de dispositivo; la
plantilla de kernel no retiene arrays ni escalares del llamador. Las asignaciones
siguen en `_pending` hasta `synchronize()`. La LRU de 128 variantes y `close()`,
después de sincronizar, liberan el módulo y el vector. Esta optimización no cambia
kernels, matemática, contextos ni stream; ver [resultados de despacho CUDA](cuda-dispatch-results.md).
Matmul y BLAS siguen siendo pasos separados. Las normas estables float32/64
se admiten en eager y compile, independientemente
de si el proveedor BLAS elegido es custom o cuBLAS. Las entradas bool sí pueden
reducirse con acumulador float32/64 explícito; la acumulación bool y graphs CUDA
siguen pendientes. Los algoritmos de solver siguen en examples, fuera del runtime;
el [contrato CUDA](cuda.md) detalla precisión y dependencias opcionales.

`_opencl.py` crea un contexto y una cola en orden por dispositivo y configuración
BLAS. Los buffers se reutilizan mediante MemoryPool. Los kernels se compilan bajo
OpenCL C 1.2 y solo habilitan fp64
cuando el dispositivo lo ofrece. No usan extensiones específicas de fabricante
ni atómicos de punto flotante. Los kernels cooperativos eligen un tamaño de grupo
compatible con los límites del dispositivo.

Elementwise/gather reutilizan metadatos inmutables con LRU acotada, sin retener
valores ni arrays. Usan aritmética de índices int32 si cabe cada acceso intermedio,
conservando int64 para formas/offsets mayores. Esto es independiente del dtype
almacenado. Las transpuestas eager y las expuestas como resultados independientes
siguen materializándose; un producto compilado puede omitir una transpuesta
privada mediante la capacidad de layout descrita más abajo.

Hay kernels para operaciones elemento a elemento con índices de broadcasting,
gather para slicing/transposición, reducciones y productos vector/matriz, incluidos
lotes con broadcasting de dimensiones iniciales. Las reducciones eager totales
grandes usan parciales por bloques de 256 elementos; otras reducciones eager
asignan un elemento de salida por work-item. El camino `blas="custom"` coopera
entre work-items para GEMV con vector derecho y DOT suficientemente grandes;
DOT largo usa parciales. GEMM float32 elegible combina memoria local y bloques de
registros: 64 hilos calculan 32×32 salidas, con 4×4 acumuladores por hilo. Otros
tipos/tamaños conservan tiling local o el kernel simple según las capacidades.
En OpenCL, M/N múltiplos de 32 y K múltiplo de 16 permiten eliminar guardas
redundantes de esos tiles; las demás formas conservan las guardas anteriores.
No cambian layout público, precisión ni asignaciones.

`_gemm_rect.py` añade tiles de 16×32/K=16 en OpenCL y 16×16/K=32 en CUDA
para una región medida de matrices 2D float32 con pocas filas y K largo.
Cada backend declara su configuración; el selector común conserva la ruta
anterior fuera de esa región, en lotes y con padding explícito. No hay nuevos
buffers ni packing. Vulkan conserva el tile original. Ver
[selección y resultados](three-iterations-results.md).

La opción [padding](padding.md), desactivada por defecto, permite además alinear
GEMM custom float32 en OpenCL/Vulkan/CUDA. `_padding.py` prepara las formas físicas
y el plan; `_padding_copy.py` copia palabras uint32 para preservar subnormales y
otros patrones float32. Cada backend declara `prepare_padded_matmul(...)`; el
compilador enlaza el plan resultante sin condiciones por nombre de backend y
sin retener entradas. La ausencia o rechazo de esa capacidad conserva la ruta
habitual. Los buffers privados se rellenan y recortan en cada llamada; los pasos
siguientes siempre ven dimensiones lógicas. BLAS mantiene su proveedor explícito.
`blas="clblast"` usa el adaptador opcional `_clblast.py` sobre los mismos buffers:
GEMV/DOT nativos para casos vectoriales de rango 1/2 cuando sus símbolos están
presentes y GEMM para los demás. Los lotes usan GEMM batched/strided-batched nativo
cuando sus símbolos están disponibles, con offsets sobre los inputs originales.
El bucle GEMM del mismo proveedor queda como alternativa por capacidad ausente.

El hook opcional `prepare_layout_matmul(...)` permite al compilador consultar al
backend durante la preparación, sin condiciones por nombre de backend. OpenCL
con CLBlast admite transpuestas completas 2D privadas consumidas únicamente por
un matmul float32/64. El compilador verifica forma invertida, offset cero y strides
de una transpuesta completa; no interpreta un slice arbitrario como transpuesta.
`_matmul_layout.py` conserva metadatos y backend, no arrays de entrada, y ejecuta
GEMM con flags de transposición y leading dimensions del almacenamiento original.
Las nuevas entradas se reenlazan en cada llamada. La materialización se conserva
si el temporal tiene otro consumidor o es una salida pública, si hay dimensiones
vacías o si el backend rechaza la capacidad. El padding explícito elegible tiene
precedencia. No se introducen vistas públicas ni epílogos dentro de CLBlast, y
evitar una copia no garantiza que la elección interna de kernels BLAS sea más rápida.

`_ops.py` centraliza ufuncs y resolución de tipos. `_fusion.py` captura valores
mediante la misma fachada ndarray y un backend de trazado. `_compiler.py` produce
planes tipados con LRU por función, metadatos, escalares estáticos y backend.
Reenlaza arrays y escalares por llamada sin conservar los inputs de captura.
La ruta posicional completa usa nombres de parámetros precalculados para evitar
reconstruir el binding; los demás casos conservan la firma general. Todas las
validaciones y claves de especialización siguen activas.
`_lowering.py` separa generación y ejecución de grupos por backend del scheduling
común del grafo. Cada backend proporciona opcionalmente `create_lowering()`;
el plan lo resuelve una vez y enlaza `execution_scope()` cuando existe. El mínimo
`execution_scope_min_steps` puede descartar ese scope al preparar un plan corto:
CUDA declara dos; un `padded_matmul` cuenta su GEMM y las copias físicas necesarias,
aunque aparezca como un solo paso lógico. Los demás conservan el comportamiento anterior. La ausencia
de factoría conserva primitivas ordinarias; un error de preparación se propaga.
No se consulta un registro central por nombre ni se negocia por cada nodo.
CUDA proporciona `create_lowering()` propio: `CUDALowering` compila con NVRTC los
grupos elementwise/map-reduce admitidos. OpenCL y Vulkan fusionan grupos elemento
a elemento. Ambos también fusionan
expresiones con su reducción cuando el valor intermedio tiene un único consumidor
y no es una salida pública: map-reduce acumula directamente, con parciales acotados
para reducciones grandes. Las referencias repetidas de un mismo consumidor, como
`x * x`, no obligan a materializar `x`; consumidores distintos y salidas públicas
conservan esa frontera. En CUDA, los productos matriciales siguen siendo pasos
separados. OpenCL custom y Vulkan además pueden fusionar el patrón privado
`b - A @ x` con GEMV cuando los dtypes flotantes coinciden. CLBlast conserva su frontera;
NumPy evalúa las primitivas del mismo
plan. `@fuse` usa el mismo núcleo pero retraza cada llamada y solo admite una salida
elemento a elemento. Las salidas son arrays ordinarios residentes.

`_vulkan.py` administra buffers device-local, staging y colas nativas Vulkan.
Compila GLSL a SPIR-V mediante un compilador externo. Conserva los buffers usados
por comandos pendientes hasta su finalización. Su pool comparte un límite de
128 MiB de asignaciones nativas y 128 buffers entre memoria de dispositivo y staging,
con reutilización exacta preferida, o la menor capacidad compatible hasta
`2*max(solicitado,256)`. El tamaño lógico, la capacidad del VkBuffer y los bytes
nativos asignados se contabilizan por separado. Se expulsan clases antiguas al
admitir nuevas reservas completadas. Un conjunto de formas que exceda esos límites
todavía puede provocar reservas repetidas. También reutiliza comandos y, en la
ruta de descriptor sets, pools y sets completados, con un máximo de 128 recursos
de cada tipo. Agrupa
dispatches de una llamada compilada en un command buffer, mantiene envío inmediato
eager y sincroniza al alcanzar 64 operaciones grabadas/pendientes. Una fence
reutilizable por contexto señala el trabajo previo al observar resultados;
esperas exitosas preceden a reciclar recursos, y la cola vacía no provoca espera
nativa. El pool tolera reentrada de finalizadores al buscar y liberar capacidades.
Las barreras
siguen separando dependencias dentro del command buffer. El readback prefiere memoria host coherente
y cacheada compatible, separada del pool de subida. Los mapeos de staging de lectura se
conservan con la asignación nativa, también mientras está en el pool completado;
se desmapean antes de liberar/expulsar esa memoria. Se mantienen barreras, espera
previa a lectura y referencias de leases en vuelo. Las subidas conservan map/unmap
por copia, porque persistir todos los mapeos no mejoró consistentemente las
operaciones completas. Los kernels
incluyen reducciones/GEMV cooperativos, DOT largo con parciales y matmul por lotes
con bloques de registros para GEMM float32 elegible y tiles de 8×8 como alternativa.
Una LRU de 128 plantillas conserva GLSL, ABI y geometría; las estructuras nativas
de descriptor sets se reutilizan solo después de completar su uso. También se
reutilizan estructuras de envío y barreras bajo el lock. Si se detectan y habilitan
`VK_KHR_push_descriptor` y su dependencia properties2, los grupos dentro de
maxPushDescriptors escriben descriptores directamente en el command buffer.
Los registros host se pueden reutilizar tras grabarlos; los buffers GPU mantienen
su vida útil hasta completar comandos. Los demás grupos usan descriptor sets
completados. No hay tuning por fabricante.
`_vulkan_commands.py` añade una tabla privada por dispositivo con entry points
y argumentos CFFI tipados para comandos frecuentes. El backend mantiene la vida
útil y las barreras; VkResult se comprueba y las llamadas ajenas a esa tabla siguen
usando el binding opcional. No se añadió una extensión binaria. Ver
[mediciones de esta frontera](vulkan-dispatch-results.md).

La caché de kernels en memoria está limitada a 128 variantes por contexto. PyOpenCL
puede mantener su propia caché en disco. Un lock protege la mutación de argumentos
y el enqueue del kernel; la cola mantiene el orden de ejecución. `asnumpy`, `item`
y `synchronize` esperan la finalización correspondiente.

Los tres solvers de `examples/solvers.py` usan la API pública, incluido
`get_backend(array)` para recuperar la instancia existente sin copiar datos. El estado vectorial permanece en dispositivo;
OpenCL y Vulkan fusionan actualizaciones y mantienen coeficientes del paso en GPU.
El estado del residuo usa una norma escalada que evita formar cuadrados extremos.
CG y descenso más pronunciado usan direcciones normalizadas para sus coeficientes.
Las decisiones de parada y validez siguen descargando escalares. El cálculo del
residuo real de cada candidato usa un plan de avance que agrupa actualización,
GEMV y norma. Curvatura agrupa GEMV/DOT y se valida leyendo directamente su
escalar, conservando el valor en GPU para la actualización tipada. Verifican el residuo
real en cada iteración y preservan los inputs. El estado de convergencia es parte
del resultado, no una excepción por agotar iteraciones.

## Decisiones

- **OpenCL primero:** expresa la intención de portabilidad entre fabricantes y funciona
  con el hardware local. Vulkan es un backend independiente con shaders propios y
  optimizaciones portables iniciales; no se presume paridad de rendimiento.
- **NumPy 2 como contrato:** se usa su resolución de tipos y broadcasting, sin ejecutar
  la aritmética GPU en NumPy. Los creadores documentados sí pueden generar inputs en host.
- **Compatibilidad acotada:** las funciones fuera del contrato fallan antes de una
  conversión oculta. El almacenamiento es de solo lectura desde la API pública para
  evitar prometer vistas y semántica de aliasing que aún no están implementadas.
- **Instalación CPU mínima:** NumPy es la única dependencia obligatoria; los extras
  `opencl`, `vulkan` y `cuda` son independientes, con errores accionables si falta
  el driver/runtime. Ejemplos y benchmarks no se incluyen en el wheel.
- **Medición honesta:** compilación, transferencia y cálculo se miden por separado;
  el baseline usa NumPy directamente y no se declara que toda GPU será más rápida.

## Verificación

Las mismas pruebas de conformidad se ejecutan en el backend NumPy y en OpenCL, con
resultados calculados por NumPy. Se incluyen precisión simple/doble, tipos enteros,
broadcasting, matrices rectangulares, dimensiones vacías, NaN/Inf y errores de API.
Los solvers externos de `examples/solvers.py` se contrastan con
`numpy.linalg.solve` y con residuos independientes.
Vulkan tiene pruebas específicas de fusión, tipos, lotes, reducciones, solvers y
vida útil de recursos pendientes. La verificación local no sustituye las pruebas
con otro fabricante.

CI incluye CPU en distintos sistemas y OpenCL de CPU con PoCL en Linux. PoCL comprueba
otra implementación del estándar; no sustituye la validación en GPUs AMD/Intel/NVIDIA.
`PACHO_REQUIRE_OPENCL=1` evita que una configuración de CI mal instalada omita las pruebas.
