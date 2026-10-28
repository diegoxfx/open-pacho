# Camino hacia CUDA y otros backends

Decisión de arquitectura del 2026-09-08, actualizada el 2026-09-18. Ya existe un
[backend CUDA inicial](cuda.md): contexto, memoria, NVRTC, primitivas y cuBLAS.
Las comparaciones CuPy/cuBLAS/nvmath siguen siendo referencias externas.
Las etapas siguientes conservan el plan de ampliación; no representan paridad
ya implementada con OpenCL/Vulkan.

## Base implementada

`_compiler.py` conserva captura, especialización, decisiones de materialización,
dependencias y vida de temporales. Los adaptadores de lowering generan y ejecutan
los grupos fusionados por backend. `create_lowering` en `_lowering.py` solicita una vez
la factoría opcional `backend.create_lowering()`; OpenCL, Vulkan y CUDA ofrecen sus
adaptadores (`CUDALowering` en CUDA). NumPy, sin factoría, ejecuta sus propias
primitivas sin fusión.
El plan enlaza también `execution_scope()` si existe. Los fallos de preparación
se propagan; no autorizan un cambio de backend ni una copia a CPU.

Cada adaptador prepara los grupos elementwise/map-reduce y los ejecuta recibiendo
los operandos actuales. Su payload puede guardar metadatos y código, nunca arrays
de una llamada ni valores escalares enlazados. Las reglas de broadcasting, dtype,
intermedios compartidos y salidas públicas siguen siendo comunes. Las condiciones
del epílogo GEMV respetan al proveedor y el árbol de reducción correspondiente.

Hay pruebas con un adaptador independiente ejecutado en CPU para comprobar
reenlace escalar, liberación de inputs y salidas compartidas sin asumir un nombre
GPU concreto; la suite de GPU verifica las implementaciones reales. Es una
interfaz privada, no una API de plugins estabilizada.

## Implementación CUDA por etapas

1. **Contexto y memoria — base implementada.** Módulo opcional `_cuda.py`, descubrimiento explícito,
   contexto primario retenido por dispositivo, stream propio, buffers y pool
   acotado. Aislar arrays de otros contextos y conservar referencias en vuelo.
   Importar OpenPacho en CPU no carga CUDA. `upload` aísla la entrada host antes
   de retornar; `asnumpy`, `item` y `synchronize` observan el stream correcto.
2. **Primitivas y BLAS — parcial.** Elementwise, gather y reducciones flotantes
   ya se ejecutan en CUDA, incluido matmul con lotes/broadcast y tiling float32.
   [GEMV/DOT cooperativos custom](three-actions-results.md) ya están implementados;
   cuBLAS ya selecciona DOT para dos vectores y GEMV para matrices con batch
   explícito, según las [mediciones de selección](three-iterations-results.md).
   Normas estables float32/64 están implementadas para ejecución eager y compilada,
   con kernels propios y disponibles tanto con `blas="custom"` como con
   `blas="cublas"`; productos conservan cuBLAS y la opción custom observable.
   Extender la validación de `blas` por backend: `clblast` no es un proveedor CUDA.
   No inventar una emulación CPU cuando falte un símbolo o una capacidad.
3. **Compilación — elementwise/map-reduce implementados.** `CUDALowering`
   implementa la interfaz privada común y usa NVRTC para los grupos admitidos.
   La caché queda acotada por código, tipos, geometría, dispositivo y opciones
   numéricas. Preserva los redondeos de elementwise fusionado, el ABI escalar y
   el árbol de reducción de 128 lanes/chunks de hasta 4096 términos, con
   `fmad=false` y `ftz=false`. Matmul y BLAS siguen siendo fronteras de proveedor;
   acumuladores bool y graphs CUDA siguen pendientes. Las normas estables necesarias
   por los ejemplos externos están implementadas.
   Ver los [resultados de fusión CUDA](cuda-fusion-results.md). Reutilizar decisiones
   del grafo; no traducir GLSL a CUDA mecánicamente.
4. **Aplicaciones externas y empaquetado.** Los escalares por valor, el extra CUDA
   inicial y las normas estables están implementados. Los solvers externos de
   `examples/solvers.py` pueden usar CUDA; no se empaquetan en la librería ni sus
   algoritmos forman parte del runtime.
   Ejecutar las pruebas comunes y adversas con CUDA obligatorio;
   verificar instalación sin CUDA y mensajes de error antes de publicar el extra.
5. **Rendimiento.** Primero paridad numérica; después medir cuBLAS directo frente
   al adaptador OpenPacho sobre los mismos buffers y stream. Graphs, memoria
   pinned y kernels especializados se incorporan cuando un control demuestre
   amortización, sin cambiar la frecuencia de convergencia.

La implementación inicial usa `cuda.bindings` (Driver API/NVRTC) y
bindings de cuBLAS distribuidos con nvmath, detrás de un adaptador pequeño.
Ambos módulos están presentes en el entorno local usado para benchmarks; eso
no prueba todavía que el backend inicial sea instalable en todas las plataformas.
CuPy permanece como referencia independiente, evitando convertir el backend
OpenPacho en un envoltorio completo de CuPy. El extra inicial fija CUDA bindings
13.3+, NVRTC 13.3+, cuBLAS 13 y nvmath 1; ampliar versiones/plataformas requiere
verificación adicional. Fuentes:
[CUDA Python](https://nvidia.github.io/cuda-python/cuda-bindings/latest/index.html),
[NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html).

Por defecto se mantendrá float32/float64 sin TF32 ni fast-math implícito. Cualquier
modalidad de precisión reducida futura tendrá selección explícita, etiquetas en
el benchmark y métricas de error. No se promete identidad bit a bit entre árboles
de reducción o proveedores. Ver [modos de cálculo cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html).

## Otros backends y prioridad

| Candidato | Qué aportaría | Prioridad / condición |
| --- | --- | --- |
| CUDA | Bibliotecas y ejecución específicas NVIDIA; hardware local disponible para verificar | Base inicial implementada; ampliar gradualmente y conservar OpenCL/Vulkan independientes. |
| HIP/ROCm | Acceso a bibliotecas y ejecución nativas AMD | Después de disponer de AMD/SO soportado para medir; reutilizar contratos, no suponer portabilidad automática de cada kernel. |
| Metal | Acceso nativo a GPU Apple y sus bibliotecas | Cuando exista un Mac de prueba y demanda; no diseñarlo sobre supuestas garantías FP64. |
| SYCL/oneAPI | Toolchain C++ con varios backends y camino a ecosistemas Intel | Evaluar frente al valor incremental sobre OpenCL/Vulkan; mayor coste de build y distribución. |

HIP está alineado con el modelo CUDA, pero los detalles y optimizaciones pueden
requerir adaptación. [Documentación HIP](https://rocmdocs.amd.com/projects/HIP/en/latest/index.html).
Metal ofrece compute y bibliotecas optimizadas en plataformas Apple.
[Documentación Apple](https://developer.apple.com/metal/).
SYCL puede dirigirse a distintos backends según implementación; no es una API
de driver única. [Khronos SYCL](https://www.khronos.org/sycl/).

No se añaden estos backends simultáneamente. Cada uno multiplica pruebas de
precisión, sincronización, vida útil, instalación y proveedores; la prioridad
principal sigue incluyendo ejecución y transferencias Vulkan.

## Qué entendemos por límite físico

CuPy/cuBLAS y OpenPacho usan la misma GPU local. La ventaja de una biblioteca
optimizada es una referencia alcanzada, no una prueba de que OpenPacho haya
llegado al límite del hardware. Los límites de ancho de banda, cómputo por dtype,
latencia y capacidad son distintos; el overhead de Python y del driver se mide
por separado. NVIDIA describe precisamente esa separación en su
[guía de rendimiento CUDA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

Para cuantificar margen faltan mediciones de timestamps GPU, bytes realmente
movidos, tasas sostenidas de copia residente y throughput por forma/dtype. El
roofline debe usar esos techos medidos y la intensidad aritmética del workload.
El máximo comercial o la velocidad de CuPy no basta. Tampoco podemos adjudicar
la brecha actual a Tensor Cores: las comparaciones publicadas deshabilitan TF32.
