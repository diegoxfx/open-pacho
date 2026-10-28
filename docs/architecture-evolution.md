# Dirección arquitectónica acordada — 2026-09-12

OpenPacho es una librería de operaciones de arrays y matrices en GPU con API
cercana a NumPy y elección explícita de backend. Los algoritmos de dominio son
aplicaciones: solvers, redes neuronales y simulaciones pertenecen a ejemplos y
benchmarks. Esta decisión delimita el producto; no exige abandonar sus pruebas.

## Estado de esta revisión

- Implementado: solvers y sistemas preparados movidos a `examples/solvers.py`,
  fuera del wheel y de los exports del paquete. Los ejemplos usan API pública;
  `get_backend(array)` obtiene su instancia existente sin transferencias.
- Ya existía: NumPy como única dependencia obligatoria; extras independientes
  `opencl`, `vulkan` y `cuda`, con importación diferida de runtimes GPU.
- El resto de este documento define trabajo futuro. No se han implementado aquí
  nuevos contratos de capacidades, layouts, modos matemáticos, autotuning ni un
  ejecutor nativo Vulkan. Los resultados históricos conservan su código original.

La [iteración posterior de dispatch Vulkan](vulkan-dispatch-results.md) implementa
una tabla CFFI tipada y un diagnóstico separado. Sigue sin grabador nativo C/C++
de grupos; no cambia el estado pendiente de layouts/modos/autotuning.
La [ronda entre backends](backend-sharing-results.md) añade variantes seleccionadas
por capacidad y comparte el GEMM tiled con CUDA. Todavía no constituye el contrato
general de capacidades propuesto aquí.

La [ronda de routing y finalización](routing-completion-results.md) implementa dos
fronteras acotadas: adaptadores RNG propios de cada backend y factoría opcional
de lowering, resuelta al preparar el plan. No constituye todavía un protocolo
general de layouts/capacidades ni un mecanismo público de plugins.

## Separación y coste de abstracción

Se evoluciona la separación existente, sin añadir una cadena de proxies por
operación ni sustituirla por un compilador general de Python:

1. **Fachada semántica:** operaciones, formas, broadcasting, promoción y errores.
2. **Planificación opcional:** IR de operaciones genéricas, layouts, usos, fusión
   y decisiones de materialización. Eager sigue siendo una ruta independiente.
3. **Contratos internos:** protocolo tipado de primitivas y descripción inmutable
   de capacidades consultada al crear contexto/especializar un plan. Incluir
   dtype, límites, layouts y garantías numéricas por operación; un booleano
   «soporta float64» no describe todas sus funciones.
4. **Ejecución por backend:** buffers, vida útil, colas, kernels y proveedores.
   Resolver implementaciones al preparar el plan y enlazar llamadas directamente;
   mantener solo la validación necesaria por llamada. No inferir capacidades de
   nombres de fabricantes ni repetir negociación dinámica por elemento/dispatch.

No existe una primitiva «solver» en el contrato. Un backend habilita un ejemplo
cuando tiene sus operaciones necesarias. CUDA ya dispone de normas estables
float32/64 para ejecución eager y compilada, con proveedor custom o cuBLAS; los
algoritmos siguen en examples. La matriz de documentación debe derivarse/comprobarse
con esos contratos y tests, sin convertir «implementado» en «validado en todos los
dispositivos».

Contexto/almacenamiento y proveedor BLAS son responsabilidades distintas. Dejar
una frontera clara facilita kernels y pruebas; compartir arrays entre instancias
de proveedor o transferir directamente entre APIs es de **baja prioridad**. No
se introduce migración automática de backend ni hace falta resolver interoperabilidad
OpenCL/Vulkan/CUDA para optimizar cada uno.

## Kernels custom, proveedores y layouts

Kernels propios competitivos para GEMM, GEMV, DOT, reducciones y sus composiciones
son trabajo principal continuo. CLBlast/cuBLAS se conservan como proveedores y
referencias; ganar en un solver fusionado no demuestra ganar en GEMM aislado.
Evaluar precisión, memoria y distribución de tamaños, incluidos rectangulares,
impares y no potencias de dos. «Mayoría» exige una suite predefinida y publicar
regresiones/casos mixtos, no contar solo formas favorables.

Empezar layouts por transpuestas lógicas 2D y flags de BLAS, con kernels capaces
de consumirlos. Elegir packing/materialización según el consumidor, reutilización
y memoria: para k usos, comparar `T_pack + k*T_packed` con `k*T_transposed`.
Incluir packing en tiempo completo y memoria máxima; no ocultarlo en preparación.
Una copia también puede ser necesaria si el consumidor no acepta ese layout:
usar una alternativa válida o materializar explícitamente, sin prometer que jamás
se copia. No se amplía todavía a vistas mutables/aliasing completo de NumPy.

## Vulkan: diagnóstico antes de sustituir el adaptador

La brecha medida no establece un muro teórico de Vulkan. Mezcla preparación
Python/CFFI, administración propia, llamadas al driver, GPU y esperas. La mejora
ya comprobada del pool muestra margen de implementación; no cuantifica qué
porcentaje restante corresponde al binding Python.

Prioridad: medir por separado tiempo de host, llamadas nativas y timestamps GPU;
contadores de asignación, descriptores, envíos, bytes y memoria. No llamar «tiempo
Python» a pared menos duración de kernels: hay colas, driver y esperas intermedias.
Khronos documenta [reutilización y coste de command buffers](https://docs.vulkan.org/samples/latest/samples/performance/command_buffer_usage/README.html)
y [responsabilidades de sincronización del host](https://docs.vulkan.org/guide/latest/threading.html).

Si el perfil lo justifica, construir un ejecutor nativo pequeño de comandos,
descriptores, pools y envío que consuma el mismo plan. Compararlo con el adaptador
actual usando mismos kernels, inputs, barreras, precisión, política de memoria y
salidas. Publicar también latencia, errores y coste de empaquetarlo por SO.
Conservar protección de recursos en vuelo y locks donde correspondan. Extenderlo
solo con ganancia reproducible; un binding propio completo no es requisito previo.
Ni esta ruta ni Vulkan prometen superar limitaciones del driver/hardware.

## Integridad de ejemplos y comparaciones

Las mejoras deben activarse mediante operaciones ordinarias y APIs públicas
documentadas (`compile` opcional), dentro del compilador/backend. No reconocer
nombres de algoritmos, cambiar frecuencias de chequeo, tolerancias o datos para
favorecer OpenPacho. Las fases compile/prepare preexistentes en los solvers se
conservan y se declaran; no se presentan como ejecución directa sin adaptación.

Una mejora de algoritmo requiere justificación y comparación separada con
referencias equivalentes. Comparar eager/compilado, primera ejecución, preparación,
residente y extremo a extremo, con mismo trabajo o convergencia y error comparable.
La velocidad de CuPy es una referencia práctica, no un límite físico demostrado.
Igualarla sería valioso por la portabilidad, selección de backend e integración;
CuPy también ofrece una interfaz de arrays de alto nivel.

El alcance no incluye autodiff, entrenamiento distribuido ni todas las
[transformaciones de JAX](https://docs.jax.dev/en/latest/key-concepts.html).
Hay solapamiento en operaciones de arrays y compilación; comparar esos workloads
puede ser útil sin convertir la paridad con JAX en objetivo del producto.

## Modos matemáticos y autotuning: tracks futuros

Conservar por defecto el contrato actual y diseñar políticas explícitas de
aritmética rápida y acumulación más robusta. Especificar por operación: dtype de
entrada/acumulación/salida, contracción FMA, reasociación, aproximaciones de
trascendentes, subnormales, NaN/Inf, ceros con signo y permisos para precisión
reducida. Una política «precisa» no promete reproducibilidad bit a bit ni resultados
representables cuando no caben en el dtype. Rechazar garantías no soportadas.

FMA calcula multiplicación más suma con un solo redondeo y puede mejorar tanto
precisión como velocidad. MAD no es una garantía portable equivalente a FMA
correctamente redondeada; ver [funciones matemáticas de OpenCL C](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html#math-functions).
No tratar «desactivar FMA» como sinónimo universal de precisión. Políticas futuras
deben participar en claves de kernels/planes/tuning y metadatos de benchmarks;
no comparar fast-math contra modos conservadores como si fueran el mismo contrato.

Autotuning es un track secundario, con activación/presupuesto explícitos, búsqueda
acotada y ruta portable sin calibración. Elegir entre variantes correctas de
tile, grupos, vectorización, packing y fusiones según medidas. Cachear por
dispositivo, driver/compilador, versión de kernels, operación, shape/layout, dtype
y política matemática; invalidar ante cambios y no extrapolar un ganador local a
todo AMD/NVIDIA/Intel. Separar coste de tuning de ejecución sostenida. Si el usuario
fija proveedor, el tuning respeta esa selección.

## Validación y expansión

La primera publicación debe anunciar precisamente su matriz validada. Que la
comunidad aporte AMD/Intel u otros SO no impide probar ahora wheel CPU, aislamiento
de extras y CI reproducible. No confundir conformidad PoCL CPU con validación GPU.
Una versión estable de API puede tener un soporte de hardware aún experimental.

Añadir redes densas y simulaciones físicas como cargas de ejemplo según el
[plan aplicado](application-measurement-plan.md), con errores y memoria además
de tiempos. No añadir frameworks de dominio al paquete ni imponerlos como dependencias.

Para cambios importantes, usar una auditoría de documentación acotada y de solo
lectura con un subagente económico (Luna/Spark disponible): identificar archivos,
afirmaciones y enlaces que necesitan cambiar. El agente principal decide, edita y
verifica. Los informes históricos se conservan como cortes fechados; la auditoría
no crea una automatización permanente ni sustituye pruebas.
