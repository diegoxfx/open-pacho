# Alcance y prioridades

OpenPacho es una librería de operaciones matriciales en GPU con API NumPy acotada,
backend explícito y compilación opcional. Solvers, sistemas preparados y futuras
aplicaciones de redes/simulación viven en `examples/`, fuera del paquete instalable.
La versión sigue siendo **0.1.0a1**, sin publicación en PyPI.

La [dirección arquitectónica acordada](architecture-evolution.md) define contratos,
precisión y reglas de medición. La [ronda de dispatch Vulkan](vulkan-dispatch-results.md) añade una frontera
CFFI tipada, un control alternado y diagnóstico con timestamps. Conserva kernels
y algoritmos. La [ronda entre backends](backend-sharing-results.md) añade push
descriptors Vulkan por capacidad, caché/índices OpenCL y lotes/tiles CUDA;
todavía hay margen en preparación/envío y no implica paridad con
OpenCL/CuPy. La [ronda CUDA inicial](cuda-initial-results.md) conserva su snapshot.

## Línea principal

La [ronda de estabilidad y kernels](kernel-stability-results.md) añade reposo
observado entre procesos y promueve la especialización de guardas OpenCL en tiles
completos. El resto de variantes conserva su condición experimental; no cambia
la prioridad de los gaps de ejecución siguientes.

La [ronda de staging y aleatorios](staging-random-results.md) conserva mapeos
Vulkan solo para readback, después de descartar la persistencia general como
predeterminada por resultados mixtos. P2 añade un RNG residente. Envíos, esperas
y agrupación de comandos siguen siendo prioridad principal.

La [ronda de routing y finalización](routing-completion-results.md) elimina el
dispatch duplicado del RNG y resuelve lowering/ámbito de ejecución al preparar
planes. GEMM/GEMV/normas aún contienen rutas por nombre: su migración será gradual,
cuando se amplíen o se midan esas operaciones, conservando la aritmética compartida.

La [ronda de fusión CUDA](cuda-fusion-results.md) rota el foco desde esperas Vulkan
a eliminar lanzamientos y temporales completos en CUDA. Implementa elementwise
y map-reduce sobre la factoría del backend, sin cambiar los algoritmos de dominio.

La [ronda de preparación de argumentos CUDA](cuda-dispatch-results.md) reutiliza
almacenamiento host por kernel. El ahorro residente se repite y el empaquetado
aislado ya cuesta menos de un microsegundo en el control local. La siguiente
investigación sale de ese subpaso: contexto/envío, binding y operaciones matriciales
ofrecen más margen pendiente que seguir afinando la misma preparación. La
[ronda de tres optimizaciones](three-actions-results.md) implementa binding
posicional común, contexto CUDA anidado con scope selectivo y GEMV/DOT cooperativos
custom. Conserva mediciones separadas por cambio y reconoce los resultados mixtos.

El orden siguiente es un conjunto de frentes, no una preferencia permanente por
una API. Para elegir iteración se considera frecuencia de uso, coste eliminable,
ganancia completa/residente, repetibilidad, precisión y esfuerzo. Si un ajuste solo
da mejoras pequeñas o mixtas en varias rondas, se prioriza otro con mayor ahorro
posible. Ese ajuste queda pendiente; no significa que el backend agotó su margen.
Las regresiones de corrección conservan prioridad sobre el rendimiento.

La [inspección de kernels](kernel-inspection-results.md) elimina conversiones
float64 accidentales en GEMM OpenCL. Las siguientes variantes deben partir de
esa base corregida: no extrapolar la antigua ventaja del padding. La preparación
de copias ya ofrece ahorros marginales; priorizar eliminar temporales/envíos y
mejorar aritmética/layouts antes de seguir afinando esas cachés.

La [reevaluación de padding](padding-reassessment.md) conserva la opción por
ganancias reproducidas en algunas formas de pocas filas y K largo. La mayoría
empeora; ampliar padding tiene baja prioridad. Priorizar GEMM rectangular sin
copias adicionales y estudiar la acumulación CUDA con K largo, donde existen
elementos fuera de tolerancia tanto con padding como sin él.

Las [tres iteraciones siguientes](three-iterations-results.md) añaden tiles
rectangulares sin copias en OpenCL/CUDA, primitivas vectoriales seleccionadas de
cuBLAS y flags de transposición para temporales privados compilados con CLBlast.
Los números de padding anteriores son históricos: una ruta directa más rápida
cambia su punto de equilibrio. No se extrapolan esas ventajas al runtime nuevo.

1. **Ejecución Vulkan y observabilidad:** descomponer preparación Python/CFFI,
   driver, GPU, copias y esperas. Mejorar comandos/descriptores/staging y memoria
   según evidencia. Un ejecutor nativo pequeño se evalúa con un control equivalente
   antes de decidir un adaptador completo.
2. **Kernels propios competitivos:** GEMM, GEMV, DOT, reducciones y composiciones
   frecuentes; objetivo continuo de acercarse o superar CLBlast en una suite
   representativa. Conservar CLBlast/cuBLAS por contrato y los resultados mixtos.
   CUDA requiere ampliar la selección medida de primitivas cuBLAS, tiling y reducir
   preparación/lanzamiento alrededor del lowerer fusionado ya disponible;
   no se exige paridad inmediata con los backends más avanzados.
3. **Layouts y planes:** ampliar los flags BLAS ya disponibles para CLBlast; materialización según
   consumidor, reutilización y memoria. Reducir intermediarios, binding y envíos
   sin cambiar el código de dominio ni el algoritmo de los comparadores. Medir
   también trabajo repetido al fusionar operaciones pequeñas con broadcasting grande.
4. **Contratos internos:** reemplazar gradualmente decisiones por nombre de backend
   por capacidades de operaciones, layouts y políticas numéricas. Resolver rutas
   al preparar planes y enlazar llamadas directas para contener el coste de abstracción.
5. **Aplicaciones y precisión:** ampliar covarianza/ridge con redes densas sencillas
   y simulaciones; seguir midiendo rectangulares, impares y no potencias de dos.
   Conservar benchmarks de solvers hasta convergencia como clientes externos.
   Publicar error, preparación, primera llamada, memoria y tiempo residente/completo.

Estas líneas se priorizan con perfiles; una regresión general comprobada precede
nuevas optimizaciones. Mejorar el ejecutor no autoriza cambiar tolerancias,
frecuencia de chequeo ni especializar por nombre de un solver.

## Tracks secundarios y futuros

- **P2, generación aleatoria:** [primera API residente](random.md) para uniforme,
  normal y exponencial mediante Philox4x32-10. Pendientes: distribución entera
  sin sesgo, spawn/checkpoint, baterías estadísticas extensas y aprovechar varias
  salidas por hilo sin romper el contrato de contadores. No desplaza ejecución
  Vulkan ni kernels matriciales. Aleatoriedad en compile exige modelar efectos.

- **Autotuning acotado:** opt-in/presupuesto explícito, validación de candidatos,
  caché invalidable por dispositivo/driver/kernel/shape/layout/dtype/modo matemático.
  Respetar proveedor explícito y mantener ruta portable sin calibración.
- **Padding de reducciones:** `mean` puede sumar con ceros y dividir por el tamaño
  lógico de los ejes reducidos. Medir división separada frente a fusión final;
  conservar dtype, ejes, vacíos y no finitos. [Pendiente, no implementado](padding.md#extensión-futura-reducciones).
- **Hardware/SO:** [especialización por familias](hardware-specialization.md) cuando
  se pueda verificar localmente. Separar optimizaciones opcionales de workarounds
  necesarios para bugs comprobados; no extrapolar a fabricantes completos.
- **Políticas matemáticas:** modos rápidos y acumulación más robusta, con garantías
  por operación, FMA/reasociación/subnormales y precisión reducida explícitos.
  Participan en las claves de caché y en cada comparación; todavía no hay flags públicos.
  La [investigación de acumulación GEMM](gemm-accumulation-results.md) conserva
  variantes por bloques y compensadas como experimentos: mejoran precisión en
  distintos casos, con costes y límites diferentes. No cambia el valor predeterminado.
- **Portabilidad comunitaria:** matriz de hardware/driver/SO comprobada. Preparar
  informes reproducibles; AMD/Intel físicos dependen de equipos adicionales o
  contribuciones. CI CPU/PoCL y empaquetado siguen siendo verificables por el proyecto.
- **Interoperabilidad:** compartir almacenamiento entre proveedores/contextos o
  mover directamente entre APIs tiene baja prioridad. La elección de backend es
  explícita; no se introducen migraciones automáticas.
- **Dispersas, vistas mutables y compatibilidad total NumPy:** fuera de esta etapa.
  Autodiff, solvers como API de biblioteca y frameworks de dominio no son objetivos.

## Proveedores e instalación

NumPy es la única dependencia obligatoria. OpenCL, Vulkan y CUDA son extras
independientes; un usuario OpenCL no instala CUDA. Las capacidades ausentes
producen una alternativa válida dentro del backend/contrato elegido, o un error
explícito. No cambiar dtype, precisión o backend silenciosamente.

CLBlast ya tiene GEMM batched/strided-batched y GEMV/DOT cuando los símbolos están
disponibles; custom OpenCL/Vulkan ya usa tiles locales/de registros. CUDA tiene
memoria, NVRTC, cuBLAS, lotes, GEMM custom float32 tiled, fusión elementwise/map-reduce
y normas estables float32/64 eager/compiladas para ambos proveedores BLAS. La API
común también incluye selección residente `where(condition, x, y)` con broadcasting
de tres argumentos y composición con fusión elementwise/map-reduce; graphs siguen
pendientes. Consulta los [resultados de selección](selection-results.md). Los informes
históricos describen cada etapa y no sustituyen el [contrato actual](compatibility.md).

La [iteración de extremos](extrema-results.md) añade min/max residentes por ejes,
conservación de dtype y composición compilada. Las primeras recetas de softmax
y logsumexp quedan en `examples`, como clientes de esas primitivas. No incorporan
autodiff ni una API de redes en la biblioteca.

La [preparación de extremos materializados](extrema-planning-results.md) promueve
la ruta cooperativa existente para min/max float32 compilados en OpenCL/CUDA.
Dos rondas reproducen la mejora residente y muestran ganancias completas desiguales;
Vulkan conserva su ruta por regresiones repetidas.

El [cliente de red densa](dense-network-results.md) ya ejercita inferencia y
gradientes explícitos usando las primitivas existentes. Conserva NumPy, CuPy y
los proveedores nativos, casos pequeños/regulares/impares y errores explícitos.
La brecha float64 de CUDA custom justifica estudiar su GEMM directo, después de
separar ese coste del resto del flujo; no añade una API de redes al runtime.

El [experimento GEMM float64](cuda-fp64-gemm-results.md) confirma el peso de
GEMM, pero rechaza promover un tile compartido 16×16: dos rondas no mejoran
consistentemente el flujo y repiten regresiones estrechas.

El [experimento de contracción float64](fp64-contraction-results.md) aísla FMA
dentro de cada geometría y compilador, conservando las políticas predeterminadas
del runtime. La prioridad siguiente es reproducir su efecto en inferencia y SGD
completos de CUDA custom, con selección experimental explícita y presupuesto de
error. Solo una ganancia aplicada repetida justificaría diseñar una política
pública opt-in;
el microbenchmark no promueve un tile ni demuestra mayor precisión en todo caso.

Los [osciladores acoplados](oscillators-results.md) añaden un cliente de pasos
fijos con GEMV repetido y un paso compilado reutilizable. La trayectoria modal
y la energía distinguen error del integrador de diferencias de backend. El
modelo permanece en ejemplos, sin ampliar la API pública con física o bucles GPU.
Sus tiempos justifican, como frente independiente, descomponer despacho,
asignaciones y esperas del ejecutor preparado antes de optimizarlas.

CuPy/cuBLAS/nvmath son referencias de rendimiento y precisión; PyTorch/JAX se
incorporan cuando un workload comparable justifique la dependencia de benchmark.
No son dependencias obligatorias ni se usa su tiempo como techo físico universal.

## Publicación y documentación

Antes de publicar: revisar API, nombre/versión, wheel/sdist, errores por capacidades
faltantes y matriz validada. Anunciar separadamente estabilidad de API y soporte
experimental de dispositivos no comprobados; no prometer validación universal.
La [estimación de release](release-readiness.md) reserva 2–4 iteraciones de
endurecimiento para una candidata acotada, seguidas de validación externa antes
de declararla estable; no exige alcanzar paridad de rendimiento con CuPy.

Para cambios grandes, una auditoría económica de solo lectura identifica documentos
impactados; el agente principal integra y verifica. Mantener consistentes README,
compatibilidad, capacidades e instalación, conservando resultados históricos.
Ver [verificación local](verification.md) y [próximas aplicaciones](application-measurement-plan.md).
