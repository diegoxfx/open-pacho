# Vulkan: frontera CFFI tipada y diagnóstico — 2026-09-12

## Resultado y alcance

La tabla de comandos por dispositivo reduce el coste de los wrappers generados
en llamadas frecuentes. Sigue siendo un ejecutor Python con llamadas CFFI; no
se añadió una extensión C/C++ ni se reescribió el adaptador Vulkan completo.
No cambia el API público de arrays, kernels de los workloads medidos, barreras,
pools, número de envíos, chequeos de solvers ni precisión.

Control local: RTX 5060 Ti, Windows 11, driver 616.64 y binding Vulkan 1.3.275.1.
Dos procesos independientes, 30 casos por proceso, rutas wrapped/direct intercaladas
con 21 muestras por modalidad. Incluye float32/64, rectangulares/impares,
operaciones, aplicaciones, solvers hasta convergencia y transferencias escalares/grandes.
Los rangos siguientes son las **dos medianas**, no intervalos de confianza.

| Caso | dtype | Wrapped ms | Direct ms | Reducción de tiempo |
| --- | --- | ---: | ---: | ---: |
| gemv 2049x1023 | float32 | 0.309–0.316 | 0.277–0.314 | -1.8–12.3 % |
| gemv 2049x1023 | float64 | 0.353–0.366 | 0.335–0.338 | 5.2–7.7 % |
| gemm 257x129x513 | float32 | 0.266–0.270 | 0.245–0.247 | 7.7–8.4 % |
| gemm 257x129x513 | float64 | 0.393–0.409 | 0.358–0.372 | 8.9–9.2 % |
| covariance 4097x129 | float32 | 0.771–0.792 | 0.658–0.666 | 13.5–16.9 % |
| covariance 4097x129 | float64 | 1.093–1.129 | 0.997–1.002 | 8.3–11.6 % |
| ridge 4097x129 | float32 | 7.024–7.101 | 5.424–5.624 | 20.8–22.8 % |
| ridge 4097x129 | float64 | 7.151–7.369 | 5.466–5.602 | 23.6–24.0 % |
| cg 513x513 | float32 | 14.922–15.997 | 12.435–13.650 | 14.7–16.7 % |
| cg 513x513 | float64 | 24.392–24.735 | 20.141–20.338 | 16.6–18.6 % |
| jacobi 513x513 | float32 | 18.653–20.292 | 15.155–15.832 | 18.8–22.0 % |
| jacobi 513x513 | float64 | 31.548–32.179 | 25.454–25.595 | 18.9–20.9 % |
| sd 513x513 | float32 | 30.513–32.963 | 25.049–26.119 | 17.9–20.8 % |
| sd 513x513 | float64 | 57.564–59.371 | 46.700–48.915 | 17.6–18.9 % |

En estos controles, los solvers reducen tiempo residente entre 14,7 y 22,0 %;
ridge entre 20,8 y 24,0 %. GEMV float32 queda mixto entre procesos; la subida
grande puede quedar igual o ligeramente peor. No se afirma una mejora universal.

El modo completo también incluye subida, preparación y descarga. Para ridge
4097×129 float32, las medianas completas son:

- Wrapped: 7.514–8.054 ms.
- Direct: 5.809–6.312 ms.

Las transferencias aisladas miden el API de upload/download con staging y
sincronización, no el ancho de banda PCIe puro. La tabla completa local
(`.artifacts/vulkan-dispatch/comparison.csv`, no distribuida)
conserva ambas corridas, todos los casos, errores L2 e iteraciones.

## Qué cambió

`_vulkan_commands.py` resuelve una tabla por VkDevice mediante
[vkGetDeviceProcAddr](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeviceProcAddr.html).
El backend pasa handles, arrays nativos y NULL tipados: evita inspección/conversión
genérica por argumento y utiliza entry points del dispositivo. Se mantiene el
binding opcional existente para creación, asignación y otras llamadas.

- Las funciones VkResult comprueban errores y conservan excepciones del binding.
- La tabla no modifica el módulo Vulkan global ni comparte punteros entre VkDevices.
- `auto` usa wrappers si no puede resolver la ruta tipada; no cambia de backend.
  Un error durante ejecución se propaga, sin reintento o fallback que lo oculte.
- Comandos, leases, recursos en vuelo, locks, staging, barreras y límite de 64
  operaciones conservan su contrato. No se relajan validaciones de entradas.
- `_command_mode` es un control privado de constructor para diagnóstico, no una
  configuración que el usuario necesite conocer para obtener la mejora.

`wrapped` es el binding generado con los mismos records tipados del runtime
actual; no es un wheel antiguo byte a byte. Las mediciones iniciales `before-gaps`
y `direct-gaps` fueron exploratorias y no sustentan la tabla causal.

## Qué permite ver el diagnóstico

`benchmarks/vulkan_dispatch.py` registra fuera de las muestras de rendimiento:

- Tiempo host de cada API (incluye binding, driver y posibles esperas).
- Timestamps GPU alrededor de dispatch y copia; respeta timestampValidBits,
  período, wraparound, reset de queries y vida útil hasta completar comandos.
- Conteos de dispatch, barreras, descriptores, envíos, esperas, mapas y asignaciones.
- Bytes copiados, asignaciones nuevas y caché retenida. **No mide pico total de VRAM.**

Los timestamps usan TOP/BOTTOM_OF_PIPE y añaden comandos/dependencias. Sus
intervalos incluyen efectos de instrumentación/etapas; no son tiempos puros de
shader ni se restan de la pared sin instrumentar para inventar «tiempo Python».
Ver [timestamps de Vulkan](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp.html).

Ejemplo diagnóstico de la primera corrida, ridge float32: los 80 dispatches,
80 barreras, 20 envíos y dos esperas son idénticos. El total medido dentro de APIs
pasa de aproximadamente 3,16 a 1,65 ms; los intervalos GPU suman 0,352/0,355 ms.
Estas magnitudes pertenecen a una invocación instrumentada y no sustituyen sus
medianas de pared 7,10/5,62 ms. Señalan margen en ejecución/preparación; no establecen
un límite físico de Vulkan, NVIDIA o PCIe.

## Corrección numérica independiente

La prueba exploratoria de energía usando `** 2` descubrió NaN para bases negativas.
No era un error del nuevo dispatch: GLSL define
[pow fuera de dominio para x<0](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html#exponential-functions).
La fachada prometía comportamiento real NumPy para exponentes enteros representables.

Se añadió un helper float32 que distingue exponentes enteros/impares, ceros con
signo, NaN e infinitos y evita llamar pow fuera de su dominio definido. Se prueba
eager, compile y map-reduce con rebinding de escalares. Power float64 continúa
fuera de alcance. El control de energía usa el `y*y` del benchmark existente;
la corrección de potencia no se presenta como ganancia de velocidad del dispatch.

Todas las salidas wrapped/direct y las salidas instrumentadas coinciden en los
casos finales; se conservaron métricas de error float64, iteraciones y conteos.
Para solvers se valida el residuo verdadero de los inputs representados: la
tolerancia de parada no garantiza una tolerancia relativa por componente de la
solución. El error hacia la solución float64 se publica adicionalmente.

La suite adversa Vulkan se repitió con tres muestras: 42 configuraciones, diez
alertas y cero excepciones. Son las mismas diez alertas del subconjunto Vulkan de
la ronda CUDA inicial. Siguen visibles las limitaciones de cancelación,
subnormales y sistemas mal condicionados; una mejora del ejecutor no las corrige.

## Comparación aplicada complementaria

Se ejecutó la suite existente de covarianza y ridge para matrices 4095×127 y
4097×129, float32/64: 72 configuraciones y siete muestras, sin errores ni backends
ausentes. La tabla muestra medianas residentes en ms para 4097×129. OpenPacho usa
compile; CuPy ejecuta su versión existente de las mismas operaciones. NumPy se
mide con uno y ocho hilos. Ridge mantiene veinte pasos en todos los proveedores.

| Aplicación | dtype | NumPy 1t | NumPy 8t | OpenCL custom | OpenCL CLBlast | Vulkan | CuPy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Covarianza | float32 | 1.690 | 1.099 | 2.081 | 1.538 | 1.179 | 1.131 |
| Covarianza | float64 | 2.740 | 1.784 | 3.044 | 3.170 | 1.702 | 1.858 |
| Ridge | float32 | 1.530 | 2.971 | 6.042 | 6.671 | 6.694 | 5.407 |
| Ridge | float64 | 2.634 | 3.525 | 7.542 | 8.628 | 6.586 | 6.088 |

En esta muestra Vulkan está cerca de CuPy en covarianza y del OpenCL compilado
en ridge. NumPy de un hilo sigue ganando claramente en ridge de este tamaño.
La pequeña diferencia Vulkan/CuPy de covarianza no establece una victoria estable:
es una corrida complementaria, no una repetición pareada entre procesos. Tampoco
se deben mezclar sus tiempos absolutos con los del control wrapped/direct.

El JSON conserva todas las modalidades, p10/p90, preparación, transferencias,
tiempo completo y error numérico. Las modalidades usan muestras independientes;
sus medianas no son componentes sumables de una misma invocación. En covarianza
float32, el error L2 relativo de OpenPacho GPU ronda 7,9e-7 frente a aproximadamente
9,3e-8 de CuPy: acercar tiempos no implica igualar precisión. Para float64 la
referencia de esta suite tiene la misma precisión, no certifica exactitud.

## Reproducir

```powershell
$env:PACHO_VULKAN_COMPILER='<ruta a glslang o glslc>'
python -m benchmarks.vulkan_dispatch --repeats 21 --output .artifacts/vulkan-dispatch/run-a.json
python -m benchmarks.vulkan_dispatch --repeats 21 --seed 4782 --output .artifacts/vulkan-dispatch/run-b.json
python -m benchmarks.accuracy --backends vulkan --repeats 3 --output .artifacts/vulkan-dispatch/accuracy-new.json
python -m benchmarks.applications --irregular --backends opencl,opencl-clblast,vulkan,cupy --repeats 7 --output .artifacts/vulkan-dispatch/applications-new.json
```

Usar archivos nuevos. No ejecutar otras cargas CPU/GPU durante la medición.
Warmup, referencias, descarga de validación y diagnóstico quedan fuera del reloj.
Los solvers son aplicaciones de `examples/`; ridge conserva veinte pasos, no una
afirmación de convergencia. Los JSON conservan primera llamada y preparación
aplicada; no son un estudio de arranque frío de proceso.

Resultados finales: `.artifacts/vulkan-dispatch/final-{1,2}.json`. Las corridas
`quick*` y `paired-*` conservan exploraciones/fallos del runner. `paired-1` aplicaba
incorrectamente validación por componente a un solver; el criterio corregido usa
su residuo contractual sin cambiar el algoritmo, tolerancias ni datos cronometrados.
Ver [verificación](verification.md) para tests, precisión adversa y empaquetado.

## Siguiente margen de mejora

1. Reducir preparación/grabación y número de cruces Python→C por grupo de operaciones;
   el siguiente experimento puede ser un grabador nativo pequeño con el mismo plan.
2. Medir esperas y staging escalar sin cambiar el control de convergencia de ejemplos.
3. Mantener kernels/layouts y mejoras de compilador genéricas como líneas paralelas.
4. No atribuir a hardware los tiempos de driver/cola aún no aislados. Un adaptador
   completo nuevo, autotuning por familia y nuevas APIs permanecen decisiones posteriores.
