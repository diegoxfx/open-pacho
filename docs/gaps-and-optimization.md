# Gaps reales y optimización de OpenPacho

**Snapshot antes de la siguiente mejora de ejecución Vulkan.** Los resultados de
readback cacheado, descriptor sets reutilizables y envíos agrupados están en
[la continuación de la prioridad principal](vulkan-execution-results.md).
La etapa más reciente implementa [Vulkan, GEMM con registros y estabilidad numérica](numerics-vulkan-results.md).
La [iteración intermedia](remaining-suite-results.md) conserva map-reduce, preparación y BLAS batched.
Las especializaciones hardware/SO forman un [track de menor prioridad](hardware-specialization.md).

Fecha: 2026-09-06. RTX 5060 Ti 16 GB, Ryzen 7 5700X3D, 32 GB RAM, Windows 11,
driver 616.64. Este informe continúa el [snapshot anterior](execution-results.md).

**Hay margen importante dentro de OpenPacho. Las transferencias explican algunas
derrotas, pero no explican los solvers lentos con datos residentes ni el overhead
de Vulkan.** Un backend portátil tampoco necesita implementar todo NumPy para
aportar valor: el objetivo sigue siendo encadenar cálculos frecuentes sobre datos
residentes, con semántica conocida y selección explícita de backend/proveedor.

## Qué se implementó

- GEMV OpenCL cooperativo, con lecturas contiguas entre hilos. DOT corto usa un
  lanzamiento; DOT grande usa parciales en dispositivo. El umbral evita pagar un
  segundo lanzamiento para unos pocos miles de elementos.
- `compile` fusiona map-reduce OpenCL sin almacenar el array transformado completo.
  Las filas contiguas de 32 o más términos se reparten entre hilos; serializarlas
  por fila resultó una regresión durante esta investigación y se corrigió.
  Se preservan redondeos intermedios, dtype de acumulación y broadcasting.
- `x*x` ya cuenta como un único consumidor de x. Los productores compartidos por
  consumidores distintos o devueltos siguen teniendo una frontera conservadora.
- Los solvers GPU reutilizan el residual cuadrado que ya calcularon y comprobaron;
  finitud y residual usan planes compilados. CG elimina una observación redundante.
  Conserva residual real, tolerancias, historial y chequeos en **cada iteración**.
- Vulkan: fusión elementwise, GEMV/DOT y reducciones cooperativos, matmul con tiles
  de 8×8, pool de buffers, reutilización de command buffers y descriptor pools.
  Se cachean capacidades inmutables del dispositivo como valores Python para
  evitar leer repetidamente estructuras CFFI. Los recursos en vuelo no se reciclan.
- `matmul` admite broadcasting por lotes en los tres backends. `dot` de rango
  superior a 2 sigue fuera del contrato; no se confunden sus reglas con matmul.
- CLBlast usa GEMV/DOT nativos cuando están disponibles; GEMM permanece como
  alternativa del mismo proveedor para builds sin esos símbolos. Los lotes se
  resuelven mediante offsets y un bucle de GEMM, todavía sin BLAS batched nativo.

El backend CUDA y las matrices dispersas siguen como trabajo futuro. No se añade
un cambio automático a CPU/CUDA ni una elección de BLAS escondida al usuario.

## Método: qué mide cada evidencia

La batería principal conserva NumPy a 1/8/16 hilos BLAS, OpenCL custom/CLBlast y
Vulkan en eager/compile, CuPy, cuBLAS directo y nvmath-python. Se añaden controles
GEMV/DOT que exceden las cachés y tres formas de matmul por lotes. Cada modo tiene
siete rondas con orden aleatorio de implementaciones y validación de dtype, forma
y valores. Los solvers completos verifican además convergencia y residual host.

Los tiempos residentes incluyen despacho Python, asignación del resultado y
sincronización. Los tiempos de carga+cálculo+descarga usan entradas GPU nuevas;
NumPy conserva sus entradas host, sin una copia artificial. Los planes y programas
están calientes. La suite principal registra por separado primera ejecución,
subida, descarga y preparación. No mide arranque frío del proceso ni compilación
fría de todos los shaders.

Los perfiles adicionales usan una rueda congelada del estado anterior y el código
actual, en procesos separados. `gaps.py` conserva el mismo callable compilado entre
calentamiento, medición y perfil; no recaptura la función en cada muestra. Su CG
diagnóstico converge en dos iteraciones con un sistema bien condicionado; **no
tiene el mismo input que los solvers completos de `execution.py`**. CuPy ejecuta
la recurrencia de `benchmarks.suite.native_solver`, con sus chequeos de finitud,
curvatura y residual. No se presenta una recurrencia sin comprobaciones como
equivalente al solver público.

`cProfile` y contadores se ejecutan aparte de las muestras de pared. Sus tiempos
incluyen esperas; **no son porcentajes de utilización CPU**, y tiempos inclusivos
anidados no se suman. Los eventos OpenCL son otro control instrumentado: registran
timestamps reales de kernels/copias, excluyen comandos opacos de CLBlast y no
permiten etiquetar automáticamente `pared − kernel` como «Python».

Las rondas antes/después no son un experimento con frecuencias bloqueadas.
Los controles CuPy también varían entre procesos. Diferencias pequeñas y un único
microbenchmark no prueban una mejora causal; se conservan todas las muestras y
cuartiles. No se midieron consumo energético, memoria máxima ni rendimiento
sostenido durante horas. CuPy/cuBLAS/nvmath comparten la misma GPU NVIDIA; no son
evidencia de portabilidad entre fabricantes.

## Límites medidos y cuánto control tenemos

### Resultados cuantificados

**319 configuraciones completas sin errores de validación:** 172 operaciones/pasos,
63 solvers completos y 84 controles de tamaño/lotes. Las
tablas completas (`.artifacts/benchmarks/gaps-report/all-results.md`) y el
CSV con cuartiles (`.artifacts/benchmarks/gaps-report/all-results.csv`) son
artefactos locales no distribuidos; las siguientes tablas conservan lo esencial
en este documento.

Milisegundos residentes, float32. «Antes» es el snapshot de `execution-results.md`;
NumPy/CuPy son mediciones de la ronda final, no sus tiempos históricos.

| Caso | OpenCL antes → ahora | Vulkan antes → ahora | NumPy ahora | CuPy ahora |
| --- | ---: | ---: | ---: | ---: |
| Jacobi completo, 2048 | 25.121 → 12.732 | 350.699 → 82.502 | 10.201 | 11.712 |
| Descenso más pronunciado, 2048 | 34.135 → 19.631 | 538.382 → 100.486 | 12.972 | 22.426 |
| CG completo, 2048 | 17.428 → 9.764 | 311.270 → 49.800 | 7.819 | 10.572 |
| 100 actualizaciones, 1 048 576 elementos | 11.280 → 10.890 | 303.090 → 72.459 | 194.964 | 6.395 |
| Matmul 2048² | 16.402 → 16.416 | 20.447 → 11.905 | 29.083 | 1.478 |

**Los solvers todavía pierden contra NumPy en estos casos.** El progreso de Vulkan
es grande respecto a su implementación anterior, pero permanece lejos de OpenCL
y CuPy. En matmul 2048², CLBlast final da 3.864 ms, cuBLAS 1.471 ms y nvmath 1.578 ms:
hay un gap de proveedor/kernel que la fusión de un matmul aislado no resuelve.

Controles grandes finales; cada celda es **residente / carga+cálculo+descarga**:

| Caso | NumPy | OpenCL custom | CLBlast | Vulkan | CuPy |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMV 8192² float32 | 8.127 / 7.779 | 0.871 / 39.511 | 1.285 / 40.022 | 1.280 / 125.833 | 0.854 / 52.824 |
| GEMV 8192² float64 | 17.171 / 16.799 | 1.499 / 78.694 | 1.522 / 81.050 | 1.931 / 279.588 | 1.483 / 105.923 |
| DOT 16 777 216 float32 | 2.675 / 2.554 | 0.891 / 20.321 | 0.617 / 20.055 | 1.035 / 54.218 | 0.805 / 23.232 |
| Lote de 64 matrices 32² float32 | 0.101 / 0.094 | 0.210 / 0.401 | 2.013 / 2.308 | 0.515 / 2.063 | 0.195 / 0.328 |
| Lote de 16 matrices 128² float32 | 0.943 / 0.966 | 0.291 / 1.198 | 0.680 / 1.905 | 0.886 / 6.463 | 0.239 / 1.020 |

El GEMV grande custom está cerca de CuPy; subir 256 MiB cada vez destruye esa
ventaja. El coste Vulkan de transferencia es bastante mayor que OpenCL sobre el
mismo enlace: **ese exceso no puede justificarse solo por PCIe**. Los lotes pequeños
de CLBlast exponen claramente el coste del bucle de llamadas, aunque su GEMM grande
sea mucho mejor que el custom. El GEMV float64 es un recorrido de memoria y no
tiene la intensidad aritmética de GEMM: el límite FP64 no afecta igual a ambos.

### Kernel, ejecutor y sensibilidad de las medidas

En el control con eventos OpenCL, GEMV2048 pasó de 213 a 68 µs de kernel, mientras
la llamada instrumentada pasó de 290 a 192 µs. El CG diagnóstico de ese control
pasó de 2.284 a 0.570 ms acumulados en kernels, pero de 6.253 a 3.822 ms de pared.
La ganancia del kernel no se traduce proporcionalmente en ganancia del algoritmo.
El DOT corto conserva un caso mixto: su kernel pasó de 33.7 a 13.8 µs, mientras
la pared instrumentada pasó de 96.6 a 129.2 µs. No se oculta esa regresión.

En `gaps.py`, la energía compilada de 2048×32 tarda 90.0 µs en OpenCL; invocar un
plan ya preparado con bindings válidos tarda 66.1 µs. En Vulkan: 617.3 frente a
572.0 µs. Esto mide el margen de binding/búsqueda de variante, **no demuestra la
velocidad de un futuro ejecutor nativo**. En el CG Vulkan perfilado quedan 43
llamadas `_run`; la construcción `_new` de structs suma 6.7 ms inclusivos de una
invocación perfilada de 21.8 ms. Ese tiempo no se suma a sus llamadas hijas.

Hay una regresión aparente que sigue requiriendo cautela: la batería amplia da
1.188 ms para energía OpenCL32768×32, frente a 0.311 ms históricos. Un control
adicional, con el mismo cálculo, da 128–131 µs al medir por bloques de backend y
201 µs alternando backends; CuPy da 173–193 µs en ese control. A 2048×32, OpenCL
pasa de 101–103 a 162 µs al alternar. Las 21 muestras por modo y la vuelta al modo
por bloques están en `gaps-final/scheduling.json`.

**Alternar backends modifica el resultado y explica parte, pero no toda, la
diferencia con la batería amplia.** No se ha aislado cuánto corresponde a estado
del driver, frecuencias, contextos o transferencias precedentes. Se conservan los
tiempos amplios desfavorables y los controles aislados; no se reemplaza uno por el
mejor ni se declara solucionada esta sensibilidad. Para decisiones de tuning
pequeñas, hace falta replicación entre procesos y mayor control del orden/estado.

### Clasificación de los gaps

| Gap | Evidencia / interpretación | Control y acción |
| --- | --- | --- |
| PCIe y transferencias grandes | Enlace observado bajo carga: Gen3 ×8. Copias de 64 MiB: 6.79–7.09 GB/s según dirección y tipo de memoria host. | La capacidad física de este enlace no se cambia con kernels. Sí podemos evitar copias, mantener residencia, usar staging reutilizable y agrupar transferencias. |
| Memoria host pageable | A 4 MiB, H2D: 876 µs pageable frente a 631 µs desde memoria host asignada/mapeada por OpenCL. A 64 MiB la mejora es pequeña. | Hay margen de staging para tamaños intermedios. Este control excluye crear/mapear el buffer; no es una promesa de acelerar el API público en ese porcentaje. |
| Latencia de dispatch | Un kernel vacío OpenCL: 33.7 µs de pared, 10.7 µs de evento en este control. | Piso empírico de esa ruta y ese equipo, **no una constante física ni todo coste inevitable**. Incluye bindings/driver/cola. Fusionar y enviar grupos amortiza ese coste. |
| GEMV/DOT grandes residentes | Los controles exceden las cachés y comparan contra CuPy sobre la misma GPU. | Si ya estamos cerca de esa referencia, priorizar más tuning del kernel puede aportar menos que residencia y ejecución. Estar cerca de CuPy no demuestra alcanzar un techo físico. |
| Solvers residentes | Persisten diferencias aunque no se suban matrices por iteración. Hay reducción, GEMV, asignaciones, despacho y observación de escalares. | Controlable en buena parte. Manteniendo el control en Python hay que esperar sus decisiones. Mover control a un ejecutor/dispositivo es una extensión arquitectural; omitir chequeos cambiaría el contrato. |
| Vulkan Python/CFFI | La lectura reiterada de capacidades y construcción de structs aparece en cProfile. Cachear límites redujo el primer componente. | Coste de nuestra implementación. Siguen creación de structs/descriptores, grabación y un submit por operación; faltan batching y una ruta de envío más compacta. No es un límite de Vulkan. |
| Matmul por lotes CLBlast | El adaptador emite una llamada GEMM por matriz; lotes de matrices pequeñas pagan muchas llamadas. | Integración controlable: usar GEMM batched/strided batched y medir frente al custom. La frontera BLAS no debe impedir epílogos o kernels propios. |
| FP64 en esta GPU | Compute capability 12.0; throughput escalar FP32:FP64 de 64:1 según NVIDIA. | Restricción del hardware elegido, no del lenguaje Python ni de OpenCL. Mejorar un kernel no convierte esta GPU en una GPU de alto throughput FP64. No reducir precisión silenciosamente. |
| Rango numérico de normas | La suma de cuadrados puede subdesbordar o desbordar; existe reproducción de convergencia falsa con valores diminutos. | Gap numérico controlable, separado del rendimiento. Requiere normas y coeficientes robustos, no un epsilon arbitrario ni cambiar tolerancias. |
| AMD/Intel y otros sistemas | Solo se ejecutó en la GPU NVIDIA local. | Falta de evidencia, no limitación demostrada. Los kernels y APIs portátiles deben validarse con hardware real; PoCL o un driver software no sustituyen esa prueba. |

Para PCIe Gen3 ×8, la cuenta de codificación 128/130 da unos 7.88 GB/s por dirección
antes de overhead de protocolo; no es ancho de banda útil garantizado. El control
de copia ya está cerca de esa escala. La codificación y tasa por carril están
documentadas por [Intel](https://www.intel.com/content/www/us/en/docs/programmable/683724/18-0/physical-layer.html).
D2D cuenta bytes de payload una sola vez;
no se compara ese cociente directamente con una cifra comercial que cuente lectura
y escritura. Una matriz float32 de 2048² ocupa 16 MiB; esta CPU tiene 100 MB de
caché total publicados por [AMD](https://www.amd.com/en/newsroom/press-releases/2024-1-8-amd-reveals-next-gen-desktop-processors-for-extrem.html).
El runtime CUDA reporta 32 MiB de L2 en esta GPU. La matriz de 8192² ocupa 256 MiB:
los controles grandes evitan concluir
«ancho de banda de DRAM» a partir de problemas pequeños cacheados.

El cociente FP64 es para aritmética escalar, no Tensor Cores. No se activó TF32 ni
se sustituyó float64 por float32. [Capacidades de cómputo de NVIDIA](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
Khronos documenta el coste de asignación y de crear/liberar command buffers, pero
la atribución a nuestro runtime procede del perfil local, no solo de esas guías.
[Asignación Vulkan](https://docs.vulkan.org/guide/latest/memory_allocation.html),
[reutilización de command buffers](https://docs.vulkan.org/samples/latest/samples/performance/command_buffer_usage/README.html).

## Corrección: un fallo del proveedor y una deuda numérica

CLBlast 1.7.0 GEMV produjo NaN con inputs finitos cuando el destino reciclado
contenía NaN/Inf, aunque `beta=0`. Se aisló desactivando solo GEMV nativo: GEMM
resolvía el mismo caso; desactivar DOT no lo resolvía. El adaptador ahora llena el
destino con cero mediante una operación GPU en la misma cola. No hay subida ni
espera host, pero sí un comando adicional incluido en los tiempos finales.
Las regresiones comprueban GEMV normal/transpuesto, float32/64 y CG2048 repetido
con el pool contaminado. No se atribuye este comportamiento a todos los proveedores
BLAS ni a todas las versiones de CLBlast.

La deuda numérica preexistente se reproduce con `A=eye(2)`,
`b=[1e-30,-1e-30]` float32, `rtol=atol=0`: el cuadrado del residual subdesborda.
La comprobación en float64 da una norma de aproximadamente 1.414e-30 mientras
la suma de cuadrados float32 da cero. Una solución completa debe cubrir también
los coeficientes de CG/SD; cambiar solo la norma puede desplazar el fallo al DOT.
Se documenta explícitamente en [compatibilidad](compatibility.md).
Se reprodujo en los tres backends locales: todos declararon convergencia en cero
iteraciones con residual reportado cero; la comprobación float64 dio 1.414e-30.

## Decisión arquitectural y siguiente inversión

Se mantiene **eager con arrays residentes + compile opcional**, con IR común y
lowering específico por backend. Las asignaciones Python no son fronteras de
fusión dentro de la función compilada. No hace falta convertir el proyecto en un
trazador lazy global ni exigir 100 % de NumPy para aplicar estas optimizaciones.

1. **Ejecución y transferencias Vulkan.** Menos estructuras Python por dispatch,
   reutilizar descriptor sets de forma segura, grabar/enviar grupos de operaciones
   y mejorar el staging. Un módulo nativo pequeño puede servir al ejecutor sin
   cambiar el API de arrays. Mantener gestión de recursos en vuelo y locks.
2. **Solvers y fronteras útiles.** Fusionar GEMV con actualizaciones/reducciones
   donde el proveedor lo permita, map-reduce Vulkan y preparación reutilizable de
   sistemas inmutables. Esta última necesita un objeto/contrato explícito: no se
   deben eliminar validaciones de entradas basándose en una suposición escondida.
3. **Ejecutor de planes preparado.** El control que llama directamente al plan
   mide cuánto ahorra omitir binding/especialización; no es todavía código nativo
   ni API pública. Si ese ahorro es pequeño frente a `_run`, priorizar el envío
   del backend antes de reescribir el compilador completo.
4. **BLAS por contrato, con kernels custom.** Integrar batched nativo; optimizar
   GEMM custom OpenCL (aún sin tiling) y mejorar tiles Vulkan si el workload lo
   justifica. Mantener dtypes, broadcasting y errores al intercambiar proveedor.
5. **Robustez y evidencia externa.** Norma/coeficientes escalados con pruebas de
   rango; ejecutar con AMD e Intel, incluido un dispositivo sin fp64. La ausencia
   de ese hardware queda pendiente y no se transforma en una promesa validada.

La reducción de host overhead debe conservar comprobaciones en cada iteración.
Una API futura con menor frecuencia tendría que ser explícita y medirse por
separado. Los datos pequeños se siguen soportando; ganar ahí no dirige el diseño.

## Artefactos y reproducción

Los artefactos se guardan en `.artifacts/benchmarks`, fuera del código versionado:
`gaps-before` (rueda congelada), `gaps-after` (etapa intermedia), `gaps-final`,
`gaps-execution-final`, `gaps-solvers-final` y `gaps-report`. La etapa intermedia
permite ver las regresiones corregidas; no representa el runtime final.

```powershell
$env:PYOPENCL_NO_CACHE='1'
$env:PACHO_VULKAN_COMPILER='<ruta a glslang o glslc>'
$env:OPENPACHO_CLBLAST_LIBRARY='<ruta a CLBlast>'
$env:CUPY_CACHE_DIR='<directorio privado escribible>'
python -m benchmarks.gaps --output .artifacts/benchmarks/gaps-final/profile.json
python -m benchmarks.opencl_events --output .artifacts/benchmarks/gaps-final/events.json
python -m benchmarks.transport_floor --output .artifacts/benchmarks/gaps-after/transport.json
python -m benchmarks.scale --output .artifacts/benchmarks/gaps-final/scale.json
python -m benchmarks.execution --output .artifacts/benchmarks/gaps-execution-final/raw.json
python -m benchmarks.execution --solvers-only --output .artifacts/benchmarks/gaps-solvers-final/raw.json
python -m benchmarks.gaps_tables
python -m benchmarks.scheduling_control
```

`transport_floor` y `opencl_events` son controles OpenCL separados; el primero
recoge también `nvidia-smi` en este equipo. No ejecutar otra carga GPU durante los
benchmarks. La comparación histórica usa el wheel congelado mediante `PYTHONPATH`,
cuya ruta se registra en los perfiles. El informe de verificación conserva el
[alcance de pruebas y empaquetado](verification.md).
