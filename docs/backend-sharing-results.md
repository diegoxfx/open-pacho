# Vulkan, OpenCL y CUDA: ejecución compartida — 2026-09-13

## Implementación

- **Vulkan:** detecta/habilita `VK_KHR_push_descriptor` y properties2. Dentro de
  maxPushDescriptors graba un snapshot de bindings en el command buffer, con una
  llamada en lugar de actualizar y enlazar un set. Fuera del límite o sin la
  capacidad conserva descriptor sets. Reutiliza registros host bajo el lock y
  retiene buffers GPU en vuelo. Shaders, barreras, número de dispatches/envíos y
  controles de convergencia se conservan.
- **OpenCL:** cachés LRU de metadatos para elementwise/gather y cálculos de
  strides/offsets; nunca guardan arrays o escalares enlazados. Los índices usan
  int32 cuando una cota conservadora permite todos los accesos intermedios;
  conservan int64 para offsets/formas mayores. El dtype almacenado no cambia.
  Los layouts públicos siguen materializados; esto no implementa transpuestas
  lógicas ni flags de BLAS todavía.
- **CUDA:** matmul vectorial/matricial y por lotes broadcast float32/64, en eager
  y compile. cuBLAS usa strided-batched en lotes regulares con strides positivos;
  broadcasting general usa GEMM por lote, con punteros reutilizados y salidas
  independientes. Custom ejecuta todos los lotes en un lanzamiento. GEMM float32
  elegible comparte el generador de tiles/registros de OpenCL/Vulkan; las demás
  formas y float64 conservan el kernel directo. Siguen pendientes normas estables,
  fusión CUDA y primitivas vectoriales cuBLAS específicas.

Las optimizaciones se seleccionan internamente. No se añadieron extras, macros
públicas, cambios en los ejemplos ni un ejecutor C/C++ nuevo. Los proveedores
BLAS siguen siendo explícitos y los errores no cambian silenciosamente de backend.

## Aprendizajes entre APIs y capacidades

Los principios se pueden compartir: reutilizar metadatos sin retener valores,
mantener buffers residentes, accesos coalescentes, memoria local y bloques de
registros. El GEMM float32 ahora tiene un generador común para las tres APIs.
Compartir fuente/algoritmo no garantiza el mismo código máquina ni el mismo
tiempo: compilador, dispatch, cola, barreras y esperas siguen siendo diferentes.

En el equipo local, Vulkan anuncia 32 push descriptors; OpenCL anuncia API 3.0
pero OpenCL C 1.2 y no anuncia `cl_khr_subgroups`. El número de versión de la API
por sí solo no permite seleccionar instrucciones de subgrupo. Se usa evidencia
de capacidad, no el nombre comercial de la GPU.

Se probó `cl_nv_pragma_unroll` en el bucle interno del GEMM OpenCL, en cuatro formas
float32 y 21 muestras intercaladas. Los resultados fueron idénticos, pero los
tiempos no mostraron una ganancia consistente: se conserva como experimento y
no se habilita en el runtime. En 1025×513×769 los eventos GPU medianos fueron
aproximadamente 0,329 ms en ambas variantes; en 1024³, 0,770/0,772 ms.
Es un diagnóstico con cola perfilada y espera del evento, no una comparación
directa contra la pared de aplicaciones con queue.finish ni un techo físico.

Las macros son útiles para expresar una variante de kernel; no sustituyen
consultar/habilitar capacidades o elegir llamadas host. Una instrucción FMA puede
mejorar error y velocidad y aun así cambiar redondeos; instrucciones matriciales
especiales también pueden cambiar acumulación o precisión. No se activan como
si fueran equivalentes sin establecer el contrato correspondiente. El track de
[hardware](hardware-specialization.md), modos matemáticos y autotuning continúa.

Fuentes primarias: [push descriptors](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_push_descriptor.html),
[contrato de cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched),
[hint de desenrollado](https://registry.khronos.org/OpenCL/extensions/nv/cl_nv_pragma_unroll.txt).

## Metodología y reproducción

Hardware local: RTX 5060 Ti, Windows 11, driver 616.64. La evidencia no certifica
otros fabricantes, otros drivers ni Linux. El pipeline de kernels CUDA conserva
fmad=false y ftz=false; cuBLAS conserva PEDANTIC_MATH. Las comparaciones externas
registran NVIDIA_TF32_OVERRIDE=0 y las versiones realmente importadas.

`benchmarks.vulkan_dispatch --control descriptors` intercala sets/push, con la
misma frontera CFFI, shaders y datos. Instrumenta en una invocación separada;
los tiempos de API incluyen driver/esperas y los intervalos GPU incluyen efectos
de timestamps/etapas. No se resta un reloj del otro para denominarlo tiempo Python.

`benchmarks.backend_sharing` compara operaciones rectangulares/impares, incluyendo
lotes. Publica primera llamada, pared residente, duración de llamada host y pared
completa (subida/cálculo/descarga). Las modalidades tienen muestras independientes;
sus medianas no se suman. La duración host puede incluir esperas internas. El
destructor de la salida pública queda fuera del reloj; los temporales internos
forman parte de la ejecución. No se mide pico de VRAM.

El control `--control-wheel` carga las funciones de un wheel histórico explícito
y alterna esos métodos con los actuales sobre el **mismo backend/contexto e
inputs**. OpenCL controla elementwise/gather; CUDA controla matmul. La asignación
de métodos se hace antes del reloj. Los demás métodos del runtime son los actuales.
Esto no equivale a ejecutar un wheel histórico completo. El JSON registra el
hash del wheel, el alcance y los hashes de las fuentes actuales. La comparación
amplia en procesos separados sí registra el paquete completo realmente importado;
se usa como contexto ante variaciones de tiempo en operaciones sin cambios.

```powershell
python -m benchmarks.vulkan_dispatch --control descriptors --repeats 21 --output .artifacts/push-new.json
python -m benchmarks.backend_sharing --backends opencl --control-wheel <wheel-anterior> --repeats 21 --output .artifacts/opencl-control-new.json
python -m benchmarks.backend_sharing --backends cuda-custom --control-wheel <wheel-anterior> --repeats 21 --output .artifacts/cuda-control-new.json
python -m benchmarks.backend_sharing --repeats 21 --output .artifacts/operations-new.json
```

Usar archivos nuevos, workloads GPU secuenciales y repetir en otro proceso.
Los seeds del control alteran el orden, no los inputs. Todas las comparaciones
publican error numérico. Para float64, la referencia NumPy de estas operaciones
tiene la misma precisión; la suite adversa adicional usa Decimal80.

## Controles de las mejoras

Dos procesos por control, 21 muestras intercaladas por modalidad. Las tablas
muestran medianas de pared residente en ms, no tiempo puro de kernel. Los
porcentajes son reducción de latencia; el factor es tiempo anterior/actual.

| OpenCL, float32 | Anterior → actual, ronda 1 | Anterior → actual, ronda 2 |
| --- | ---: | ---: |
| Cadena elementwise, 33 elementos | 0,449 → 0,345 | 0,445 → 0,292 |
| Cadena, 1.048.577 elementos | 0,557 → 0,328 | 0,496 → 0,374 |
| Transposición y copia, 4097×129 | 0,235 → 0,161 | 0,172 → 0,108 |
| Reversión y copia, 4097×129 | 0,148 → 0,099 | 0,175 → 0,112 |

En los ocho casos afectados, contando ambas precisiones y las dos rondas,
la reducción residente es del **23–41 %**. El control mide conjuntamente caché
de metadatos e índices estrechos; no atribuye todo el ahorro al shader ni separa
el aporte de cada cambio. GEMM OpenCL no cambió en esta ronda: sus variaciones,
incluso el lote pequeño con una diferencia grande, sirven como control negativo
del ruido, no como evidencia de una mejora o regresión del kernel.

| CUDA custom | Anterior → actual, ronda 1 | Anterior → actual, ronda 2 |
| --- | ---: | ---: |
| GEMM float32, 257×129×513 | 0,153 → 0,145 | 0,150 → 0,139 |
| GEMM float32, 1025×513×769 | 1,028 → 0,361 | 0,999 → 0,325 |
| GEMV float64, 2049×1023 | 0,218 → 0,220 | 0,214 → 0,214 |
| GEMM float64, 1025×513×769 | 4,750 → 4,750 | 4,746 → 4,746 |

El tile compartido acelera el GEMM grande float32 **2,8–3,1×**. No mejora float64,
que conserva su kernel directo. La primera versión de lotes añadió trabajo host
y aritmética de índices a llamadas 2D; el control lo detectó y se restauró una
ruta directa para vectores/matrices sin lotes. Los controles finales de cuBLAS
ya no muestran la penalización sistemática observada en los casos pequeños.
Los lotes CUDA no tienen tiempo anterior comparable: antes eran operaciones
no implementadas. Todas las parejas ejecutables conservan las métricas de error
registradas; eso no constituye una garantía bit a bit para cualquier entrada.

| Vulkan, float32, sets → push | Ronda 1 | Ronda 2 |
| --- | ---: | ---: |
| DOT, 2049 | 0,238 → 0,234 | 0,238 → 0,233 |
| Ridge, 4097×129, 20 pasos | 5,552 → 5,310 | 5,288 → 5,070 |
| Descenso más pronunciado, 513², misma convergencia | 23,403 → 22,341 | 23,491 → 22,255 |
| Suma, 1.048.577 elementos | 0,311 → 0,377 | 0,321 → 0,351 |

**Vulkan mejora de forma modesta y mixta.** Hay una llamada host menos por
dispatch, pero no menos dispatches, envíos, copias ni comprobaciones. Las rutas
de transferencia no usan descriptores; sus diferencias tampoco son ganancias
causadas por push. CG y varios casos float64 siguen sin una mejora consistente.

La regresión de suma grande exigió un diagnóstico adicional: 51 muestras
intercaladas sobre **el mismo VkDevice, cola y buffers**. Solo se alternan
pipelines/layouts compatibles con cada modalidad; un comentario de shader
separa las claves de caché y no cambia sus operaciones. Allí sets/push dan
0,221/0,220 ms para la suma grande float32; ridge 5,079/4,958 ms y descenso
22,954/22,601 ms. Las salidas son idénticas. La regresión no se reproduce en este
control; aún no está aislada su causa en el control con dos dispositivos lógicos.
Se conservan ambos resultados y no se introduce una excepción por tamaño o GPU
basada solo en esa correlación. La capacidad habilita una ruta válida, no promete
una aceleración universal. Un A/B de despliegue en otro driver/hardware sigue
siendo necesario.

## Precisión y validación

La suite aplicada final ejecuta **104 configuraciones** de covarianza y ridge
en matrices 4095×127 y 4097×129, float32/64, con siete muestras. Incluye NumPy a
1/8 hilos, OpenCL custom/CLBlast, Vulkan, CUDA custom/cuBLAS y CuPy. No hay
excepciones ni proveedores ausentes. El primer intento tuvo 96 ejecuciones y
CuPy no disponible porque su caché predeterminada estaba fuera del workspace;
se repitió con `CUPY_CACHE_DIR` apuntando a `.artifacts/cupy-cache`.

El error relativo L2 máximo en esas aplicaciones es **7,95e-7 en float32** y
**1,51e-15 en float64**, frente a NumPy float64. Para float64 es igual precisión.
La suite adversa Decimal80 ejecuta **158 configuraciones**, sin excepciones, y
conserva las **40 alertas anteriores**: nueve por cada proveedor OpenCL, diez
Vulkan y seis por cada proveedor CUDA. Coinciden exactamente los casos alertados
con la ronda CUDA inicial. Cancelación, subnormales Vulkan float32 y sistemas
mal condicionados continúan como problemas conocidos. Las 52 combinaciones
CUDA de normas/solvers no soportadas se registran aparte. El código de salida 1
del diagnóstico corresponde a las alertas; no es una verificación toda correcta.

Las recurrencias de `examples/solvers.py` conservan su AST respecto al sdist
anterior. En `benchmarks/applications.py` cambia solo la descripción del alcance
CUDA en los metadatos. No se modificaron cálculos, tolerancias ni frecuencia de
comprobación para beneficiar a un backend.

La suite completa previa al último ajuste CUDA pasó **1.084 pruebas, una omitida**
(necesita dos dispositivos OpenCL). Después de añadir cobertura de lotes y corregir
la ruta sin lotes, las **75 pruebas CUDA** pasan. Incluyen float32/64, formas
vacías/impares/rectangulares, broadcast, salidas recicladas con NaN/Inf, retención
de recursos y cuBLAS strided-batched. Las pruebas Vulkan cubren push y sets,
140 dispatches cruzando el límite de drenaje y el fallback por límite de bindings.
OpenCL comprueba que las cachés reenlazan escalares/buffers y que estrechar índices
no estrecha los datos, incluyendo gathers con strides negativos.

El wheel construido desde sdist pasa **326 pruebas, con 719 omisiones** en el
entorno CPU sin runtimes GPU. Se verifica importación desde el wheel, extras
independientes y la separación de ejemplos/solvers. NumPy sigue como única
dependencia base. Ruff/formato pasan; se analiza sintaxis Python 3.10 y se
ejecuta Python 3.12. Esta versión aún no se ha publicado.

## Gaps entre proveedores

Se ejecutaron también **144 configuraciones con un único proveedor por proceso**:
ocho procesos secuenciales, mismos datos y 21 muestras. Sirven para comparar el
uso habitual de un backend. No eliminan toda deriva de frecuencia/carga del SO.
La comparación adicional que intercala todas las APIs en un proceso también
valida 144 configuraciones, pero perturba fuertemente algunos tiempos: el GEMM
grande OpenCL custom pasa de 0,401 ms aislado a 3,312 ms intercalado, y CUDA custom
de 0,322 a 3,021 ms. El mecanismo exacto entre contextos, driver y planificación
no está aislado. No se usan esos números intercalados como techo de cada API ni
como medida del coste de nuestra abstracción.

GEMM 1025×513×769, ms; cada celda contiene **residente / completo**:

| Proveedor aislado | float32 | float64 |
| --- | ---: | ---: |
| NumPy, 1 hilo | 7,271 / 7,306 | 14,816 / 14,964 |
| NumPy, 8 hilos | 2,067 / 1,973 | 4,058 / 4,062 |
| OpenPacho OpenCL custom | 0,401 / 1,822 | 2,196 / 4,639 |
| OpenPacho OpenCL CLBlast | 0,336 / 1,808 | 2,168 / 4,573 |
| OpenPacho Vulkan | 0,338 / 2,121 | 2,267 / 5,500 |
| OpenPacho CUDA custom | 0,322 / 1,929 | 4,741 / 7,228 |
| OpenPacho CUDA cuBLAS | 0,230 / 1,673 | 2,729 / 5,192 |
| CuPy | 0,165 / 1,487 | 2,658 / 5,222 |

El GEMM float32 común es competitivo entre nuestras APIs: Vulkan supera OpenCL
en este caso, mientras OpenCL gana en el GEMM pequeño y los lotes pequeños. CUDA
custom todavía tarda aproximadamente 1,95× lo de CuPy en el GEMM grande, y
OpenPacho/cuBLAS 1,39×. Esta última diferencia incluye selección de llamadas,
binding, contexto, pool y ejecución; no es una medición exclusiva de Python.
Los errores relativos L2 float32 del GEMM grande son 4,06–4,08e-7 en los custom
y 2,89e-7 en las rutas CUDA/cuBLAS y CuPy. Es evidencia de precisión registrada,
no una declaración de equivalencia numérica entre proveedores.

Los lotes (7,33,17)@(7,17,65) siguen favoreciendo claramente a CPU: alrededor
de 0,020 ms residente, frente a 0,089 OpenCL, 0,126 CUDA custom y 0,200 Vulkan.
El broadcasting general CUDA custom usa un lanzamiento y da 0,138 ms, mientras
cuBLAS usa seis GEMM y da 0,371 ms en (2,1,33,17)@(1,3,17,65). No se fuerza un
proveedor distinto al seleccionado; una futura ruta pointer-batched tiene margen.

Los datos residentes importan: la cadena grande OpenCL da 0,332 ms, pero
subida/cálculo/descarga da 2,720 ms, frente a 1,684/1,805 ms NumPy de un hilo.
CuPy también pierde en ejecución completa de esa cadena: 2,345 ms. El ahorro de
transferencia y residencia sigue siendo específico de la carga, no una promesa
de que GPU siempre gane a NumPy.

## Próximas optimizaciones justificadas

1. **Vulkan:** staging/mapeos, envío y esperas, especialmente lecturas pequeñas;
   push reduce trabajo de descriptores pero no resuelve estas fronteras. Medir
   event/fence y agrupación/reutilización del ejecutor antes de añadir C/C++.
2. **OpenCL:** separar generación/binding, ejecución y espera de cola. El pragma
   ensayado no cerró el gap de GEMM; medir tiles/packing según formas, con CLBlast
   como proveedor de referencia y controles aislados del contexto.
3. **CUDA:** GEMV/DOT cuBLAS dedicados, fusión elementwise y tiles float64 cuando
   los casos lo justifiquen. Evitar construir una lista Python por lote broadcast
   mediante una ruta batched por punteros, respetando la vida útil de esos buffers.
4. **Compartido:** layouts/transpuestas lógicas y política explícita de precisión.
   Capacidades habilitan variantes; autotuning, perfiles de hardware y modos
   rápidos precisan mediciones y contratos propios. La abstracción y el hardware
   no justifican atribuir los gaps actuales a un límite físico que no hemos medido.

Artefactos locales en `.artifacts/backend-sharing`: `control-opencl-{1,2}.json`,
`final-control-cuda-{custom,cublas}-{1,2}.json`, `push-{1,2}.json`,
`push-same-device.json`, `isolated-*.json`, `operations-final.json`,
`applications-final.json`, `accuracy.json` y `comparison.csv`. Los archivos
before/after, quick y los primeros controles CUDA son evidencia exploratoria,
no sustituyen los controles finales. El manifiesto distingue cambios de fuentes
posteriores en archivos ajenos al backend de cada control.
