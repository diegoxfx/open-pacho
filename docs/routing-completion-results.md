# Routing por implementación y finalización Vulkan — 2026-09-13

Se adopta dispatch mediante métodos del backend y factorías resueltas al preparar
planes. Se elimina el retorno del RNG a un selector central por nombre, se cachean
sus plantillas completas y Vulkan pasa a esperar una fence reutilizable. También
se corrige un fallo real de reentrada de finalizadores en su pool.

## Decisión arquitectónica

La preocupación por `if backend == ...` es válida cuando cada operación común
necesita conocer todas las implementaciones. En RNG existía un recorrido redundante:
fachada → método del backend → helper común que volvía a preguntar por el backend.
El coste relevante no era comparar tres strings: OpenCL repetía sustituciones de
casts y otras APIs reconstruían envoltorios de fuente en cada llamada.

Ahora la propiedad está definida así:

```mermaid
flowchart LR
    F[API random: validación y contadores] --> B[backend.random_data]
    B --> A[Adaptador propio: fuente y lanzamiento]
    A --> S[Philox compartido: aritmética y escalares]
    A --> G[Runtime GPU opcional]
    C[compile: preparación] --> L[backend.create_lowering]
    L --> P[Plan: lowerer y execution_scope enlazados]
```

- `random.py` conserva parámetros, rangos, capacidades por distribución/dtype,
  reservas de contadores y exclusión de captura. Cada backend importa su adaptador
  directamente. `_random.py` recibe fragmentos de dialecto como datos inmutables;
  no conoce nombres, colas ni clases GPU. NumPy usa directamente su referencia.
- Las LRU de fuente tienen límite 128 y claves inmutables. OpenCL/Vulkan reutilizan
  el cuerpo por dtype/distribución; CUDA incluye count porque lo incrusta en su
  fuente. Semilla, contador, límites, escala y salida se reenlazan siempre. No se
  retienen arrays ni se cambia el stream Philox.
- El plan consulta una vez la factoría opcional `create_lowering()` y enlaza
  `execution_scope()`. OpenCL/Vulkan proporcionan lowerer; CUDA/NumPy siguen con
  primitivas. Una factoría que falla propaga el error; su ausencia no cambia de
  dispositivo. El ámbito es un contexto nuevo por ejecución, con cleanup ante errores.
- Selección pública por nombre, validaciones de dtype/forma y decisiones por
  capacidades siguen siendo necesarias. Esta iteración no convierte clases
  arbitrarias en plugins públicos ni elimina todas las condiciones del proyecto.
  `_gemm.py`, `_gemv.py`, `_norm.py` y parte del tratamiento de escalares conservan
  deuda de routing. Se migrarán al ampliar esas operaciones, con controles propios.

| Alternativa | Decisión |
| --- | --- |
| Selector central en cada operación | Evitar cuando duplica dispatch y obliga a editar código común por cada backend |
| Métodos y factorías del backend | Adoptar: llamadas directas, propiedad clara, preparación fuera del bucle |
| Registro global, singledispatch o plugins automáticos | No introducir: el backend ya identifica la implementación; añadirían otra resolución |
| IR completo de lenguajes de kernels | Posponer: excede el problema; conservar aritmética compartida y adaptadores pequeños |

No cambia la API NumPy, eager/compile/fuse, los extras independientes ni la elección
explícita de hardware/proveedor. No se agregan trucos a solvers o comparadores.

## Vulkan: sincronización y pool

`synchronize()` envía primero los comandos grabados. Si no hay pendientes, retorna
sin espera nativa. Si hay trabajo, envía un marcador vacío con una fence del contexto,
espera su señal, la resetea y solo entonces recicla comandos/descriptores/leases.
No se crea una fence por operación ni se agrupan menos controles de convergencia.
La ruta privada `queue` conserva `vkQueueWaitIdle` para comparaciones.

El alcance del marcador incluye comandos anteriores de la cola propia, según
la [especificación de señalización de fences](https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-fences-signaling).
Esperar una fence no sustituye la visibilidad de memoria al host: se conservan
barreras de dependencias y readback. Los resultados de
[`vkWaitForFences`](https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitForFences.html)
se comprueban también por la tabla CFFI. Ante fallo de submit/wait/reset, la
instancia retiene referencias y rechaza nuevos envíos; no reintenta ni recicla
recursos inciertos. Su cierre también puede propagar el fallo: recuperar un
dispositivo perdido sigue siendo trabajo futuro.

El primer control prolongado falló con `OrderedDict mutated during iteration`.
Un finalizador puede devolver o expulsar buffers mientras una asignación Python
dispara GC, incluso teniendo el RLock en el mismo hilo. La búsqueda usa ahora una
instantánea de claves y revalida la capacidad escogida. La liberación separa el pool
actual antes de recorrerlo. Las regresiones CPU simulan tanto devolución como
expulsión durante el recorrido y comprueban contabilidad y ausencia de doble liberación.
Es un bug de gestión Python, sin evidencia de dependencia de fabricante.

## Método de medición

Windows / RTX 5060 Ti / NVIDIA 616.64 / Python 3.12. Procesos secuenciales por
backend, reposo observado al menos 10 s y techo 52 °C antes de cada proceso,
warmup mínimo 0,25 s por modo y 31 muestras. Se guardan muestras, p10/p90,
telemetría, fuentes y referencia histórica. El reposo no garantiza régimen térmico
sostenido ni elimina la variación de CPU/driver/WDDM.

El control Vulkan compara tres rutas en el mismo dispositivo/contexto: método
`synchronize` del wheel anterior, espera de cola omitiendo vacíos y fence. Los
kernels, el pool corregido y la planificación son comunes. Dos rondas con semillas
941/942 más una confirmación con 117: **144 configuraciones**, con tiempos
residentes y, cuando procede, completos. Las mediciones instrumentadas son separadas:
no se interpreta pared menos timestamps GPU como «tiempo puro de Python».

El control RNG cambia solo el método `random_data` en el mismo backend y conserva
los kernels/sincronización. Usa la ruta Vulkan de cola en ambos brazos, previa a
promover la fence; sus ganancias no deben sumarse a las de finalización. Dos rondas
OpenCL/Vulkan/CUDA custom, float32/64 y tamaños 33, 257×129, 1025×1023:
**264 configuraciones válidas y 12 exclusiones por capacidad**. Cada par verifica
igualdad exacta de outputs, además de error frente a Philox/NumPy y momentos.

## Resultados Vulkan

Microsegundos residentes; cada celda muestra las tres medianas por orden de ronda.
El archivo CSV adjunto al artefacto conserva todos los casos, tiempos completos y
dispersión. Los controles incluyen asignación; las descargas entregan datos al host.

| Caso | Anterior | Cola sin vacíos | Fence adoptada |
| --- | ---: | ---: | ---: |
| Suma 33, float32 | 197 / 183 / 199 | 197 / 198 / 199 | 182 / 165 / 176 |
| Descargar 1, float32 | 151 / 146 / 156 | 135 / 130 / 133 | 125 / 107 / 126 |
| Descargar 1.048.576, float32 | 1.379 / 1.357 / 1.390 | 1.346 / 1.334 / 1.358 | 1.342 / 1.320 / 1.333 |
| Energía 2049×33, float32 | 235 / 203 / 287 | 229 / 201 / 213 | 201 / 177 / 190 |
| Covarianza 1025×65, float32 | 436 / 403 / 421 | 414 / 398 / 430 | 399 / 393 / 408 |
| CG 129, float32 | 11.866 / 10.723 / 11.084 | 12.505 / 10.569 / 11.977 | 11.093 / 10.334 / 10.521 |
| Descenso 129, float32 | 22.167 / 19.795 / 20.245 | 21.685 / 20.495 / 22.404 | 19.537 / 18.235 / 20.267 |
| CG 129, float64 | 18.512 / 19.566 / 19.621 | 20.065 / 19.736 / 19.766 | 17.964 / 20.398 / 18.938 |
| Descenso 129, float64 | 44.330 / 39.351 / 39.754 | 46.763 / 40.629 / 42.453 | 37.058 / 37.135 / 42.352 |

La suma corta mejora aproximadamente 7–11 % frente a la espera anterior, sin
cambiar su kernel. Una llamada explícita sobre cola vacía, más la sincronización
del medidor, baja de 20–26 µs a 0,7–1,4 µs: elimina trabajo redundante, no acelera
cálculo GPU. Las copias grandes cambian poco. CG float64 empeora en una ronda y
descenso float64 en la confirmación frente al método histórico. También hay
regresiones completas, por ejemplo suma float64 444→553 µs en la primera ronda
y covarianza float64 741→794 µs en la segunda. Se conservan en el CSV.

Se adopta la fence por mejoras repetidas en las operaciones cortas y resultados
mayoritariamente favorables frente a la alternativa de cola sin vacíos. No se
afirma mejora universal ni haber alcanzado un límite físico. La diferencia entre
las dos llamadas nativas puede depender de implementación del driver; estas cifras
no separan ese efecto de scheduling del host ni prueban otros fabricantes.

## Resultados de routing RNG

El control CPU sustituye asignación y lanzamiento por stubs, para aislar preparación
del adaptador; 11 muestras de 2.000 llamadas float32. No mide latencia pública ni GPU.

| Backend | Preparación anterior, µs | Adaptador cacheado, µs |
| --- | ---: | ---: |
| OpenCL, rango de cuatro distribuciones | 15,5–20,1 | 3,1–3,8 |
| Vulkan | 13,1–14,0 | 10,4–11,2 |
| CUDA | 3,9–4,6 | 3,1–3,9 |

Los barridos públicos tienen resultados mixtos. Un caso repetido fue random(33)
OpenCL float32: 194→244 µs y 194→233 µs. Se investigó con **21 bloques alternados
de 11 muestras por modo**, aleatorizando su orden, en procesos adicionales con
reposo observado. El mismo caso dio **221→213 µs**; float64 **203→179 µs**.
CUDA dio **134→128 µs** y **103→103 µs**. Los resultados originales se conservan:
la regresión inicial no se reprodujo bajo esa alternancia, pero no se demuestra
que toda diferencia sea ruido ni que desaparezcan todas las regresiones.

La decisión se apoya en propiedad de implementación más clara, preparación más
barata y equivalencia numérica. No se presenta como una victoria general de
rendimiento GPU. En CUDA la reducción de host es menor que 1 µs en este control;
el despacho y trabajo nativo siguen dominando.

## Error y semántica

Los pares RNG producen exactamente la misma secuencia antes/después. Frente a
la transformación NumPy de la misma precisión, el máximo error absoluto fue
2,75e-6 y el relativo L2 3,33e-7; no es una referencia de precisión superior para
float32 comparado con float32. Uniformes, no finitos y momentos se validan por caso.

En solvers no cambiaron tolerancias, frecuencia de chequeo, algoritmo ni iteraciones:

| Aplicación | Iteraciones | Norma de residuo real / límite | Error relativo de solución |
| --- | ---: | ---: | ---: |
| CG float32 | 13 | 6,331e-4 / 8,970e-4 | 1,746e-4 |
| Descenso float32 | 29 | 8,554e-4 / 8,970e-4 | 3,780e-4 |
| CG float64 | 23 | 5,025e-7 / 8,970e-7 | 1,390e-7 |
| Descenso float64 | 59 | 7,819e-7 / 8,970e-7 | 3,519e-7 |

Los residuales se recalculan en CPU float64. El error de solución se compara con
`numpy.linalg.solve` float64, no con un oráculo de precisión arbitraria. El criterio
de parada controla residuo, por lo que el error relativo de solución puede exceder rtol.
Los tiempos y errores de suma/energía/covarianza también se conservan por caso.

## Artefactos y reproducción

Artefactos locales: `.artifacts/routing-completion/`. `before.whl` es el wheel de
la ronda staging/RNG; se guarda el SHA de cada módulo de control. `completion-control`
es la tanda fallida y se excluye. `completion-control-v2` y `completion-confirmation`
contienen las rondas válidas. `corrected-completion` rectifica solo metadatos de
referencia de idle/download del primer control, conservando original/SHA: tiempos
y errores no cambian. `completion-summary.csv` y `routing-summary.csv` contienen
todas las filas y dispersión; `summary.txt` permite inspección rápida.

```powershell
# Configurar glslang, runtimes y la ruta del wheel histórico local previamente.
python -m benchmarks.isolated --worker completion --backends vulkan --rounds 2 --repeats 31 --max-temperature 52 --control-wheel PATH_TO_WHEEL --output NEW_COMPLETION_DIR
python -m benchmarks.isolated --worker routing --backends opencl,vulkan,cuda-custom --rounds 2 --repeats 31 --max-temperature 52 --control-wheel PATH_TO_WHEEL --output NEW_ROUTING_DIR
python -m benchmarks.routing_host --control-wheel PATH_TO_WHEEL --output NEW_HOST_JSON
```

Los workers de control ejecutan explícitamente el módulo del wheel local confiable;
no descargan código. El script adicional `interleaved.py` queda en el artefacto con
sus JSON para reproducir el diagnóstico específico. La verificación final de
tests, empaquetado y dependencias figura en [verification.md](verification.md).

## Siguiente margen útil

1. Fusionar envío y marcador de finalización cuando el trabajo aún no se ha enviado,
   sin añadir fences por dispatch. Perfilar el grabador nativo solo si la preparación
   Python restante lo justifica.
2. Continuar transpuestas lógicas/flags, GEMV/DOT y kernels custom, con controles
   por forma/dtype y precisión; el routing se migra al tocar cada operación.
3. Modelar capacidades de operación y futura fusión CUDA sobre las factorías ya
   disponibles. No se necesita un framework general de plugins para el paso siguiente.
4. Ampliar RNG solo como P2: enteros sin sesgo, streams/spawn y baterías estadísticas
   extensas. Su aritmética y los efectos con estado siguen fuera de compile.
