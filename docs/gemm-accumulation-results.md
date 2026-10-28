# Acumulación GEMM float32 — 2026-10-22

Esta investigación conserva el runtime y sus valores predeterminados. La suma
por bloques elimina las violaciones de tolerancia en los casos aleatorios
estudiados, con coste bajo en CUDA y mayor en Vulkan. Neumaier también resuelve
los casos de cancelación severa, pero multiplica el tiempo de las matrices
grandes entre 2,37 y 4,99. Ninguna variante se promueve como solución universal.

## Motivo y alcance

Las rondas de [padding](padding-reassessment.md) y
[GEMM rectangular](three-iterations-results.md) conservaron errores float32 con K
largo. Antes de ampliar una política matemática pública, se separan tres causas:
redondeo de productos, acumulación y contracción multiplicación-suma.
El análisis de [Higham, *The Accuracy of Floating Point Summation*](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf)
explica por qué la cancelación y el orden importan y por qué un método de suma no
domina uniformemente a los demás. La decisión aquí usa las mediciones locales.

Solo cambian fuentes experimentales en `benchmarks/`. El generador parte de
`_gemm`/`_gemm_rect`, conserva la selección real de tile de cada backend y falla
si la estructura que transforma cambia. No añade buffers auxiliares en GPU,
packing, precisión reducida ni modificación de la configuración del backend.

| Variante | Operación por salida |
| --- | --- |
| `current` | Fuente actual, sin cambios; control de la ruta pública |
| `sequential` | Producto float32 y suma float32, sin contracción |
| `blocked` | Suma ordinaria dentro de cada tile K y suma de parciales |
| `kahan` | Corrección de Kahan en cada producto ya redondeado |
| `neumaier` | Corrección de Neumaier en cada producto; suma final de corrección |
| `blocked_neumaier` | Parciales ordinarios por tile, fusión de parciales con Neumaier |

Los tiles K son 16 o 32 según la geometría ya seleccionada. Los candidatos usan
`FP_CONTRACT OFF` en OpenCL, `precise` en los acumuladores/intermedios Vulkan y la
opción NVRTC `--fmad=false` ya existente en CUDA. Las rutas actuales OpenCL/Vulkan
permiten contracción. Esto no constituye una promesa general de igualdad de bits
entre APIs. Las correcciones omiten la cola K rellenada con ceros y manejan
operandos/sumas no finitos sin introducir `inf-inf` en la corrección. No reparan
overflow de producto, overflow de suma ni todo posible overflow de la corrección.

## Protocolo y verificación

- RTX 5060 Ti, driver 617.14, Windows, Python 3.12.14, NumPy 2.5.3.
- Seis procesos secuenciales: OpenCL, Vulkan y CUDA custom, seguidos del orden
  inverso. Reposo observado de 10,50–10,58 s antes de cada proceso; temperatura
  máxima permitida 52 °C y tolerancia respecto a la línea base de 2 °C. Las
  temperaturas observadas al comenzar fueron 38–41 °C. No se relajaron umbrales.
- Cinco entradas aleatorias fijas: `(33,31)@(31,65)`, dos semillas históricas
  972/973 de `(65,8191)@(8191,2051)`, `(65,16383)@(16383,257)` y 1024³.
  La segunda ronda repite los mismos datos y cambia el orden de medición;
  no cuenta como cinco entradas nuevas.
- Siete matrices con tres órdenes de `[1e8, 1, -1e8]`, K=15/16/17/31/32/33/8191,
  y un caso de NaN, infinitos, `0*inf`, overflow de producto y suma. K=15 es una
  prueba forzada del generador fuera del umbral de GEMM tiled de la API pública.
- **78 grupos caso/proceso, 468 resultados de variante y 4.320 muestras**:
  24 repeticiones por variante y entrada aleatoria; cada variante ocupa cada
  posición cuatro veces. Primera llamada, compilación y calentamiento excluidos.
  Calentamiento mínimo de tres llamadas y 0,10 s por variante.
- Tiempos de pared sincronizados con entradas y salida preasignadas residentes,
  incluido dispatch Python. No miden transferencia, asignación, ejecución completa
  de una aplicación ni tiempo GPU aislado. Los casos adversarios no se cronometran.
- Control idéntico bit a bit a `matmul` público en todos los casos; los resultados
  cronometrados también conservan bits tras las repeticiones.
- Referencia completa float64, tolerancia heredada `atol=rtol=5e-5`, sin relajar.
  Puntos seleccionados —incluido el peor error normalizado del control— usan
  sumas dyádicas exactas, tanto del producto original como del producto redondeado
  float32. Errores absolutos, relativos y divididos por la suma de productos
  absolutos se calculan con Decimal80. El oráculo no depende de BLAS.
- **54 pruebas CPU pasan** sobre generación, barreras/colas, algoritmos,
  cancelación/no finitos, oráculo y controles del runner. Ruff check/formato pasan.
  Una revisión independiente del oráculo contrastó además 240 puntos con Fraction.
  QA final independiente: **APPROVE**; verificó los 342 puntos guardados, hashes,
  controles, conteos y afirmaciones del informe.

## Precisión observada

Número de elementos fuera de tolerancia en las cinco matrices aleatorias, por
ronda. Ambas rondas reprodujeron los mismos conteos:

| Variante | OpenCL | Vulkan | CUDA custom |
| --- | ---: | ---: | ---: |
| Actual | 3 | 3 | 4 |
| Secuencial sin contracción | 4 | 4 | 4 |
| Por bloques | 0 | 0 | 0 |
| Kahan | 0 | 0 | 0 |
| Neumaier | 0 | 0 | 0 |
| Bloques + Neumaier | 0 | 0 | 0 |

El máximo error absoluto aleatorio baja de aproximadamente `1,83e-4` a
`2,34e-6` con Kahan/Neumaier. Bloques + Neumaier queda entre `3,56e-6` y `4,84e-6`
según backend; bloques ordinarios entre `2,96e-5` y `4,97e-5`. Son máximos de esta
suite, no cotas matemáticas ni garantías para otras entradas.

En cancelación severa, solo Neumaier pasa las siete matrices completas en los
tres backends. Kahan y las variantes por bloques siguen perdiendo términos:
compensar la suma de parciales no recupera información ya perdida dentro del tile.
Por ejemplo, las once repeticiones de `[1e8,1,-1e8]` con K=33 deben sumar 11.
Conservar este caso impide promocionar una mejora aleatoria como precisión robusta.

Todas las variantes sin contracción reproducen la clasificación secuencial
float32 de los no finitos. Las rutas actuales OpenCL/Vulkan difieren en el caso
de overflow de productos con signos opuestos; CUDA coincide. La diferencia es
consistente con la contracción permitida por sus fuentes, pero aquí no se
inspeccionó ensamblador para atribuir cada instrucción. Desactivar contracción
tampoco mejora automáticamente la precisión aleatoria: el control estricto
introduce una violación más en OpenCL/Vulkan.

## Coste medido

Rango del cociente **tiempo candidato / tiempo actual**, usando las cuatro
entradas grandes en ambas rondas. Menor que 1 significa menor tiempo. Se excluye
el caso pequeño de esta tabla porque domina el coste de dispatch y CUDA cambió
de 33,1 a 20,7 microsegundos entre procesos; no se presenta como ganancia estable.

| Variante | OpenCL | Vulkan | CUDA custom |
| --- | ---: | ---: | ---: |
| Por bloques | 0,98–1,03 | 1,12–1,18 | 0,93–0,98 |
| Kahan | 2,15–2,80 | 3,19–3,37 | 1,82–2,80 |
| Neumaier | 3,57–4,99 | 3,70–4,12 | 2,37–3,45 |
| Bloques + Neumaier | 1,05–1,17 | 1,43–1,70 | 1,00–1,09 |

Para la semilla 972 de `(65,8191)@(8191,2051)`, el control tarda 0,992–1,029 ms
OpenCL, 0,754–0,758 ms Vulkan y 0,964–0,965 ms CUDA. Neumaier tarda
3,662–3,673 ms, 3,083–3,085 ms y 2,295–2,321 ms, respectivamente.
Estos números comparan algoritmos dentro de cada backend y geometría existente;
no son una comparación general entre APIs ni contra BLAS.

## Decisión y reproducción

Se mantiene la política actual. La siguiente investigación de precisión debe
comparar una opción explícita de acumulación robusta con acumulación float64,
incluyendo producto, subnormales, rango, lotes y coste de aplicación completo.
Bloques + Neumaier merece seguimiento como compromiso para errores aleatorios;
no debe llamarse robusto frente a cancelación severa. Esta evidencia no justifica
un autotuner, selección por dispositivo ni modificar tolerancias.

Runner: `python -m benchmarks.gemm_accumulation --backend cuda-custom --repeats 24
--seed 1011 --output <archivo-nuevo.json>`; ejecutar cada backend en un proceso
aislado y usar la puerta de reposo de `benchmarks.stability` entre procesos.
Los informes conservan muestras, orden, hashes de fuentes/inputs/kernels,
primera llamada, calentamiento, errores, clasificación y telemetría.
Finalización del experimento y aceptación numérica son campos distintos.

Artefactos locales: `.artifacts/iterations-2026-10-22/gemm-accumulation/` contiene
los seis informes, logs, `session.json` y `summary.json`. El supervisor y script
de resumen están en el directorio padre. No se ha recompilado el wheel ni
ejecutado la suite completa: este lote solo añade infraestructura de investigación.
