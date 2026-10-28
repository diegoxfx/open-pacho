# GEMM CUDA custom float64: diagnóstico y experimento

La [red densa](dense-network-results.md) reveló una diferencia repetida entre
CUDA custom y OpenCL/cuBLAS en float64. Esta iteración investiga esa operación
antes de ampliar la API. Los resultados son locales a una RTX 5060 Ti, Windows,
driver 617.14; no permiten concluir lo mismo para otras GPU.

## Hipótesis y perfil

La implementación custom CUDA actual calcula cada salida float64 recorriendo K
y leyendo ambos operandos desde memoria global. El GEMM por tiles de registros
existente solo admite float32. OpenCL ya dispone de un tile compartido float64.
No basta comparar el tiempo de las aplicaciones para atribuir la diferencia a
GEMM: SGD también materializa transpuestas, reduce y evalúa funciones elementwise.

Se instrumentaron los lanzamientos de cuatro planes compilados con eventos CUDA
en el stream de ejecución, conservando diez muestras de tiempo completo del plan
sin instrumentar y tres repeticiones instrumentadas por caso. Las salidas de
todas las repeticiones conservaron sus bits y pasaron la comprobación numérica
contra NumPy float64. Los archivos del runtime no cambiaron durante el perfil.

| Caso float64 | Mediana del plan sin instrumentar, ms | Fracción GEMM de los eventos GPU registrados |
| --- | ---: | ---: |
| Inferencia, 4096×128×256×32 | 2.130 | 95.28% |
| SGD, 4096×128×256×32 | 4.909 | 91.22% |
| Inferencia, 4095×127×255×31 | 2.128 | 95.21% |
| SGD, 4095×127×255×31 | 4.831 | 92.89% |

Los eventos alteran el programa medido. Sus fracciones son un diagnóstico de
los lanzamientos registrados, no una descomposición exacta del tiempo completo
sin instrumentación. Tampoco se interpreta la resta entre tiempo de pared y
eventos como tiempo de Python. El perfil justifica un experimento acotado de GEMM.

## Candidato

`benchmarks/cuda_fp64_tile.py` genera un tile compartido 16×16: 256 hilos y
4096 bytes de memoria compartida estática por bloque. Carga operandos de forma
cooperativa y conserva exactamente K productos en orden creciente, sin añadir
productos de padding en la última iteración. Mantiene las opciones CUDA actuales:
sin contracción FMA, sin flush de subnormales y sin reducir precisión.

La generación admite matrices contiguas y broadcasting de lotes. Vectores,
dimensiones vacías y K=0 quedan fuera del candidato; la ruta existente conserva
esos contratos. Sus requisitos de lanzamiento se comprueban separadamente de
los del kernel float32.

## Validación y comparación

La comprobación nativa incluye 27 casos candidatos con dos destinos previamente
rellenados, con NaN e infinito: 54 lanzamientos. Cubre bordes 8/15/16/17/31/32/33,
broadcasting de rango cuatro, cancelación, desbordamiento, subnormales y ceros con
signo. Todos conservan los bits finitos del kernel directo y sus clases NaN/+Inf/-Inf;
no se exige conservar el payload de NaN. Nueve casos adicionales comprueban que
vectores, vacíos y K=0 siguen fuera del candidato y conservan su resultado previo.

Muestras seleccionadas se contrastan también con productos y sumas exactos de
fracciones de los valores binarios originales, separando redondeo del producto
y acumulación. La equivalencia con el control no significa exactitud matemática:
se conservan los errores de cancelación, redondeo y rango del algoritmo original.
NumPy float64 se etiqueta como referencia de igual precisión.

El runner `benchmarks.cuda_fp64_gemm` captura las fuentes reales del kernel
directo, sin ejecutarlas durante la captura, y conserva dos controles con esa
misma fuente. La comparación usa 24 GEMM y seis aplicaciones: inferencia/SGD
pequeña, regular e impar. Cada caso tiene tres modos, 24 repeticiones y las seis
permutaciones de orden con igual frecuencia. La salida conserva los fuentes
capturados y permite reutilizarlos mediante `--control-report` aunque el runtime
cambie en el futuro.

Se separan GEMM con salida preasignada y matmul con asignación. Las aplicaciones
usan tres planes independientes, estado original en cada llamada y dos alcances:
ejecución residente y flujo completo con subidas y descarga de todas las salidas.
Primera llamada, descubrimiento de rutas, preparación y calentamiento quedan
fuera de las muestras repetidas. Las comprobaciones numéricas quedan fuera del
reloj. La instrumentación de selección de rutas es común a los tres modos y
añade trabajo Python: estos tiempos no equivalen a ejecutar el runtime sin ella.

Las dos rondas usan procesos separados y reposo observado antes de arrancar.
Los mismos datos y fuentes se conservan entre rondas. Las muestras brutas y los
fallos numéricos se guardan incluso cuando invalidan una comparación.

## Resultado y decisión

**No se promueve el candidato.** Las dos rondas completan 60 casos, 8.640
muestras y 9.360 comprobaciones aceptadas. No hay una mejora consistente de
la aplicación que justifique añadir una ruta al runtime. Los kernels estrechos
con K grande empeoran repetidamente; varios productos grandes quedan casi iguales.

La tabla expresa tiempo candidato dividido por el promedio de las medianas
de ambos controles. Menor que 1 favorece al candidato. R1/R2 son procesos
independientes; las diferencias pequeñas no se tratan como ganancias firmes.

| GEMM M×K×N | Preasignado R1 | Preasignado R2 | Con asignación R1 | Con asignación R2 |
| --- | ---: | ---: | ---: | ---: |
| 4096×128×256 | 0.998 | 1.000 | 1.004 | 1.001 |
| 4096×256×32 | 1.085 | 1.025 | 1.051 | 1.076 |
| 256×4096×32 | 1.088 | 1.117 | 1.099 | 1.077 |
| 4096×32×256 | 1.001 | 1.001 | 1.004 | 1.035 |
| 128×4096×256 | 1.013 | 1.028 | 1.016 | 1.010 |
| 4095×127×255 | 1.005 | 1.006 | 1.001 | 1.002 |
| 4095×255×31 | 1.060 | 1.071 | 1.124 | 1.049 |
| 255×4095×31 | 1.122 | 1.110 | 1.108 | 1.095 |
| 4095×31×255 | 1.006 | 1.013 | 1.004 | 1.006 |
| 127×4095×255 | 1.006 | 1.029 | 0.999 | 1.027 |

| Aplicación | Residente R1 | Residente R2 | Completa R1 | Completa R2 |
| --- | ---: | ---: | ---: | ---: |
| inference, 31×7×11×3 | 1.012 | 1.011 | 1.093 | 0.927 |
| sgd, 31×7×11×3 | 1.070 | 0.954 | 0.961 | 1.000 |
| inference, 4096×128×256×32 | 1.014 | 1.004 | 1.024 | 1.005 |
| sgd, 4096×128×256×32 | 1.041 | 1.029 | 1.027 | 1.034 |
| inference, 4095×127×255×31 | 1.018 | 1.023 | 1.003 | 1.004 |
| sgd, 4095×127×255×31 | 1.028 | 1.028 | 1.008 | 1.037 |

En los diez GEMM grandes, la mayor relación entre medianas de los dos
controles es 1.038. En casos pequeños llega a 1.753; en aplicaciones pequeñas
a 1.158. La variación limita especialmente la interpretación de micromedidas
pequeñas. Los dos controles idénticos se conservan para hacer visible esa
variación, no para eliminarla por promediado.

Cambiar la organización de memoria manteniendo el orden y las operaciones
aritméticas no explica aquí la diferencia histórica entre APIs. CUDA compila
GEMM con `--fmad=false`; el programa GEMM OpenCL usa `acc += left*right` y no
incluye `FP_CONTRACT OFF`. Esto permite investigar contracción, pero el código
fuente por sí solo no demuestra qué instrucciones float64 ejecutó OpenCL.
Los lowerers elementwise/map-reduce OpenCL sí desactivan la contracción.

Una investigación posterior acotada puede guardar PTX/binarios de estas
formas y comparar fuentes idénticas cambiando únicamente la contracción.
Sería una política de redondeo distinta: necesita validación de error, no
igualdad bit a bit, y no autoriza cambiar los valores predeterminados.

La verificación dirigida comprende 19 pruebas CPU de generación/indexación
y siete del protocolo de captura, restauración, validación y planes. No se
ejecutó la suite global. El runtime, sus defaults y la API pública permanecen
sin cambios.

QA independiente: **APPROVE**. Se reejecutaron las 26 pruebas CPU, se verificaron
las tablas y muestras, se reprodujeron las 23 fuentes de control sin lanzarlas y
se comprobaron 172 puntos de los oráculos exactos. Los 33 archivos del runtime
son idénticos al estado anterior. La revisión no volvió a ejecutar la GPU.

## Evidencia local

Los informes y scripts privados se conservan en
`.artifacts/iterations-2026-10-22/`: `cuda-fp64-profile.json`,
`profile_cuda_fp64.py`, `cuda-fp64-correctness.json`, `validate_cuda_fp64_tile.py`,
`cuda-fp64-tile-cpu.xml` y `cuda-fp64-gemm/`. Este último contiene los dos informes,
la sesión y el resumen reproducible. El perfil conserva eventos por lanzamiento,
formas, hashes de fuentes y resultados, estado de la GPU y muestras.
No se incluyen estos artefactos locales en el paquete distribuido.
