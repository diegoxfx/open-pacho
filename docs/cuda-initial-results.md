# Vulkan por capacidad y CUDA inicial

Iteración del 2026-09-12 en RTX 5060 Ti, Ryzen 7 5700X3D, Windows 11 y driver
616.64. Continúa [solvers y aplicaciones](solver-applications-results.md).
Los números de sesiones anteriores se mantienen como históricos.

## Optimización portable de Vulkan

El pool anterior evitaba que tamaños viejos bloquearan permanentemente a los
nuevos, pero alternar formas cercanas seguía generando muchas reservas. Ahora
prefiere capacidad exacta y, en su ausencia, toma el menor buffer libre compatible
hasta `2*max(solicitado,256)`. No crea buffers más grandes para este propósito.

La capacidad original del VkBuffer se conserva al reciclar; el tamaño lógico
se usa para descriptores y copias. Los bytes nativos asignados mantienen su
contabilidad separada. Se conservan locks, referencias en vuelo, clases separadas
para GPU/subida/readback y límites de 128 entradas/128 MiB.

Se alternaron cuatro procesos: baseline, optimizado, baseline, optimizado. El
baseline carga un snapshot del runtime anterior; compilador y shaders son iguales.
Cada proceso conserva cinco matrices/transpuestas y alterna ridge de veinte pasos,
GEMV y GEMV transpuesto, con 21 muestras por caso. Medianas residentes:

| Forma ridge float32 | Baseline, dos procesos | Optimizado, dos procesos |
| --- | ---: | ---: |
| 4096×128 | 20.54–21.72 ms | 7.28–7.78 ms |
| 4097×128 | 9.07–9.94 ms | 7.09–7.52 ms |
| 4096×129 | 9.43–11.65 ms | 6.92–7.79 ms |
| 4097×129 | 22.07–22.71 ms | 6.81–7.55 ms |
| 4095×127 | 36.70–36.92 ms | 7.16–7.34 ms |

Un perfil separado de diez rondas mixtas pasó de **1.432 reservas y liberaciones
a cero**, conservando 4.100 dispatches. Los errores de las quince operaciones
coincidieron exactamente entre procesos; ridge quedó entre 5.79e-8 y 8.48e-8
de error relativo L2 frente a la trayectoria float64.

La caché final retuvo 3.05 MiB frente a 1.00 MiB. Es memoria libre retenida, no
pico de memoria total; ambas versiones conservaron 128 entradas. Formas muy
distintas o un conjunto que exceda los límites todavía pueden causar reservas.
El resultado es una mejora del runtime portable, sin ajuste por fabricante/SO.

## Primera base CUDA

`_cuda.py` implementa contexto primario retenido, stream propio, memoria residente,
pool acotado, NVRTC, aritmética/broadcasting, gather, reducciones flotantes y
productos de vectores/matrices 2D. El proveedor por defecto es cuBLAS mediante
bindings nvmath; también se puede elegir un kernel custom directo. CuPy no es
dependencia del runtime. No hay fusión CUDA, normas estables, solvers ni lotes.

La selección y límites están en el [contrato CUDA](cuda.md). `compile` ya puede
capturar y reutilizar el plan, pero las filas CUDA se etiquetan explícitamente
como `eager-primitives`: no anuncian kernels fusionados.

La revisión añadió protección para fronteras nativas importantes: completar
la subida antes de publicar el array al stream no bloqueante, retirar cada
puntero liberado del pool aunque una liberación posterior falle, y sincronizar
la expulsión de módulos antes de cargar uno nuevo. Hay pruebas CPU que reproducen
estas condiciones y pruebas reales de propiedad/restauración del contexto.

cuBLAS usa `PEDANTIC_MATH`; NVRTC deshabilita FMA contraction, FTZ y fast-math.
Son decisiones explícitas de esta etapa. No se atribuyen diferencias con CuPy
a Tensor Cores ni al límite físico del hardware sin un control correspondiente.

## Aplicaciones

La suite compara las mismas expresiones, datos, dtypes, preparación y veinte
pasos de ridge. Incluye NumPy 1/8 hilos, OpenCL custom/CLBlast, Vulkan, CuPy y los
dos proveedores CUDA. OpenPacho aparece en modo eager y plan compilado; CUDA
todavía carece de fusión. Son 13 implementaciones intercaladas, siete muestras
residentes y completas por caso. La preparación y diagnósticos quedan visibles.

Para 16384×256 float32, las medianas de planes OpenPacho y controles son:

| Implementación | Covarianza residente/completa | Error L2 covarianza vs float64 | Ridge residente/completo |
| --- | ---: | ---: | ---: |
| NumPy 8 hilos | 7.432 / 6.888 ms | 1.161e-7 | 5.330 / 13.516 ms |
| OpenCL custom | 4.216 / 7.073 ms | 1.907e-6 | 5.979 / 8.471 ms |
| OpenCL + CLBlast | 1.545 / 3.864 ms | 1.907e-6 | 12.303 / 14.920 ms |
| Vulkan | 1.898 / 4.902 ms | 1.906e-6 | 7.498 / 8.290 ms |
| CuPy | 0.861 / 4.072 ms | 1.125e-7 | 4.804 / 6.300 ms |
| CUDA + cuBLAS, sin fusión | 0.859 / 3.471 ms | 1.288e-7 | 12.853 / 15.556 ms |
| CUDA custom, sin fusión | 2.845 / 5.240 ms | 1.906e-6 | 19.633 / 22.905 ms |

CUDA/cuBLAS queda cerca de CuPy en covarianza; la diferencia de dos microsegundos
no establece una victoria. En ridge tarda 2.68× CuPy y más que OpenCL/Vulkan.
El producto custom inicial no tiene tiling y tampoco es una alternativa rápida
general a cuBLAS. El nombre del backend por sí solo no explica su rendimiento.

Ridge CUDA/cuBLAS tiene error L2 6.802e-8 y CUDA custom 4.291e-7 frente a la misma
trayectoria float64. Estas diferencias numéricas también se conservan en el
reporte. Veinte pasos no significan convergencia. Las modalidades se cronometran
en rondas independientes; ruido puede hacer que la mediana completa sea menor
que la residente. Los JSON contienen muestras y percentiles.

La suite mixta tiene más implementaciones que la anterior. No se atribuyen las
diferencias entre sesiones a la optimización del pool: el control alternado
anterior es la evidencia que aísla ese cambio.

Ambas rondas aplicadas terminaron: **104 configuraciones alineadas y 104
irregulares**, siete muestras por modo, cero errores/ausencias y todos los
diagnósticos finitos. En ridge 4097×129 float32, las medianas residentes fueron
2.753 ms NumPy 8 hilos, 5.907 CuPy, 6.606 OpenCL custom, 6.423 CLBlast, 7.644
Vulkan, 14.959 CUDA/cuBLAS y 12.140 CUDA custom. La diferencia pequeña entre
OpenCL custom y CLBlast no es una superioridad general del proveedor.

## Precisión adversa y verificación

La suite Decimal80 midió **158 configuraciones**: 42 por ruta OpenCL custom,
CLBlast y Vulkan, y 16 por cada proveedor CUDA inicial. Las 26 combinaciones de
norma/solver por proveedor CUDA se registran en `unsupported_cases`, no como
ejecuciones correctas. Hubo cero excepciones/backends ausentes y salidas finitas,
pero **40 alertas numéricas**: las 28 previas y seis por cada proveedor CUDA.

Las doce alertas CUDA corresponden a cancelación en sum, dot y GEMM, en ambas
precisiones. Los errores relativos L2 rondan 0.9565–1.0 frente al cálculo Decimal
de los inputs realmente representados. Una salida finita con dtype correcto
puede perder prácticamente todo el resultado de esa cancelación. La reducción
por árbol y cuBLAS pedantic no garantizan una acumulación compensada. Mejorar
esas rutas o permitir acumulación más precisa explícita tiene margen, coste y
contrato propios; no se presenta como una frontera inevitable del hardware.

CUDA conservó la suma de subnormales en ambos dtypes en este control; Vulkan
float32 mantiene su alerta previa. No se generaliza ese resultado a todos los
shaders, proveedores o dispositivos. El comando de precisión retorna código 1
por alertas diagnósticas, no por excepciones. Sus umbrales no son una promesa
universal de precisión del API.

La suite completa pasa **1.019 pruebas, con una omisión** por requerir dos
dispositivos OpenCL. Incluye 38 casos CUDA, siete regresiones CPU de fronteras
nativas y nueve pruebas nuevas de reutilización Vulkan. OpenCL, Vulkan y CUDA
fueron obligatorios; CLBlast nativo estuvo habilitado.

El wheel se construyó desde sdist y se instaló sin dependencias GPU en el entorno
CPU: **311 pasan, 672 omitidas**, importando desde site-packages fuera de `src`.
CUDA también ejecutó un subprocess que bloquea CuPy. Ruff, formato y sintaxis
Python 3.10 se comprueban; el runtime se ejecutó en Python 3.12.14. No se ha
verificado instalación CUDA en un segundo sistema ni CI remota.

## Prioridad siguiente

1. Mantener Vulkan como trabajo principal: perfilar envío, lectura escalar y
   staging después de reducir reservas; medir un ejecutor nativo acotado si
   domina Python/CFFI. Comprobar un segundo fabricante cuando exista hardware.
2. CUDA inicial: medir GEMV/DOT dedicados y coste por llamada/contexto antes de
   ampliar operaciones. Después implementar lowering fusionado y normas estables,
   con precisión y vida útil como requisitos para habilitar solvers.
3. Mantener comparaciones aplicadas y formas irregulares; ampliar distancias y
   transformaciones por lotes cuando esas cargas guíen una mejora concreta.
4. Optimización por familias de hardware/SO continúa como track secundario.
   Dispersas y compatibilidad completa con NumPy siguen fuera de esta etapa.

## Evidencia local

Controles y snapshot: `.artifacts/cuda-initial`. Resultados y logs:
`.artifacts/benchmarks/cuda-initial`. Los cuatro archivos
`vulkan-capacity-{baseline,current}-{1,2}.json` aíslan el pool. Las aplicaciones
finales son `applications.json` y `applications-irregular.json`.
Los archivos `applications-quick*.json` y `cuda-cli.json` son verificaciones
exploratorias previas al cierre de fronteras nativas y no sustentan la tabla.
`accuracy.json` conserva alertas y exclusiones. Distribuciones en
`.artifacts/dist-cuda-initial`; hashes de fuentes/artefactos en
`.artifacts/benchmarks/cuda-initial/manifest.json`.
