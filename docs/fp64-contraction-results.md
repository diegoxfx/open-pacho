# Contracción float64: experimento causal acotado

Rondas completadas el 2026-10-22. El runtime y sus políticas predeterminadas
permanecen sin cambios.

La [red densa](dense-network-results.md) y el [candidato GEMM compartido](cuda-fp64-gemm-results.md)
dejaron una pregunta concreta: cuánto de la diferencia entre CUDA y OpenCL
proviene de contracción aritmética. Esta iteración compara políticas de redondeo
con código de benchmark privado del runtime. No modifica defaults ni API pública.

## Diseño

CUDA compara el GEMM directo actual y el candidato compartido 16×16 por separado.
Dentro de cada geometría, ambas compilaciones reciben exactamente la misma
fuente y opciones, salvo `--fmad=false` frente a `--fmad=true`. Se mantienen
subnormales y precisión de división/raíz. OpenCL captura su fuente GEMM actual
y añade únicamente `#pragma OPENCL FP_CONTRACT OFF` al control sin contracción.
Ambos programas OpenCL conservan `-cl-std=CL1.2`; se guardan también las opciones
efectivas y las variables de entorno que podrían alterarlas.

Cada ejecutable pertenece al experimento: se descarga después de sincronizar,
sin instalarlo en la caché del runtime. Los hashes e identidades de cachés y
flags se comprueban durante preparación y ejecución. Se conserva fuente,
opciones, versiones, capacidades, atributos del kernel y PTX/binario. El conteo
de `fma.f64`, `mul.f64` y `add.f64` se limita a la entrada `compute`: es evidencia
del IR estático del compilador, no del código máquina cargado ni del número
dinámico de operaciones. Un binario opaco, una mezcla no resuelta de instrucciones
o flags matemáticos adicionales impiden una conclusión causal fuerte.

Los 19 casos pequeños usan M=N=8 y K=15/16/17/31/32/33, entradas aleatorias,
cancelación, producto redondeado, subnormales, ceros con signo, desbordamiento
y no finitos. Sus 64 salidas se contrastan con dos recurrencias binary64:
producto y suma redondeados por separado, o producto más acumulador redondeados
una sola vez. El segundo oráculo usa fracciones exactas; no depende de BLAS ni
de FMA hardware. Cada recurrencia empieza en +0 y respeta el orden creciente
de K. Los productos +0×+0 del último tile OpenCL se modelan explícitamente:
pueden afectar el signo de un cero final.

La conformidad con una política no equivale a precisión matemática. Se conserva
por separado el producto escalar exacto de las entradas binarias y la suma exacta
de productos ya redondeados, cuando estos son finitos. En los GEMM grandes se
comprueban cinco coordenadas con esos oráculos y toda la salida contra NumPy
float64, referencia de igual precisión, con rtol=2e-10 y atol=2e-12.

Los tamaños medidos, M×K×N, son 4096×128×256, 128×4096×256 y 256×4096×32:
productos ya usados por la aplicación de red densa. Cada comparación contiene
dos controles con la misma política y buffers independientes, además del
candidato. Se usan las seis permutaciones con igual frecuencia, 24 muestras
por modo, primera llamada separada y calentamiento de al menos tres llamadas
y 0,10 segundos. Dos procesos por backend invierten el orden CUDA/OpenCL y
requieren reposo observado antes de cada worker.

El reloj mide pared con sincronización, inputs residentes y destino preasignado;
incluye despacho Python. No mide una aplicación ni aísla tiempo GPU. Cada llamada
recibe un destino con NaN/Inf alternados, fuera del reloj, y se comprueba completa
contra su primera salida. El control reproduce además los bits finitos y clases
no finitas de la ruta de producción. Las políticas distintas tienen referencias
distintas; no se exige que una FMA reproduzca el redondeo separado.
La validación y la escritura del informe quedan entre grupos de muestras y pueden
dejar la GPU sin trabajo. Son mediciones de llamadas sincronizadas en ráfagas,
no una prueba de rendimiento sostenido; el calentamiento previo no elimina ese
intervalo. Los controles intercalados permiten observar la variación en ese alcance.

## Ejecución reproducible

```powershell
.venv/Scripts/python.exe -m benchmarks.fp64_contraction --backend cuda-custom --profile all --repeats 24 --output <json-nuevo>
.venv/Scripts/python.exe -m benchmarks.fp64_contraction --backend opencl --profile all --repeats 24 --output <otro-json-nuevo>
```

Ejecutar secuencialmente, con dispositivo exclusivo y reposo externo. Los
artefactos del compilador se guardan junto al JSON en un directorio nuevo;
los archivos existentes no se sobrescriben. Los estados de ejecución, aceptación
numérica y suficiencia de evidencia son independientes y se conservan los fallos.

## Resultados y decisión

Las cuatro ejecuciones completan **132 filas, 264 variantes y 1.296 muestras
de rendimiento**, con 5.971 registros aceptados de comparación de salidas,
incluidas las primeras llamadas. Las 44.316 evaluaciones de
coordenadas contra las dos políticas incluyen
repeticiones de datos y no representan puntos independientes. No hay errores de
ejecución, aceptación numérica ni limpieza. Los 33 archivos del runtime y las
fuentes de benchmark son los mismos en las cuatro ejecuciones. Los datos de
cada caso también conservan sus hashes.

Hardware local: NVIDIA GeForce RTX 5060 Ti, Windows, driver 617.14; Python
3.12.14, NumPy 2.5.3, PyOpenCL 2026.1.4 y NVRTC 13.3.33. Los resultados no
establecen rendimiento en otras GPU ni otros drivers.

En los tres tamaños grandes, el PTX `compute` contiene ocho multiplicaciones
y ocho sumas float64 en CUDA directo sin contracción, frente a ocho FMA en su
variante contraída. CUDA tile16 y OpenCL contienen 16 multiplicaciones más 16
sumas, o 16 FMA, respectivamente. Son conteos estáticos del cuerpo, no el total
dinámico del GEMM. OpenCL nativo genera FMA en este driver; `FP_CONTRACT OFF`
la elimina. Las comprobaciones de salida concuerdan con esas políticas.

La tabla usa el promedio de las dos medianas de control cuando una política
tiene dos controles. La última columna es tiempo separado dividido por tiempo
FMA; mayor que uno favorece FMA. R1/R2 son procesos distintos.

| Ruta y M×K×N | Separado, ms R1 / R2 | FMA, ms R1 / R2 | Separado / FMA R1 / R2 |
| --- | ---: | ---: | ---: |
| CUDA directo, 4096×128×256 | 1.938 / 1.704 | 1.089 / 0.832 | 1.779 / 2.047 |
| CUDA directo, 128×4096×256 | 1.907 / 1.817 | 0.882 / 0.803 | 2.162 / 2.262 |
| CUDA directo, 256×4096×32 | 0.484 / 0.487 | 0.277 / 0.261 | 1.746 / 1.867 |
| CUDA tile16, 4096×128×256 | 1.933 / 1.707 | 1.075 / 0.826 | 1.799 / 2.066 |
| CUDA tile16, 128×4096×256 | 1.888 / 1.826 | 0.935 / 0.849 | 2.019 / 2.149 |
| CUDA tile16, 256×4096×32 | 0.527 / 0.540 | 0.277 / 0.309 | 1.904 / 1.745 |
| OpenCL nativo, 4096×128×256 | 1.927 / 1.709 | 1.069 / 0.842 | 1.803 / 2.030 |
| OpenCL nativo, 128×4096×256 | 1.869 / 1.930 | 0.910 / 0.912 | 2.055 / 2.116 |
| OpenCL nativo, 256×4096×32 | 0.529 / 0.532 | 0.284 / 0.299 | 1.862 / 1.776 |

La aceleración se repite: **1.75–2.26× en CUDA** y
**1.78–2.12× en OpenCL**, dentro de este microbenchmark.
La mayor relación entre las medianas de los dos controles idénticos es
1.087 (8,7% de diferencia). Se conservan las muestras y la dispersión; diferencias pequeñas
entre APIs o geometrías no se convierten en ganancias firmes.

La contracción explica una parte importante de la diferencia de GEMM observada
entre las rutas predeterminadas. No se atribuye toda la diferencia aplicada a
FMA: estos tiempos excluyen asignación, transferencias y las demás operaciones
del modelo. Tampoco se balanceó el orden entre geometrías, por lo que esta
iteración no selecciona una sobre otra. El [resultado anterior de tile16](cuda-fp64-gemm-results.md)
sigue sin justificar su promoción. **No cambia el runtime ni se añade un modo
matemático público.**

La reanudación conserva la primera ronda CUDA, que terminó antes de la parada
por cuota. Su puerta de reposo quedó registrada a 41 °C, pero no su telemetría
posterior. Un primer intento de reanudar agotó 300 segundos sin lanzar un worker:
la GPU ociosa estaba alrededor de 44–45 °C, por encima del antiguo umbral de
43 °C. Ese intento permanece en `session-resume.json`.

La nueva sesión establece una referencia de 45 °C y conserva el límite absoluto
de 52 °C, tolerancia de 2 °C y utilización máxima de 5% en las tres lecturas
finales estables, con una espera mínima de diez segundos. Esto no exige diez
segundos continuos sin actividad: la puerta de la segunda ronda OpenCL registra
9% de utilización GPU a los 4,265 segundos, antes de las tres lecturas finales
aceptadas. Las dos rondas OpenCL pasan sus
comprobaciones en `session-resume-2.json`; la última puerta CUDA de ese intento
agota 300 segundos con actividad y temperatura superiores a los límites, sin
lanzar su worker. Tras dejar la GPU en reposo, `session-resume-3.json` conserva
los mismos umbrales y amplía solo la espera máxima a 900 segundos. La última
ronda CUDA pasa la puerta a 45 °C en 10,516 segundos; sus lecturas y estado
posterior quedan en ese tercer registro. La pausa y el cambio de condición
térmica impiden tratar
las cuatro ejecuciones como una sesión continua idéntica; la comparación principal
es entre políticas intercaladas dentro de cada proceso y su repetición.

## Precisión y límites de FMA

En los casos pequeños aleatorios, las 384 coordenadas distintas del control
CUDA directo de la primera ronda muestran menor error absoluto con FMA en 151,
mayor en 107 e igual en 126. En las 15 coordenadas muestreadas de los tres GEMM
grandes, los conteos son 7/6/2. Se compara contra el producto escalar exacto de
las entradas binarias. No se duplican controles, geometrías ni rondas en estos
conteos, ni se interpreta esta muestra fija como una garantía estadística.

Los casos estructurados explican por qué no se debe llamar a FMA un modo de
precisión siempre superior. Sea `x=1+2^-52`, `r=1+2^-51`, `u=2^-1074` y `M` el
mayor float64 finito. La tabla describe la recurrencia CUDA en orden creciente
de K, con acumulador inicial +0; los términos indicados terminan el producto.

| Términos, en orden | Suma matemática exacta | Redondeo separado | FMA secuencial |
| --- | --- | --- | --- |
| `-r`, `x*x` | `2^-104` | `+0` | `2^-104` |
| `x*x`, `-r` | `2^-104` | `+0` | `+0` |
| `2^54`, `1`, `-2^54` | `1` | `+0` | `+0` |
| `u`, `u*0.5` | `1.5*u` | `u` | `2*u` |
| `-u*0.5` | `-u/2` | `+0` | `-0` |
| `-M`, `M*2` | `M` | `+Inf` | `M` |
| `M*2`, `-M*2` | `0` | `NaN` | `+Inf` |

FMA evita un redondeo intermedio y algunos desbordamientos, pero no recupera
información perdida en pasos anteriores. En el empate subnormal `1.5*u`, ambos
resultados tienen el mismo error absoluto; `2*u` es el resultado de redondear
una vez al par. La fracción exacta del error se conserva aunque convertirla a
float64 produzca cero.

En `negative-underflow-zero`, CUDA FMA termina en -0. OpenCL añade los productos
+0×+0 del tile incompleto y termina en +0, incluso con FMA. Esa diferencia de
orden efectivo se modela en el oráculo; no es un fallo de sobrescritura ni una
prueba de equivalencia general entre los backends.

## Criterio para la siguiente iteración

La prioridad es llevar la comparación FMA/separado al flujo completo de la red
densa en CUDA custom, primero con ejecutables experimentales y selección
explícita. OpenCL nativo ya genera FMA en este equipo. La comparación debe usar
la misma geometría, entradas, dtype y algoritmo en inferencia y SGD, incluyendo
los casos pequeño, regular e impar ya existentes. Medir preparación, primera
llamada, tiempo residente y completo; comprobar predicción, pérdida, gradientes
y parámetros. La decisión requiere una mejora repetida de esos flujos y un
presupuesto de error explícito. Este microbenchmark no demuestra esa ganancia.

Si el resultado aplicado la justifica, una política pública opt-in debe definir
su alcance por operación/proveedor y participar en la identidad de los kernels
y planes. No debe activar reasociación, FTZ u otros cambios bajo el nombre de
FMA ni reutilizar un ejecutable compilado con otra política. Los defaults actuales
permanecen como están; la ventaja temporal no convierte políticas distintas en
equivalentes bit a bit.

Como frente independiente, el [cliente de osciladores](oscillators-results.md)
justifica medir despacho, asignaciones y esperas internas antes de optimizar su
ejecutor preparado. Esa recomendación viene de sus tiempos aplicados y del
código de retención CUDA, no de extrapolar estos GEMM. El borrador privado de ese
diagnóstico no se integra ni se ejecuta en esta iteración.

## Verificación y evidencia

Los informes, fuentes y binarios quedan en
`.artifacts/iterations-2026-10-22/fp64-contraction/`. `summary.json` reúne los
conteos y medianas; los cuatro `round-*.json` conservan muestras, referencias,
opciones y hashes. Los registros `session*.json` preservan también los intentos
fallidos de reposo y la ausencia de telemetría posterior de la primera ronda CUDA.
La comprobación inicial de 60 filas permanece aparte, sin mezclarse con estas
mediciones de rendimiento.

**QA independiente: APPROVE.** Pasan las 86 pruebas CPU dirigidas de
`tests/test_fp64_contraction*.py`, además de Ruff check y formato. La revisión
reconstruye los 22 pares de entradas, recalcula 2.191 puntos de referencia
con redondeo entero explícito y sumas racionales exactas, y reconcilia artefactos,
conteos, tablas y hashes. La descripción del reposo refleja las tres lecturas
finales, sin afirmar diez segundos continuos de inactividad. El dictamen y su
auditoría están en `qa-fma-final.md`, `qa-fma-final-audit.json` y
`qa-fma-final.xml`, junto al directorio de esta iteración. No se ejecuta la
suite global ni se inicia la siguiente iteración.
