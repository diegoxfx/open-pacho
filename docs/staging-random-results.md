# Staging Vulkan y generación aleatoria — 2026-09-13

La línea principal continúa en ejecución/transferencias Vulkan. El P2 incorpora
[generación aleatoria residente](random.md), sin cambiar algoritmos de ejemplos,
solvers, tolerancias ni frecuencia de chequeo. No hay nuevas dependencias.

## Control de staging

Se separó el mapeo host de la vida útil del lease GPU. Una asignación de staging
puede conservar su mapeo al volver al pool; ningún buffer pendiente se recicla
por estar mapeado. El backend conserva barreras y espera de readback. La expulsión
y liberación desmapean antes de liberar memoria; el presupuesto de pool sigue
siendo compartido, 128 MiB y 128 buffers. Memoria host coherente evita flush /
invalidate explícitos, pero no sustituye la sincronización; ver la
[especificación de Vulkan](https://github.khronos.org/Vulkan-Site/spec/latest/chapters/synchronization.html).

El primer control de modo comparó mapeo transitorio/persistente del código nuevo.
El control siguiente carga únicamente `upload/download` del wheel anterior
local y los enlaza al **mismo backend/contexto**. Conserva kernels, pool, driver
y aplicación, restaura métodos al terminar y registra el hash del código anterior.
Cada bloque se calienta al menos 0,25 s / tres llamadas. Hay dos rondas de 31
muestras, ocho casos por dtype float32/64: **64 configuraciones**.

Mantener todos los mapeos elimina las llamadas repetidas, pero sus tiempos son
mixtos. CG float64 baja de 19,103→18,313 ms y 18,091→16,946 ms en ejecución
completa (4–6 %); su diagnóstico registra 51 mapas→0. CG float32 es mixto en
residente y apenas cambia en completo. Una reducción de energía float64 empeora
0,529→0,584 ms y 0,540→0,587 ms en completo. No se presenta la variante general
como mejora universal: se ensaya a continuación persistencia solo de readback.

El control final de **readback únicamente**, otras 64 configuraciones, conserva
las subidas con map/unmap por copia y elimina los mapas de descarga repetidos.
Esta es la selección predeterminada. Las barreras y los kernels siguen iguales.

| Caso completo Vulkan | Anterior → readback persistente, ronda 1 (ms) | Ronda 2 (ms) |
| --- | ---: | ---: |
| Energía 2049×33, float32 | 0,493 → 0,494 | 0,489 → 0,480 |
| Energía 2049×33, float64 | 0,679 → 0,675 | 0,516 → 0,495 |
| CG 129×129, float32 | 12,400 → 11,448 | 12,408 → 11,879 |
| CG 129×129, float64 | 21,136 → 19,265 | 20,279 → 17,675 |

CG completo mejora aproximadamente 4–8 % float32 y 9–13 % float64 en estas dos
rondas. Las descargas grandes apenas cambian; una descarga escalar float32 incluso
empeora en una ronda. No se atribuye toda fluctuación a map/unmap ni se anuncia
que la persistencia resuelva el coste de Vulkan. La evidencia favorece esta
selección acotada frente a conservar todos los mapeos.

Las diferencias entre tiempos residentes de controles sin cambios también
muestran ruido. Un tiempo host de Vulkan incluye binding, driver y esperas;
no se interpreta pared menos timestamp GPU como tiempo puro de Python.

## P2: aleatorios en GPU

La fachada reserva contadores; los backends escriben la salida directamente.
Philox4x32-10 comparte algoritmo entero entre las APIs y usa sus instrucciones
de multiplicación alta uint32. No introduce un estado por hilo ni requiere
atómicos, shaderInt64 o cuRAND. La semilla es un escalar dividido en dos palabras,
no un array aleatorio generado por CPU. La versión NumPy del mismo algoritmo es
una referencia funcional; el comparador de rendimiento utiliza NumPy directamente.

La selección de distribución se hace por capacidades. Las transformaciones
normal/exponencial Vulkan en float64 quedan fuera del contrato por falta de
trascendentes GLSL de doble precisión; el resto admite float32/64 según fp64.
La generación con estado se rechaza dentro de compile/fuse. Las restricciones
y garantías de semilla/contador están en [random](random.md).

Se probaron tres formas: 33, 257×129 y 1025×1023, cuatro distribuciones y dos
dtypes, sobre NumPy, OpenCL, Vulkan, CUDA custom y CuPy. Dos rondas con 21
muestras dan **228 configuraciones válidas y 12 exclusiones explícitas** de
Vulkan float64 normal/exponencial. No hubo fallos de validación.

Normal float32, 1025×1023; cada celda es ronda 1 / ronda 2, milisegundos:

| Implementación | Resultado residente | Generación y entrega host | NumPy seguido de subida |
| --- | ---: | ---: | ---: |
| NumPy 1 hilo | 7,811 / 8,000 | 8,044 / 7,909 | — |
| OpenPacho OpenCL | 0,222 / 0,263 | 1,148 / 1,045 | 8,799 / 8,569 |
| OpenPacho Vulkan, selección final | 0,233 / 0,218 | 1,668 / 1,641 | 8,915 / 8,866 |
| OpenPacho CUDA custom | 0,111 / 0,112 | 1,195 / 1,018 | 8,931 / 8,627 |
| CuPy | 0,037 / 0,036 | 0,946 / 1,072 | 8,767 / 8,714 |

Vulkan usa la repetición con selección final: 36 configuraciones adicionales
válidas y 12 exclusiones de capacidad. Los otros proveedores conservan los
controles completos anteriores al endurecimiento final de validación de escalares;
sus kernels, contadores y transformaciones no cambiaron.

Aquí hay valor incluso entregando al host: generar en GPU evita subir una matriz
creada por CPU. En 33 elementos, NumPy sigue ganando: normal tarda 1,3–2,7 µs,
frente a unos 200–218 µs OpenCL/Vulkan, 104–168 µs CUDA y 30–31 µs CuPy. No se
introduce un fallback silencioso para ese tamaño.

CuPy conserva una ventaja clara en normal grande; nuestra implementación aún
usa un bloque Philox por salida y calcula solo un componente Box-Muller. Hay
margen para mejorar ocupación/instrucciones y trabajo por hilo respetando el
contrato de contadores. No se declara esa diferencia un límite físico.

Los comparadores usan algoritmos diferentes: Philox4x32-10 propio, PCG64 de
`numpy.random.default_rng` y el cuRAND predeterminado de `cupy.random.RandomState`.
NumPy usa random/standard_normal/standard_exponential con dtype nativo; uniform
no tiene dtype y se convierte dentro del tiempo. Las medidas incluyen asignación,
generación y sincronización, y descarga solo donde se indica; ninguna es un
timestamp de kernel puro. La primera llamada y el calentamiento están separados.

## Precisión y estadística

Tres vectores conocidos publicados de Random123 comprueban el algoritmo entero.
Las pruebas GPU contrastan con ese núcleo CPU y comprueban continuidad al partir
llamadas, acarreo de contador uint32, agotamiento uint64, tamaños vacíos, rangos
adyacentes, salida 0-D, independencia de llamadas y ausencia de upload de matrices.

En las comparaciones, `random` coincide exactamente con la referencia Philox CPU.
El máximo error absoluto float32 observado en las distribuciones transformadas
es 9,54e-7 OpenCL/CUDA y 2,75e-6 Vulkan. En float64 el máximo observado es 8,89e-16
(Vulkan únicamente uniforme). Esta referencia usa el **mismo dtype**: no es un
oráculo de precisión mayor ni reproduce las secuencias de NumPy/CuPy.

Se registran media, varianza, extremos, no finitos y correlación de lag 1. Las
pruebas incluyen histograma uniforme y colas/proporciones de normal/exponencial
con semilla fija. Son controles de regresión estadística, no TestU01/PractRand.

Una primera prueba detectó pérdida de precisión de la transformación normal GLSL
cuando `log(1-u)` se evalúa cerca de cero. Se corrigió con una serie para u < 1/8,
manteniendo la tolerancia original. OpenCL/CUDA/CPU usan `log1p(-u)`. No se cambia
la precisión de trabajo ni se habilita fast math.

## Reproducción y alcance

El runner aislado espera la salida del proceso anterior, observa reposo y luego
calienta el siguiente proveedor. Los diez reposos RNG duraron 10,4–31,1 s: hubo
esperas más largas de 16,6, 31,1 y 29,0 s. Todas cumplieron el criterio configurado;
no se fijan clocks, potencia ni ventiladores, ni se elimina todo ruido del SO.
Solo se comprobó RTX 5060 Ti / Windows / driver NVIDIA 616.64.

```powershell
python -m benchmarks.isolated --worker staging --backends vulkan --control-wheel <wheel-anterior> --rounds 2 --repeats 31 --output <directorio-nuevo>
python -m benchmarks.isolated --worker random_generation --backends numpy-1t,opencl,vulkan,cuda-custom,cupy --rounds 2 --repeats 21 --output <directorio-nuevo>
```

Artefactos en `.artifacts/staging-random`: `staging-control` es exploratorio;
`staging-final` controla todos los mapeos contra el wheel histórico; `readback-final`
comprueba la selección final. `random-control`
contiene las diez ejecuciones RNG. Los JSON conservan cada muestra, errores,
telemetría y hashes. No se publicó una versión.

`random-vulkan-final` comprueba el backend después de seleccionar readback.
`corrected-random` contiene informes derivados que corrigen únicamente la etiqueta
`reference_is_higher_precision`: la referencia RNG usa el mismo dtype y debe ser
false. Los originales y sus hashes se conservan; tiempos, momentos y magnitudes
de error no se modifican. El helper y su prueba de regresión ya evitan ese error.

La suite completa pasó **1.202 pruebas, una omitida** por requerir dos dispositivos
OpenCL. Tras los últimos ajustes pasaron **72 pruebas RNG/staging**, **198 Vulkan**
y **30 de métricas/runner**. Wheel desde sdist: **354 pasan, 772 omitidas** en CPU,
con importación verificada dentro del wheel y extras independientes. Ver el
[alcance exacto de verificación](verification.md). No se repitió precisión adversa
matricial: sus kernels permanecen iguales y sus alertas anteriores siguen abiertas.
