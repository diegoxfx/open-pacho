# Seguimiento del error numérico

La [ronda de GEMM rectangular, cuBLAS y layouts](three-iterations-results.md)
mantiene los errores junto a tiempos. Los tiles custom nuevos conservan los bits
de la ruta anterior en 84 grupos medidos; siete grupos siguen fuera de la
tolerancia float32 por acumulación larga/cancelación, incluyendo controles donde
la optimización ni siquiera se activa. No se elevaron tolerancias ni se presenta
la mejora de velocidad como mejora de precisión. Clasificar estas alertas y
definir garantías por operación es parte de la [preparación de release](release-readiness.md).
La confirmación aislada Vulkan también registra un elemento fuera del mismo
umbral en un producto con K=8191; lo reproducen sus dos controles y los cinco
tiles candidatos. CLBlast con flags y la selección vectorial cuBLAS pasan sus
respectivas tolerancias en las rondas medidas.

El tiempo y la calidad del resultado se registran por separado. Las comprobaciones
numéricas se ejecutan después de detener el cronómetro y sincronizar el backend;
no se eliminan las validaciones ni se relajan las tolerancias del benchmark.

La [reevaluación de padding](padding-reassessment.md) del 19 de septiembre
conserva resultados idénticos bit a bit en 172 grupos al activar/desactivar
padding. En la suite principal aparecen tres grupos CUDA con un elemento fuera
de `rtol=5e-5, atol=5e-5` contra float64: K=8191, formas/semillas documentadas
en el informe. Los dos controles sin padding y la modalidad con padding fallan
en los mismos elementos. Una reproducción CPU con productos y suma secuencial
float32 coincide exactamente con el valor CUDA del caso investigado. Queda
pendiente mejorar acumulación/cancelación; no se cambia FMA ni tolerancia.
`benchmarks.padding_decision` distingue `complete` de `all_reference_close` y
conserva las violaciones finitas mientras mide los demás modos.

La [ronda de staging/RNG](staging-random-results.md) añade seguimiento separado de
distribuciones: vectores Philox conocidos, continuidad de contadores, rangos,
momentos, histograma/colas y error de transformaciones frente al mismo dtype CPU.
No reemplaza las pruebas adversas matriciales ni resuelve sus alertas de cancelación.
Los kernels matriciales y algoritmos de solvers no cambian en esa ronda.

La [ronda de guardas GEMM](kernel-stability-results.md) vuelve a medir 42 casos
adversos OpenCL: las mismas nueve alertas y cero excepciones. Los 88 controles
de llamadas públicas conservan sus métricas entre el generador anterior/actual;
el cambio de guardas no modifica la acumulación. CUDA/Vulkan no repiten la suite
adversa en esta ronda porque mantienen sus shaders predeterminados.

La [ronda de ejecución compartida](backend-sharing-results.md) del 13 de septiembre
repite 158 configuraciones con Decimal80: cero excepciones y las mismas 40 alertas,
sin casos nuevos ni resueltos. CUDA sigue declarando 52 combinaciones de
normas/solvers fuera de alcance. Las 104 configuraciones aplicadas irregulares
tienen resultados finitos, con error relativo L2 máximo de 7,95e-7 en float32
y 1,51e-15 en float64 frente a NumPy float64; para float64 es igual precisión.
Estas entradas ordinarias no sustituyen los diagnósticos adversos.

La [ronda de comandos Vulkan](vulkan-dispatch-results.md) repite sus 42 casos
adversos: las mismas diez alertas, sin excepciones nuevas. La corrección de
potencia float32 con bases negativas se cubre además mediante regresiones eager,
compile y map-reduce; no resuelve las limitaciones de cancelación de reducciones.

La [ronda CUDA inicial](cuda-initial-results.md) añade proveedores nativos
`cuda-cublas` y `cuda-custom` a `benchmarks.accuracy`. Se registran explícitamente
sus exclusiones de normas/solvers. En esta ronda hay 158 configuraciones medidas
y 40 alertas: seis por proveedor CUDA, además de las 28 previas de OpenCL/Vulkan.
Las alertas CUDA son cancelación en suma, dot y GEMM; finitud no equivale a precisión.

## Mediciones a escala

`benchmarks.execution`, `benchmarks.scale` y `benchmarks.prepared` registran el
error absoluto máximo y el error relativo L2, junto al dtype y la referencia.
Para operaciones se recalcula una referencia NumPy en float64 a partir de los
inputs ya redondeados al dtype original. Para resultados float64 esto es una
comparación de igual precisión, **no una certificación de exactitud**. Cambiar el
orden de reducción, el proveedor BLAS o la contracción FMA puede cambiar los bits
del resultado sin incumplir el contrato numérico.

Los solvers de `examples/solvers.py`, fuera de la API del paquete, registran además el residuo verdadero `b - A @ x`, evaluado con
escalado y float64, y el error hacia atrás normwise:

`||b - A @ x||₂ / (||A||F * ||x||₂ + ||b||₂)`.

La norma de A es Frobenius: se indica explícitamente para evitar confundir esta
métrica barata con una norma espectral. Se registra también el error hacia la
solución generadora cuando está disponible. Como el RHS se genera mediante una
multiplicación en el dtype de entrada, esa solución puede diferir ligeramente de
la solución exacta del sistema redondeado. La etiqueta de referencia conserva
esta limitación. El informe de sistemas preparados separa las métricas de cada
modalidad, para hacer visibles cambios de precisión o de convergencia.

Las normas diagnósticas usan escalado; no elevan directamente al cuadrado valores
extremos. Una referencia cero produce error relativo cero si el resultado también
es cero, e infinito explícito en otro caso. Los resultados no finitos se cuentan y
se marcan. JSON usa `null` o una cadena explícita cuando no puede representar un
número; no escribe tokens NaN o Infinity incompatibles con JSON estricto.

`benchmarks.gaps_tables` exporta columnas de error y una tabla de tiempo residente
junto a precisión. Los informes históricos sin estas métricas muestran una celda
vacía; no se inventa un error a partir de una antigua marca de validación.

## Casos adversos con referencia más precisa

```powershell
.venv\Scripts\python.exe -m benchmarks.accuracy --repeats 3 --output .artifacts/benchmarks/accuracy/raw.json
```

El comando produce JSON y Markdown. Para una ejecución sin GPU:

```powershell
.venv\Scripts\python.exe -m benchmarks.accuracy --backends numpy,pacho-numpy --output .artifacts/benchmarks/accuracy/cpu.json
```

La suite usa `decimal` de la biblioteca estándar con 80 dígitos. Convierte los
valores flotantes de entrada exactamente a Decimal y calcula las referencias de
suma, DOT, GEMV, GEMM, energía y norma. Para sistemas pequeños resuelve por
eliminación con pivoteo el sistema realmente representado, y evalúa también su
residuo en Decimal. La referencia de la energía corresponde a la expresión real;
la diferencia incluye el redondeo de sus operaciones intermedias.

Se inspecciona `finfo(longdouble)` y se guardan sus bits de mantisa: no se supone
que Windows ofrece un tipo NumPy más preciso que float64. Decimal80 es una
referencia de alta precisión para estos casos acotados, no aritmética exacta para
cualquier combinación imaginable de rangos.

Los casos cubren cancelación, normas que antes sufrían underflow/overflow,
aritmética subnormal, tamaños matriciales irregulares, fusión y sistemas con RHS
pequeño/grande. Los sistemas de Hilbert permiten observar que un residuo pequeño
puede coexistir con un error grande en la solución. Se publica la condición 2
estimada en float64, sin tratarla como una cota certificada.

CuPy cubre las operaciones y solvers expresados con su API; cuBLAS y nvmath cubren
GEMM mediante los adaptadores existentes. `numpy` identifica el comparador directo
y `pacho-numpy` la implementación de OpenPacho sobre su backend CPU. Los detalles
del algoritmo de solver comparador deben leerse junto a los resultados;
la ruta OpenPacho usa el ejemplo externo `examples/solvers.py`. La ronda
final usa la misma recurrencia normalizada en OpenPacho y los comparadores.
NumPy calcula la norma escalando primero; el solver CuPy llama cuBLAS NRM2 para
evitar una norma frágil basada en `sum(v*v)`. Se restaura su pointer mode después
de cada llamada. Los casos de operación `norm` de la suite siguen midiendo la
función nativa de cada comparador; esos resultados no describen cuBLAS NRM2.
Los registros de desarrollo con recurrencias anteriores son históricos.

Los umbrales de esta suite son indicadores para investigar, **no una promesa
universal de error relativo**, ni reemplazan las tolerancias de los tests. Una
cancelación severa o un problema mal condicionado puede superar el umbral aun
con una implementación válida. El proceso sale con código distinto de cero si
encuentra excepciones, backends pedidos no disponibles o indicadores fuera del
umbral; conserva todas las filas y no interrumpe el resto de casos. Los tiempos
de estos inputs pequeños sirven para contextualizar el diagnóstico, no para
clasificar el rendimiento a escala de los backends.
