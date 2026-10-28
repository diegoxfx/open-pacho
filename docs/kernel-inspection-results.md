# Inspección de kernels y revisión de copias — 2026-09-19

La inspección encontró un problema corregible en el GEMM float32 de OpenCL:
los literales `0.0` de las cargas condicionales promovían valores a float64.
Cambiar solamente esos literales a `0.0f` elimina las conversiones y reduce
aproximadamente **38 %** el tiempo del caso irregular grande medido. Parte de la
ventaja anterior del padding evitaba esa ineficiencia. No era evidencia suficiente
para atribuir toda la mejora a eliminar branches.

La revisión de las copias de padding no detectó corrupción de datos o índices.
Se reduce preparación repetida y se corrige la selección de ámbito de ejecución
para un producto con varios envíos. No se afirma que los kernels de copia hayan
alcanzado un óptimo global ni que padding vaya a mejorar cualquier producto.

## Qué se inspeccionó realmente

Equipo local: RTX 5060 Ti, Windows, Python 3.12, driver NVIDIA 616.64. El mismo
GEMM custom usa grupos de 64 hilos, tiles 32×32, bloques K de 16 y 16 acumuladores
float32 por hilo. Las fuentes comparten esa aritmética, pero los compiladores no
generan necesariamente el mismo programa.

| Backend | Evidencia disponible | Límite de la evidencia |
| --- | --- | --- |
| OpenCL | Binario de programa devuelto por el driver, que aquí contiene PTX | PTX es IR; sus registros son virtuales, no registros físicos |
| CUDA | PTX de NVRTC, atributos de la función cargada y SASS obtenido reenlazando ese PTX | El reenlace no es un volcado del módulo ya cargado y puede diferir de su JIT |
| Vulkan | GLSL y SPIR-V exactos, estadísticas del pipeline con captura habilitada | El driver devuelve cero representaciones internas; no obtuvimos ISA Vulkan |

También se reenlazó el PTX de OpenCL con el CUDA Driver API, fuera de los relojes.
Su SASS conserva las conversiones del control anterior y las elimina con el fix.
Esto refuerza el diagnóstico, pero **no es el ejecutable OpenCL original**.
La extracción guarda fuentes, flags, hashes, PTX/SPIR-V, metadatos y desensamblado
en `.artifacts/kernel-inspection`; no añade dependencias al runtime instalable.

El adaptador Vulkan consulta la extensión
[`VK_KHR_pipeline_executable_properties`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_pipeline_executable_properties.html)
mediante ABI tipada, inicializa las estructuras y repite consultas incompletas.
El dato del driver `Local Memory Size=68719476736` por hilo es sospechoso y no tiene
corroboración. Una comprobación independiente CFFI/ctypes no encontró error de
layout o lectura; **no usamos ese valor para inferir spills ni ocupación**.
Los exports nuevos conservan bytes originales y señalan esta limitación.

## Controles del compilador

Dos rondas, tres procesos separados por ronda, tres tamaños y 21 muestras por
variante: **72 configuraciones y 1.512 tiempos de pared**, más cinco diagnósticos
GPU separados por configuración. Entradas y salidas preasignadas; fuera del reloj
quedan compilación, transferencias y preparación. Cada llamada medida sí incluye
despacho y sincronización. Son diagnósticos directos del GEMM, no aceleraciones
de una aplicación ni de la API completa. OpenCL usa una cola con profiling y
Vulkan pipelines con flags de captura; estos tiempos no sustituyen una medición
normal del runtime.

Antes de cada proceso se exige reposo observado y temperatura aceptable. Se
invierte el orden de backends entre rondas y se alternan las variantes dentro de
cada caso. La telemetría y todas las muestras se conservan; esto reduce sesgos,
pero no garantiza clocks constantes ni equilibrio térmico durante cada llamada.

Todos los casos cumplen `rtol=5e-5, atol=5e-5` frente a NumPy float64. En esta
suite, incluyendo los modos matemáticos experimentales, el máximo error relativo
L2 observado es **8,113×10⁻⁷** y el máximo absoluto **4,141×10⁻⁵**. No son cotas
para matrices arbitrarias ni una promesa de resultados iguales entre backends.

### OpenCL: promoción accidental a doble precisión

Para `(2049,2047) @ (2047,2051)`:

| Variante | Medianas de pared de las dos rondas | Conversiones float64 en PTX | Branches en PTX |
| --- | ---: | ---: | ---: |
| Literal histórico `0.0` | 6,322–6,323 ms | 28 | 38 |
| Literal float32 `0.0f` | 3,907–3,943 ms | 0 | 38 |
| `0.0f` y grupo fijo declarado al compilador | 3,907–3,918 ms | 0 | 38 |

Los conteos son estáticos, no instrucciones ejecutadas por elemento. Las guardas,
tiles, entradas, salidas y flags son iguales en la primera pareja. El PTX anterior
incluía `cvt.f64.f32` y `cvt.rn.f32.f64` alrededor de las cargas; el reenlace SASS
muestra 14 `F2F.F64.F32` y 14 `F2F.F32.F64`, frente a cero tras corregir el literal.
La cantidad de branches no disminuye con este cambio.

Los resultados de ambas variantes son numéricamente iguales en las mediciones.
No se cambia el dtype público, el orden del producto ni los flags matemáticos.
En tamaños ya alineados, donde OpenCL eliminaba esas cargas condicionales, no
aparece una ganancia consistente. Declarar el tamaño de grupo tampoco aporta una
mejora adicional general: no se promueve esa variante.

El fix está en `src/openpacho/_gemm.py`, compartido por los tres backends. En CUDA
el compilador ya eliminaba las conversiones; en Vulkan los dos literales producen
exactamente el mismo SPIR-V. No atribuimos ganancias allí al sufijo.

### Contracción, guardas y diferencias entre backends

CUDA mantiene `fmad=false`; el PTX/SASS reenlazado del bucle contiene multiplicación
y suma separadas. La variante experimental `fmad=true` sustituye 32 `FMUL` y 32
`FADD` estáticos por 32 `FFMA`. El mismo caso irregular pasa de 3,487–3,629 ms con
literales float32 a 3,324–3,326 ms con contracción. Los resultados **cambian de
redondeo**. Esta opción queda exclusivamente en el diagnóstico: FMA no equivale
a activar todo fast-math, y estos ejemplos no establecen garantías universales
de error. Véanse las [opciones de NVRTC](https://docs.nvidia.com/cuda/nvrtc/).

OpenCL genera FMA en el control normal. Al prohibir contracción se ve el cambio
correspondiente en PTX; la diferencia de rendimiento es pequeña en este GEMM.
En Vulkan, `precise` añade 38 decoraciones `NoContraction`; quitando esas
decoraciones recuperamos exactamente el SPIR-V del control. La variante sin
contracción cuesta 3,047–3,240 ms frente a 2,704–2,741 ms del control con literales
float32, pero sin ISA Vulkan no afirmamos haber contado sus FMA nativas.

Hay un control negativo importante: **baseline y `float_literals` Vulkan tienen
SPIR-V idéntico en los seis pares**. Aun así, sus medianas irregulares difieren
aproximadamente 6–7 %. No interpretamos esa diferencia como una optimización del
shader; evidencia que temporización, ejecución y ubicación de pipelines también
pueden introducir diferencias que este experimento no separa.

Quitar guardas de tamaños alineados reduce branches intermedias CUDA/Vulkan,
pero no garantiza una mejora. En Vulkan sube el dato de registros del driver de
60 a 73. La ganancia pequeña en 1024³ no se reproduce en el tamaño mayor alineado.
No extrapolamos ocupación efectiva solo de ese contador. Los compiladores pueden
usar predicación y realizar transformaciones posteriores; un `if` de la fuente
no corresponde necesariamente a una rama divergente de la ISA.

Por tanto, la promoción float64 explica una parte grande y demostrable del gap
OpenCL. La política de contracción explica otra diferencia de aritmética y parte
del tiempo. Quedan sin aislar suficientemente planificación de instrucciones,
desenrollado, acceso a memoria y código nativo final de Vulkan. No hay evidencia
de que la diferencia restante sea un muro físico de OpenCL, CUDA o Vulkan.

## Revisión de la implementación de copias

Los kernels siguen copiando palabras uint32 sobre almacenamiento float32. Esto
preserva subnormales, ceros con signo y payloads de NaN, y escribe `0x00000000` en
el relleno. Se verifican dimensiones, límites, almacenamiento y metadatos en
cada llamada. Las formas de lote y el recorte se mantienen dentro de los buffers.

Cambios seleccionados:

1. Caché LRU de 128 entradas para geometría validada y para el cuerpo de copia.
   Solo retiene tuplas de enteros y cadenas; ni arrays, ni buffers, ni objetos
   retenidos indirectamente por `dtype.metadata`. Los valores y metadatos actuales
   se comprueban incluso cuando la geometría ya está en caché.
2. El compilador cuenta los envíos físicos conocidos del GEMM con padding:
   hasta dos copias, GEMM y recorte. Así CUDA puede amortizar el contexto de ese
   paso, sin alterar kernels, tolerancias, datos ni API. Un producto alineado
   conserva el criterio de un solo envío. La ruta eager no cambia su ámbito.

El microbenchmark de preparación host, nueve rondas de 100.000 llamadas, da:
`_body` 1,246 → 0,077 µs; `_validate` 2,321 → 2,246 µs. El segundo ahorro es
marginal. No se presentan estos tiempos como coste GPU ni como ganancia completa.

El benchmark público usa el wheel anterior **solo para cargar `_padding_copy.py`**;
todos los modos ejecutan el GEMM corregido y los backends actuales. `old_copy` y
`cached_copy` aíslan la preparación; `cached_copy` y `current` aíslan el ámbito
CUDA. En OpenCL/Vulkan estos últimos son controles equivalentes, útiles para
observar ruido. `unpadded` es una referencia distinta, no una ablación de caché.
Son **144 configuraciones y 4.464 muestras**, con seis formas, dos rondas y 31
muestras por modo. Incluyen asignaciones, relleno, GEMM, recorte y sincronización,
con entradas residentes. Las cuatro salidas de cada uno de los 36 grupos son
idénticas bit a bit; todas cumplen tolerancias contra una referencia float64.
El máximo error relativo L2 de esta segunda suite es **8,106×10⁻⁷**, y el absoluto
**4,093×10⁻⁵**. Las pruebas de valores especiales se ejecutan aparte del benchmark
de datos aleatorios finitos.

Resultados representativos, medianas de cada ronda:

| Caso | Antes → actual, ronda 1 | Antes → actual, ronda 2 | Lectura |
| --- | ---: | ---: | --- |
| Vulkan 33×17×65 | 222,1 → 214,1 µs | 222,7 → 211,7 µs | Ahorro pequeño de preparación |
| Vulkan 1025×1023×1027 | 641,1 → 623,8 µs | 631,9 → 619,6 µs | Ahorro pequeño, repetido |
| CUDA 33×17×65 | 214,0 → 200,1 µs | 414,6 → 410,6 µs | Menos coste, gran variación entre rondas |
| CUDA 2049×2047×2051 | 3770,3 → 3751,7 µs | 3672,7 → 3695,3 µs | Mixto; no mejora demostrada |

El diagnóstico CUDA reduce 12 pares push/pop a 5, conservando cuatro lanzamientos
en ese alcance instrumentado. No afirma que el kernel se acelere. Los resultados
OpenCL de las cachés son pequeños o mixtos. No procede seguir priorizando ese
microajuste sobre GEMM, envíos, temporales o layouts con mayor coste eliminable.

Después de corregir los literales, OpenCL irregular grande **sin padding** cuesta
3,925–3,940 ms en el ejecutor preparado, frente a **4,110–4,112 ms con padding**.
Esta comparación incluye las copias. La ventaja anterior del padding no persiste
en ese caso con la ruta base corregida. La opción se conserva desactivada por
defecto; CUDA sigue siendo mixto y Vulkan varía entre rondas. Los resultados
históricos de [padding](padding-results.md) permanecen como snapshot anterior.

No se introduce vectorización `uint4` indiscriminada: el ancho impar cambia la
alineación de cada fila. Tampoco se elimina una guarda necesaria de copia. Un
siguiente experimento con más margen sería escribir la salida lógica desde GEMM
para evitar el buffer intermedio y el recorte, contrastando el efecto de stores
con stride irregular. Necesita su propio control y no está implementado aquí.

## Reproducir y continuar

```sh
python -m benchmarks.isolated --worker kernel_inspection --backends opencl,vulkan,cuda-custom --rounds 2 --repeats 21 --max-wait-seconds 300 --max-temperature 52 --output .artifacts/inspection-new
python -m benchmarks.isolated --worker padding_copy_review --control-wheel <wheel-anterior.whl> --backends opencl,vulkan,cuda-custom --rounds 2 --repeats 31 --max-wait-seconds 300 --max-temperature 52 --output .artifacts/copy-review-new
```

Requieren las dependencias del backend y el compilador Vulkan configurado.
`PACHO_NVDISASM` es opcional para reenlazar/desensamblar PTX CUDA, sin cambiar el
runtime. La captura Vulkan exige la extensión de introspección; su ausencia
produce un error explícito. Los controles históricos y nuevas capturas conservan
sus hashes propios. Los reportes iniciales denominan `exact_baseline_equality` a
igualdad numérica; el worker actual distingue igualdad numérica y bit a bit.

Pendientes: medir políticas matemáticas explícitas, variantes de acceso/desenrollado
con control de precisión, obtener evidencia de ISA Vulkan por herramientas del
fabricante cuando estén disponibles y validar otros dispositivos. `mean` con
suma rellenada y divisor lógico queda anotado como extensión futura en
[padding](padding.md#extensión-futura-reducciones), sin implementación en esta ronda.
Las pruebas y distribuciones verificadas se registran en [verificación](verification.md).
