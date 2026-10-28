# Verificación local de 0.1.0a1

## Etapa actual: contracción float64 — 2026-10-22

El [experimento FMA/separado](fp64-contraction-results.md) completa dos rondas
CUDA y dos OpenCL: 132 filas, 264 variantes, 1.296 muestras y 5.971 comprobaciones
de salida aceptadas. Las 44.316 evaluaciones de coordenadas incluyen repeticiones;
no representan observaciones independientes. Se verifican política de redondeo,
reproducción del control nativo, destinos envenenados, cachés y limpieza.

Los 33 archivos del runtime permanecen idénticos; CUDA custom conserva
`fmad=false` y no se incorpora un selector público. Las mejoras de los kernels
justifican una comparación futura del flujo de red densa, no una promoción
automática. Los informes distinguen precisión matemática de equivalencia,
telemetría ausente de la primera ronda y los intentos fallidos de reposo.

Pasan las 86 pruebas CPU dirigidas y Ruff check/formato en los seis archivos
Python nuevos. QA independiente: **APPROVE**, con las 22 entradas reconstruidas,
2.191 puntos de referencia recalculados mediante redondeo entero y sumas exactas,
y reconciliación de artefactos, tablas, hashes y condiciones de reposo. No se
ejecuta la suite completa ni se reconstruye el wheel.

## Etapa anterior: osciladores acoplados — 2026-10-22

El [cliente velocity-Verlet](oscillators-results.md) reutiliza un paso compilado
durante un bucle sin observaciones host. Pasan 40 pruebas dirigidas: 21 CPU del
ejemplo, ocho nativas float32/64 y 11 del benchmark. Se cubren convergencia de
segundo orden, energía, reversibilidad, 65 GEMV por recorrido, rebinding e inputs
inmutables. No cambia el runtime ni se ejecuta la suite completa.

Catorce procesos completan 156 filas y 3.744 muestras, todas aceptadas. Los
12 procesos GPU pasan reposo observado. Se separan precisión de backend, error
temporal modal, energía y costes completos. En los tamaños medidos, NumPy ocho
hilos supera las rutas GPU; los planes mejoran OpenPacho frente a eager. No se
infiere un cuello de botella exacto a partir de tiempos de pared.
QA independiente: **APPROVE**, con 32 pruebas CPU reejecutadas y verificación de
156 celdas, referencias físicas, hashes y 12 puertas de reposo; runtime intacto.

## Etapa anterior: experimento CUDA GEMM float64 — 2026-10-22

El [diagnóstico y candidato](cuda-fp64-gemm-results.md) no modifican el runtime.
GEMM concentra 91.2–95.3% de los eventos GPU registrados en cuatro planes; el
tile compartido conserva los resultados, pero no mejora consistentemente los
flujos. Se decide no promoverlo. Dos rondas completan 60 casos, 8.640 muestras y
9.360 comprobaciones aceptadas. La validación adicional cubre 54 lanzamientos
con destinos envenenados y nueve rutas de vectores/vacíos/K=0.

Pasan 26 pruebas CPU dirigidas del generador y del protocolo. Los casos numéricos
exactos separan equivalencia de precisión. No se ejecuta la suite completa.
QA independiente: **APPROVE**, con las 26 pruebas CPU reejecutadas, reconciliación
de tablas/muestras, 23 controles reproducidos y 172 puntos exactos comprobados.
Los 33 archivos del runtime permanecen idénticos al estado anterior.

## Etapa anterior: red densa como cliente — 2026-10-22

El [ejemplo de red densa](dense-network-results.md) añade inferencia y un paso SGD
con gradientes explícitos, sin cambios de runtime. Pasan 45 pruebas dirigidas:
24 del ejemplo en CPU, ocho nativas y 13 del protocolo de benchmark. Se verifica
la derivada de cada parámetro mediante diferencias finitas y el rebinding compilado.

Dos rondas comparan NumPy 1/8 hilos, OpenCL custom/CLBlast, Vulkan, CUDA
custom/cuBLAS y CuPy. Se conservan 288 filas aceptadas y 6.912 muestras
residentes/completas, 24 filas Vulkan float64 no soportadas y los 24 fallos
iniciales de caché CuPy. Dos procesos CuPy adicionales con caché privada completan
su comparación. Las limitaciones, variación y costes de preparación/transferencia
están separados. No se ejecuta la suite completa ni se mide el pico de memoria.
QA independiente: **APPROVE**, con 37 pruebas CPU reejecutadas y reconciliación
de las 140 celdas de resultados, 23.328 diagnósticos y las 14 puertas GPU.

## Etapa anterior: preparación de extremos — 2026-10-22

La [comparación de planes](extrema-planning-results.md) registra seis procesos,
66 grupos de casos y 11.880 muestras con dos controles independientes, dos rondas
y reposo observado. Todas las salidas conservan los bits del control y cumplen la
referencia. Se promueve min/max float32 materializado en OpenCL/CUDA; las regresiones
Vulkan impiden promoverlo allí. No se generalizan ganancias pequeñas o dtypes no medidos.

Pasan 106 pruebas dirigidas y se omiten cuatro por enteros CUDA no soportados.
Se cubren extremos, fronteras de productores, rebinding, retención de arrays,
extensiones de lowerer y el protocolo del benchmark; no se ejecuta la suite global.
La comprobación posterior a la promoción conserva 1.188 muestras adicionales con
controles forzados. QA independiente: **APPROVE**, incluidas 45 pruebas CPU y
reconciliación de las mediciones, tablas y hashes.

## Etapa anterior: extremos residentes — 2026-10-22

La [iteración de min/max](extrema-results.md) añade extremos por ejes en todos
los backends y en compile, con dtype preservado y reglas de NaN/cero explícitas.
Softmax/logsumexp permanecen como ejemplos externos. Pasan 267 pruebas dirigidas;
cuatro se omiten por los enteros no admitidos en CUDA. QA independiente aprobado.
No se ejecutó la suite
global ni se afirma una aceleración sin medición.

## Etapa anterior: investigación de acumulación GEMM — 2026-10-22

La [comparación de acumuladores](gemm-accumulation-results.md) añade infraestructura
experimental, sin cambiar el runtime: seis procesos aislados, 78 grupos de casos,
468 resultados de variante y 4.320 muestras residentes. Neumaier pasa los casos
de cancelación severa con coste alto; las variantes por bloques mejoran los casos
aleatorios, pero conservan pérdidas por cancelación. No se promueve un candidato.
54 pruebas CPU y Ruff pasan; QA independiente aprobado. Ver protocolo, límites
y evidencia en el informe.

## Etapa anterior: selección residente — 2026-10-22

La [iteración de where](selection-results.md) añade selección con broadcasting,
promoción de escalares NumPy 2 y fusión/map-reduce en todos los backends.
44 pruebas nuevas y 53 regresiones dirigidas pasan; una prueba de int64 se omite
en CUDA por su alcance explícito. QA independiente aprobado. No se ejecutó
la suite global ni se afirma una aceleración sin medición.

## Etapa anterior: normas estables CUDA — 2026-10-22

La [iteración de normas](cuda-norm-results.md) comparte el algoritmo escalado
con CUDA, en eager y compile, y permite ejecutar los solvers externos sin
cambiar sus recurrencias ni tolerancias.

- **257 pruebas dirigidas pasan**, con OpenCL/Vulkan/CUDA obligatorios;
  51 avisos PyOpenCL. Ruff check y formato pasan en los ocho archivos Python
  modificados. No se ejecutó la suite global ni se reconstruyó el wheel.
- **52 diagnósticos Decimal80** de normas/solvers entre CUDA custom y cuBLAS,
  sin excepciones; seis alertas hacia la solución en sistemas de Hilbert se
  conservan. Las normas extremas y los sistemas ordinarios pasan.
- Windows, Python 3.12.14, RTX 5060 Ti, driver 617.14. Los tests integrados usan
  `--capture=sys`; con captura de descriptores se reprodujo una excepción nativa
  NVRTC `0xc0000008`. Vulkan usa un TEMP accesible y compilador GLSL explícito.

Artefactos en `.artifacts/iterations-2026-10-22`; detalle de alcance y límites en
el informe enlazado. Las etapas siguientes son snapshots históricos.

## Etapa anterior: tres iteraciones — 2026-09-24

La [ronda de septiembre](three-iterations-results.md) incorpora GEMM rectangular custom
sin copias, DOT/GEMV batched cuBLAS seleccionados según medición y flags de
transposición privada compilada con CLBlast. Vulkan conserva su kernel después
de dos confirmaciones aisladas desfavorables para los nuevos tiles.

- **1.652 pruebas pasan, una omitida**, ejecutando `tests` y `benchmarks` con
  OpenCL/Vulkan/CUDA obligatorios y CLBlast nativo; 277 avisos PyOpenCL. La omitida
  exige un segundo dispositivo OpenCL físico.
- **38.880 muestras públicas** entre candidatos y selección final, más 2.352
  muestras de confirmación Vulkan con salida preasignada. Doce procesos
  secuenciales, sin pruebas GPU de subagentes; puertas de reposo observadas,
  de 10,5–229,2 s, sin relajar umbrales. Dos controles por caso, orden equilibrado.
- Los 84 grupos GEMM rectangular conservan bits; siete siguen fuera de tolerancia
  igualmente en controles y variante. Un grupo Vulkan también conserva una
  violación idéntica en todos sus modos. CLBlast-layouts y cuBLAS vectorial pasan
  sus tolerancias. Finalización, rendimiento y validez numérica siguen separados.
- Wheel desde sdist: **525 pasan, 909 omitidas** en entorno CPU sin GPU bindings.
  Importación desde el wheel, smoke con imports opcionales bloqueados, extras
  independientes, NumPy como única dependencia base y fuentes del wheel idénticas.
- Ruff check/formato globales y `pip check` pasan. El workflow CPU añade los
  tests de infraestructura de benchmarks; ejecución remota aún pendiente.

Artefactos en `.artifacts/next-three`, con informes completos, muestras,
telemetría, controles históricos y hashes. Wheel SHA256:
`83b7c21a716da655aaf4ed7f6b66af703ba259e5a9213924937cec31da94a270`.
La versión sigue siendo `0.1.0a1`; no se publica. Ver los
[criterios para la primera estable](release-readiness.md).

## Etapa anterior: reevaluación de padding — 2026-09-19

La [reevaluación](padding-reassessment.md) conserva padding explícito y
desactivado por defecto: aporta ganancias en algunas formas rectangulares,
pero empeora la mayoría. No se modifica el runtime ni su política numérica.

- **30.960 muestras públicas**, 172 grupos forma/backend/semilla, 22 formas
  distintas y trece procesos completos. OpenCL/Vulkan/CUDA en la suite principal;
  OpenCL/CUDA en el seguimiento de tamaños cercanos. Reposo observado antes de
  cada proceso, con esperas de 10,4–62,2 segundos; sin procesos GPU superpuestos.
- Padding activo/inactivo conserva resultados idénticos bit a bit en los
  172 grupos. Tres grupos CUDA de la suite principal superan la tolerancia
  por un elemento, igualmente en los dos controles y con padding. Se conservan
  los fallos y la tentativa incompleta inicial; completar el benchmark no se
  presenta como aprobar precisión. OpenCL/Vulkan y el seguimiento pasan el umbral.
- **173 pruebas de benchmarks pasan**. Las nuevas cubren balance de órdenes,
  planes independientes, transferencias del recorrido completo, persistencia
  de fallos, reenlace, validación numérica y CLI. Se corrigen literales float32
  del generador experimental de variantes, sin modificar kernels del paquete.
- Ruff check y formato de `src`, `tests`, `benchmarks` y `examples` pasan;
  sintaxis Python 3.10 comprobada en **128 archivos Python**.
- Los **29 archivos Python del runtime** coinciden byte a byte con el wheel
  de la etapa anterior, y sus hashes coinciden en los trece procesos medidos.
  No se repite la suite completa del runtime; su última ejecución fue la de
  1.341 pruebas aprobadas documentada debajo.

Artefactos en `.artifacts/padding-decision`, con muestras, errores, telemetría,
fuentes de los workers, selección de informes y controles de procedencia.
El benchmark reproducible está en `benchmarks.padding_decision`. El sdist de esta
ronda incluye el benchmark y documentación actualizados; no hay nueva publicación.
Se conserva el alcance local: Windows, RTX 5060 Ti, driver 616.64.

## Etapa anterior: inspección de kernels y copias — 2026-09-19

La [inspección](kernel-inspection-results.md) corrige la promoción accidental a
float64 en cargas de GEMM float32, cachea preparación de copias sin retener
entradas y cuenta los envíos del padding para seleccionar el ámbito CUDA.
La política matemática predeterminada no cambia.

- Runtime con OpenCL/Vulkan/CUDA obligatorios y CLBlast nativo: **1.341 pasan,
  una omitida** por necesitar otro dispositivo OpenCL; 273 avisos PyOpenCL.
  La selección inicial de pruebas afectadas dio **115 pasan**.
- **141 pruebas de benchmarks pasan**, incluidas captura Vulkan/ABI, parser PTX,
  controles de fuentes, aislamiento del wheel y restauración tras errores.
- Ruff y formato pasan; sintaxis Python 3.10 comprobada en **126 archivos Python**.
  Ejecución real en Python 3.12 / Windows / RTX 5060 Ti / driver 616.64.
- Diagnóstico de compiladores: **72 configuraciones, 1.512 muestras** de pared y
  360 diagnósticos GPU separados. Revisión de copias: **144 configuraciones,
  4.464 muestras públicas** con todas las copias incluidas; 36 grupos de resultados
  idénticos bit a bit. Dos rondas por suite y reposo observado sin relajar controles.
  Cada informe conserva errores numéricos y alcance de sus relojes.
- Wheel construido desde sdist: **449 pasan, 893 omitidas** en entorno CPU,
  importando el wheel y comprobando que no carga dependencias GPU/ejemplos.
  NumPy continúa como única dependencia base. Runtime del wheel idéntico a fuentes;
  documentación del sdist y enlaces locales comprobados.

Artefactos en `.artifacts/kernel-inspection`. SHA256 del wheel:
`a577d80ffde9c8be17ab1c2ca9c51278aaf34412e6be84bd0ecd24450fa66e3b`.
Las herramientas de inspección no forman parte del paquete instalable. No se
obtuvo ISA Vulkan: el driver no expuso representaciones internas. No se afirma
validación en otros fabricantes/SO ni ejecución de CI remota. Versión 0.1.0a1,
sin publicación. `mean` con padding queda documentado como trabajo futuro.

## Etapa anterior: padding opcional — 2026-09-18

Se añade [padding opcional](padding.md) a `compile`/su decorador y `matmul`,
desactivado por defecto. Los ejecutores preparados heredan la política. GEMM
custom float32 de OpenCL, Vulkan y CUDA puede usar buffers alineados con sus
tiles; los demás casos conservan su ruta y proveedor.

- Runtime con OpenCL/Vulkan/CUDA obligatorios y CLBlast nativo: **1.323 pasan,
  una omitida** por requerir otro dispositivo OpenCL; 273 avisos de compilación
  PyOpenCL. Se usa `PYOPENCL_NO_CACHE=1` y el compilador Vulkan local.
- **96 pruebas de benchmarks pasan**. Las regresiones nuevas cubren el contrato
  del flag, elección por capacidad, rechazo de BLAS, reenlace/vida de entradas,
  resultados independientes, límites de memoria/índices, lotes broadcast,
  dimensiones lógicas antes de reducciones y copias bit a bit de subnormales,
  ceros con signo, NaN e infinitos. Los errores de preparación se propagan.
- [Benchmark de padding](padding-results.md): seis procesos, dos rondas,
  72 configuraciones, **6.048 muestras públicas** y 756 diagnósticas con entradas
  preempaquetadas. Todos los controles de reposo se cumplieron sin relajarlos;
  pausas efectivas de 10,5–156,4 segundos. Las 36 parejas producen resultados
  exactamente iguales entre padding activo/inactivo dentro del mismo backend.
  Las pruebas conservan errores contra una referencia float64, no solo tiempos.
- Ruff y formato pasan. Sintaxis Python 3.10 comprobada en **120 archivos Python**;
  ejecución real en Python 3.12, Windows, RTX 5060 Ti y driver 616.64.
- Wheel construido desde sdist: **431 pasan, 893 omitidas** en el entorno CPU
  sin runtimes GPU, importando desde el wheel. Se verifica que NumPy sigue siendo
  la única dependencia base y que los extras GPU/ejemplos no se importan al usar
  las operaciones CPU. El runtime del wheel coincide byte a byte con los fuentes.
- Documentación del sdist y enlaces locales contrastados con el workspace.

Artefactos en `.artifacts/padding`: muestras, telemetría, errores, metadatos de
planes, logs/XML de tests, scripts y distribuciones. SHA256 del wheel:
`77f55a8f31d4bd915be9cc763be7f6143fbee3b7937f53815d453da82cd9a2f9`.
Las cifras no validan otros fabricantes/SO ni CI remota. Versión 0.1.0a1,
sin publicación.

## Etapa anterior: binding, contexto y GEMV CUDA — 2026-09-18

La [ronda de tres optimizaciones](three-actions-results.md) incorpora binding
posicional común, activación CUDA anidada con scope selectivo y GEMV/DOT custom
cooperativos. No modifica algoritmos de dominio, extras ni flags matemáticos.

- Runtime con OpenCL/Vulkan/CUDA obligatorios y CLBlast nativo: **1.250 pasan,
  una omitida** por requerir un segundo dispositivo OpenCL; 264 avisos de compilación
  PyOpenCL. Se configura `PYOPENCL_NO_CACHE=1` y el compilador Vulkan local.
- **89 pruebas de benchmarks pasan**, incluidas selección de controles, planes
  separados por variante y limpieza de backends sin `close()` o con fallo al limpiar.
- Las nuevas regresiones cubren binding/errores/defaults/estáticos y vida de inputs,
  contexto ajeno, anidación, excepciones, concurrencia, fronteras 255/256/257,
  matrices rectangulares y lotes, DOT largo, cancelación y valores no finitos.
  Se conserva el input 129×8193 donde fallaba la tolerancia del kernel serial anterior.
- **660 configuraciones principales, 492 comparaciones controladas**, 31 muestras
  por alcance y ocho procesos válidos. Los 168 casos con selección final pasan las
  tolerancias registradas; los fallos numéricos de controles anteriores permanecen
  visibles. Reposo observado entre procesos, con un timeout conservado sin saltar el gate.
- Ruff y formato pasan. Sintaxis Python 3.10 comprobada en 113 archivos Python;
  ejecución real en Python 3.12 / Windows / RTX 5060 Ti, driver 616.64.
- Wheel construido desde sdist: **391 pasan, 860 omitidas** en entorno CPU,
  importando el runtime desde el wheel y comprobando la frontera sin imports
  opcionales GPU/ejemplos. NumPy sigue siendo la única dependencia base.
- Runtime del wheel idéntico a los fuentes; documentación del sdist contrastada
  contra el workspace y enlaces locales comprobados. El seguimiento adicional
  de ridge no pudo medir por timeout de reposo de 300 segundos; conserva telemetría
  y no reemplaza los resultados mixtos de las tandas válidas.

Artefactos en `.artifacts/three-actions`: resultados por variante, muestras,
telemetría, errores, diagnóstico, logs/XML de tests y scripts de reproducción.
Los ensayos exploratorios y fallidos se conservan separados de la selección final.
Otros fabricantes/SO y CI remota siguen pendientes. Versión 0.1.0a1, sin publicación.

## Etapa anterior: argumentos de lanzamiento CUDA — 2026-09-13

La [ronda de despacho CUDA](cuda-dispatch-results.md) reutiliza almacenamiento
host por kernel sin modificar shaders, planes, algoritmos, contextos ni política
numérica. Cada llamada vuelve a enlazar valores bajo el lock existente.

- Suite de runtime con OpenCL/Vulkan/CUDA obligatorios y CLBlast nativo:
  **1.200 pasan, una omitida** por necesitar dos dispositivos OpenCL; 264 avisos
  de compilación PyOpenCL. `PYOPENCL_NO_CACHE=1` evita la caché persistente no
  escribible. Las pruebas de infraestructura de benchmarks añaden **80 que pasan**.
- Cinco controles CPU nuevos cubren tamaños/bytes ABI, signo del cero, reenlace
  sin retención de inputs, tipos/cantidad incorrectos, lock, lanzamiento fallido
  y reintento, y liberación de argumentos al expulsar un módulo completado.
  Una regresión GPU comprueba dos fills bool pendientes con valores distintos.
- La prueba inicial encontró una incompatibilidad `np.bool_`/`ctypes.c_ubyte`;
  la implementación final usa `c_bool`, con ABI de un byte verificada. La primera
  tanda amplia de RNG tuvo fallos OpenCL por omitir la configuración de caché y
  omisiones Vulkan por falta de ruta al compilador; la suite completa final sí
  configura ambos y pasa. Los logs iniciales se conservan.
- **236 configuraciones principales**, 31 muestras por alcance: 144 CUDA y 92
  referencias NumPy/CuPy, en ocho procesos secuenciales válidos y dos rondas.
  La primera tentativa falló en el runner antes de medir; no se usa como resultado.
- El seguimiento de ridge cuBLAS f32 añade **seis configuraciones**, tres bloques
  de 51 pares por alcance. No reproduce la pérdida completa inicial; se conservan
  ambos conjuntos. Errores numéricos y resultados exactos del control quedan
  registrados. El ABI bool se corrigió después de los controles exclusivamente
  flotantes; no cambia sus rutas. El seguimiento ya usa la corrección final.
- Wheel construido desde sdist: **374 pasan, 827 omitidas** en entorno CPU,
  importando desde el wheel y bloqueando imports GPU/ejemplos para verificar
  independencia. No cambian extras ni dependencias base.
- Ruff/formato, sintaxis Python 3.10 y enlaces locales comprobados. Runtime real
  en Python 3.12 / Windows / RTX 5060 Ti. Otros dispositivos/SO y CI remota pendientes.

Artefactos en `.artifacts/cuda-dispatch`: tests.xml/log, benchmark-tests.log,
wheel-tests.log, summary.csv, paired-gains.csv, perfiles, JSON y telemetría.
La versión sigue siendo 0.1.0a1, sin publicación.

## Etapa anterior: fusión CUDA — 2026-09-13

La [ronda de fusión CUDA](cuda-fusion-results.md) implementa lowering elementwise
y map-reduce mediante la factoría del backend. Conserva las primitivas, la
política numérica y las funciones de dominio de los comparadores.

- Suite completa con OpenCL/Vulkan/CUDA obligatorios y CLBlast nativo:
  **1.194 pasan, una omitida** por requerir dos dispositivos OpenCL; 264 avisos
  de compilación PyOpenCL. Se usa `PYOPENCL_NO_CACHE=1`.
- **127 pruebas CUDA enfocadas** pasaron antes de la suite completa. Cubren
  ambos proveedores, colas/ejes/entradas vacías, broadcasting, reenlace escalar,
  salidas compartidas, vidas de buffers, concurrencia, límites de parciales,
  promoción/estrechamiento, no FMA oculta, cancelación y subnormales.
- El ejemplo `examples/compiled_numpy.py --backend cuda` valida energía por
  filas y muestra un plan map-reduce con caché reutilizada.
- **362 configuraciones medidas**, 21 muestras por alcance: 144 de control CUDA
  y 218 de referencias. Cuatro procesos CUDA y once de referencias completaron;
  la segunda ronda NumPy a un hilo no se ejecutó por timeout del reposo. Todos
  los comparadores GPU tienen dos rondas. Muestras/errores/fallos se conservan.
- Un seguimiento de covarianza custom float32 añade **seis configuraciones**,
  31 pares alternados por alcance en tres bloques. No reprodujo la pérdida
  inicial; ambos conjuntos permanecen separados y la variabilidad se documenta.
- Wheel construido desde sdist: **369 pasan, 826 omitidas** en entorno CPU.
  Importación y cálculo/RNG desde el wheel funcionan bloqueando imports GPU y
  ejemplos. NumPy sigue como única dependencia base, con extras independientes.
- Ruff/formato y sintaxis Python 3.10 pasan; ejecución real en Python 3.12,
  Windows y RTX 5060 Ti. No se verificó segundo fabricante ni CI remota.

Artefactos: `.artifacts/cuda-fusion`, incluidos tests.xml/log, wheel-tests.log,
summary.csv, controles originales y seguimiento de covarianza. La versión sigue
siendo 0.1.0a1, sin publicación. Los informes anteriores conservan su snapshot.

## Etapa anterior: routing y finalización Vulkan — 2026-09-13

La [ronda de routing/finalización](routing-completion-results.md) separa adaptadores
RNG, enlaza factorías/ámbitos al preparar planes y adopta fence reutilizable Vulkan,
con corrección de reentrada de finalizadores en el pool.

- Suite final con OpenCL/Vulkan/CUDA obligatorios y CLBlast nativo: **1.142 pasan,
  una omitida** por necesitar dos dispositivos OpenCL; 264 avisos de compilación
  PyOpenCL. `PYOPENCL_NO_CACHE=1` evita la caché persistente no escribible del entorno.
- El primer intento completo descubrió un contador de tests que no aceptaba
  submits vacíos. Se corrigió para respetar count y verificar un único marcador
  al leer; 136 pruebas enfocadas y después la suite completa final pasan.
- Regresiones: factorías independientes del nombre, errores sin fallback,
  ámbitos enlazados y cleanup, escalares RNG reenlazados, secuencias/contadores,
  pool reentrante, recursos pendientes y fallos de finalización sin reciclaje.
- Vulkan: **144 configuraciones** en tres procesos válidos, 31 muestras por modo;
  diagnóstico separado, tiempos completos, error y convergencia. La primera tanda
  incompleta por el bug del pool se excluye y se conserva como evidencia.
- RNG: **264 configuraciones válidas y 12 exclusiones por capacidad** en seis
  procesos, 31 muestras; igualdad exacta antes/después. Dos controles intercalados
  adicionales investigan latencias pequeñas OpenCL/CUDA. Control CPU de preparación
  separado; no equivale a tiempo de dispatch GPU.
- Wheel construido desde sdist: **369 pasan, 774 omitidas** en entorno CPU.
  Se verifica importación desde el wheel y cálculo/RNG bloqueando importaciones
  GPU/ejemplos. NumPy sigue como única dependencia base y los extras son independientes.
- Ruff/formato, sintaxis Python 3.10 y enlaces locales pasan. Runtime real en
  Python 3.12 / Windows / RTX 5060 Ti. No se verificó segundo fabricante ni CI remota.

Artefactos: `.artifacts/routing-completion`, incluidos tests-final.xml/log,
wheel-tests.log, CSV con todas las mediciones y JSON originales/derivados. La
corrección de metadatos de referencias idle/download conserva hash y original;
no modifica tiempos ni errores. La versión continúa 0.1.0a1, sin publicación.

## Etapa anterior: staging de lectura y RNG — 2026-09-13

La [ronda de staging/RNG](staging-random-results.md) conserva mapeos Vulkan de
readback, manteniendo las subidas anteriores. Añade Philox residente con una API
acotada y capacidades por distribución/dtype.

- Suite completa con OpenCL/Vulkan/CUDA obligatorios y CLBlast nativo:
  **1.202 pasan, una omitida** por necesitar dos dispositivos OpenCL, 264 avisos
  de compilación PyOpenCL. Esta ejecución precede a acotar persistencia a readback.
- Tras esa selección y el ajuste de parámetros RNG: **72 pruebas enfocadas**
  pasan (staging y RNG en los cuatro backends) y **198 pruebas Vulkan** pasan.
  Se cubren mapeos, pool, comandos, recursos pendientes, fusión y operaciones.
  La regresión CPU de escalas negativas que redondean a cero y enteros enormes
  pasa también; no consume contadores inválidos.
- **30 pruebas CPU** de métricas/validación RNG/runner pasan después de corregir
  la etiqueta de precisión de referencias float32 del mismo dtype.
- **64 configuraciones** finales comparan readback con métodos del wheel anterior,
  31 muestras en dos rondas. Otras 64 midieron la variante de persistencia general
  descartada como predeterminada. No cambian los kernels matriciales/solvers.
- RNG: **228 configuraciones válidas, 12 exclusiones por capacidad** en diez
  procesos; otro control Vulkan final añade 36 válidas y 12 exclusiones. Se
  conservan error, momentos, muestras y reposo observado (hasta 31,1 s).
- Wheel desde sdist: **354 pasan, 772 omitidas** en entorno CPU. Importación
  verificada dentro del wheel y cálculo/RNG con módulos GPU/ejemplos bloqueados.
  Extras separados, NumPy como única dependencia base; ejemplos/benchmarks fuera
  del wheel. No cambia la versión ni se publica.
- Ruff/formato, sintaxis Python 3.10 y enlaces locales verificados. Ejecución real
  en Python 3.12 / Windows / RTX 5060 Ti. Sin segundo fabricante ni CI remota.

Artefactos en `.artifacts/staging-random`. Los JSON derivados `corrected-random`
rectifican una etiqueta de referencia de igual precisión y conservan hash/original;
no alteran tiempos ni errores numéricos. La primera suite sin configurar CLBlast
queda como log histórico; la ejecución requerida posterior sí lo incluye.

## Etapa anterior: estabilidad y guardas GEMM — 2026-09-13

La [ronda de kernels](kernel-stability-results.md) separa finalización, reposo
observado y calentamiento; compara diez variantes y promueve solo la
especialización de guardas OpenCL en tiles completos.

- **1.122 pruebas pasan, una omitida**, ejecutando `tests` y `benchmarks`, con
  OpenCL/Vulkan/CUDA obligatorios y CLBlast disponible. La omisión requiere dos
  dispositivos OpenCL; se emitieron 256 avisos de compilación PyOpenCL.
- **300 configuraciones de kernels** (tres backends, dos rondas), 21 muestras,
  con precisión y centinelas de salida correctos. Reposo observado 10,4–16,6 s,
  ventanas térmicas aceptadas de 40–42 °C. No implica clocks constantes.
- **88 configuraciones públicas** de control GEMM OpenCL, misma instancia y
  contexto, sin errores y con métricas numéricas iguales entre variantes.
- **42 casos adversos OpenCL**, cero excepciones y las mismas nueve alertas
  anteriores. El resto de proveedores no se volvió a medir en precisión adversa;
  sus shaders predeterminados se conservaron.
- Wheel desde sdist: **327 pasan, 727 omitidas** en el entorno CPU. Importación
  comprobada dentro del wheel, extras independientes y ejemplos/solvers fuera
  del paquete. Único cambio del runtime: `_gemm.py`; los shaders guardados de los
  12 controles siguen siendo idénticos al wheel anterior.
- Pruebas CPU del protocolo: timeout, actividad/temperatura, pérdida de sensores,
  identidad, orden de procesos, fallo del worker, calentamiento y restauración
  del control ante excepciones. No se usan sus tiempos como benchmarks GPU.
  Tras el ajuste final de propagación del selector de telemetría al worker,
  **25 pruebas enfocadas** pasan; no se repite la suite GPU por ese cambio de CLI.
- Ruff y formato pasan (88 archivos). Sin cambios de dependencias o API pública;
  artefactos en `.artifacts/kernel-stability`. Otro hardware/SO y CI remota pendientes.

## Etapa anterior: ejecución compartida — 2026-09-13

Push descriptors Vulkan por capacidad y fallback a sets, caché de metadatos e
índices acotados OpenCL, tiles float32 compartidos y matmul por lotes CUDA.
Ver [resultados, controles y gaps](backend-sharing-results.md).

- Suite completa: **1.084 pasan, una omitida**, con los tres backends obligatorios
  y CLBlast disponible. La omisión requiere dos dispositivos OpenCL. Después del
  ajuste final de la ruta sin lotes y nueva cobertura, **75 pruebas CUDA pasan**.
  La ejecución completa anterior emitió 252 avisos del compilador PyOpenCL.
- Wheel final construido desde sdist: **326 pasan, 719 omitidas** en el entorno
  CPU, importando directamente del wheel sin reemplazar su instalación histórica.
  NumPy es la única dependencia base, los extras son independientes y ejemplos,
  benchmarks y solvers quedan fuera del wheel; el sdist conserva los ejemplos.
- Vulkan: dos controles de 60 configuraciones y un control adicional de cinco
  casos sobre el mismo VkDevice. Recursos en vuelo, fallback por límite, salidas
  y convergencia se validan. La mejora de tiempo es modesta/mixta.
- OpenCL/CUDA: dos rondas por proveedor con funciones anteriores/actuales sobre
  el mismo contexto; 21 muestras. Los lotes ausentes en el wheel histórico se
  declaran no soportados. El control detectó y permitió quitar una penalización
  inicial CUDA en las operaciones sin lotes.
- **144 configuraciones** en procesos con proveedor aislado y otras **144** con
  APIs intercaladas. Todas válidas, sin backends ausentes. La intercalación altera
  mucho algunos tiempos; no se usa como techo físico ni coste puro de abstracción.
- **104 configuraciones aplicadas irregulares**, siete muestras, sin excepciones
  ni ausencias. El primer intento omitió CuPy por su caché fuera del workspace;
  el final usa CUPY_CACHE_DIR local.
- **158 configuraciones adversas Decimal80**, cero excepciones y las mismas
  **40 alertas** de la ronda CUDA inicial; 52 casos CUDA norma/solver excluidos
  explícitamente. Las alertas numéricas siguen pendientes.
- AST de solvers idéntico al sdist anterior; las aplicaciones cambian solo el
  texto del alcance CUDA en metadatos. Ruff y formato pasan (81 archivos),
  sintaxis Python 3.10 analizada; ejecución real en Python 3.12.

La evidencia local corresponde a RTX 5060 Ti, Windows 11 y driver 616.64. Otro
hardware/SO y CI remota siguen pendientes. Artefactos, scripts de diagnóstico y
proveniencia en `.artifacts/backend-sharing`.

## Etapa anterior: frontera de comandos Vulkan — 2026-09-12

La tabla CFFI por dispositivo reduce conversiones en llamadas frecuentes sin
cambiar las recurrencias, tolerancias, frecuencia de comprobación, kernels de los
workloads medidos ni número de comandos. Incluye diagnóstico separado de tiempos
host/GPU y una corrección independiente de potencia float32 con bases negativas.
Ver [mediciones y límites](vulkan-dispatch-results.md).

- **1.045 pruebas pasan, una omitida**, con OpenCL/Vulkan/CUDA obligatorios y
  CLBlast nativo disponible. La omisión requiere dos dispositivos OpenCL. Incluye
  regresiones de ABI/VkResult, aislamiento por dispositivo, recursos en vuelo,
  potencia y limpieza de la instrumentación. Se emitieron 251 avisos de compilación
  de PyOpenCL, sin fallos de pruebas.
- **324 pasan, 677 omitidas** cargando el wheel directamente en el entorno CPU
  sin runtimes GPU. Se verificó el origen dentro del wheel, importación/cálculo
  con módulos GPU y ejemplos bloqueados, y extras independientes de CUDA.
- Wheel construido desde sdist: excluye ejemplos, benchmarks y solvers; el sdist
  conserva los ejemplos. NumPy sigue siendo la única dependencia base. No se
  añadió una dependencia binaria ni se reemplazó la instalación CPU histórica.
- **Dos controles de 60 configuraciones**, 21 muestras por modalidad: rutas
  wrapped/direct intercaladas, 30 casos float32/64 por proceso. Sin errores; mismas
  salidas, métricas numéricas, iteraciones, conteos de comandos y bytes copiados.
  Diagnóstico instrumentado fuera de las muestras cronometradas.
- **72 configuraciones aplicadas irregulares**, siete muestras: NumPy, OpenCL
  custom/CLBlast, Vulkan y CuPy. Sin errores ni backends ausentes. No sustituye el
  control pareado para atribuir causalmente una mejora al ejecutor.
- **42 configuraciones de precisión adversa Vulkan**, tres muestras: diez alertas,
  cero excepciones. El conjunto de alertas coincide con el subconjunto Vulkan de
  la ronda CUDA anterior. La salida 1 del diagnóstico señala esos hallazgos; no
  significa que hayan quedado resueltos los problemas de cancelación o subnormales.
- Ruff y formato pasan (78 archivos). Sintaxis Python 3.10 analizada; ejecución
  real en Python 3.12. Otro fabricante/SO y CI remota siguen pendientes.

Artefactos: `.artifacts/vulkan-dispatch`, con `final-1.json`, `final-2.json`,
`comparison.csv`, `accuracy.json`, `applications.json`, logs, distribuciones y
manifiesto. Las fuentes del runtime y runner coinciden con los hashes registrados
en ambos controles finales. No es todavía un ejecutor C/C++ ni una medición del
límite físico de Vulkan o PCIe.

## Etapa anterior: separación de aplicaciones — 2026-09-12

Los solvers y sistemas preparados se movieron a `examples/solvers.py`, fuera de
`src/openpacho` y de la API/rueda instalable. Se retiró la capacidad solver del
runtime CUDA. `get_backend(array)` permite que ejemplos y clientes recuperen
la instancia existente por API pública, sin copiar ni reconfigurar sus datos.
Ver [dirección acordada](architecture-evolution.md) y [ejemplos](../examples/README.md).

- **1.024 pruebas pasan, una omitida**, con OpenCL/Vulkan/CUDA obligatorios y
  CLBlast nativo disponible. La omisión sigue requiriendo dos dispositivos OpenCL.
- **316 pasan, 672 omitidas** al cargar el wheel desde un entorno CPU sin runtimes
  GPU; se comprobó la ruta de importación dentro del wheel. Se cargó directamente
  sin reemplazar la instalación histórica del entorno CPU.
- Wheel construido desde sdist: excluye `openpacho.solvers`, ejemplos y benchmarks.
  El sdist incluye los ejemplos. Metadatos: NumPy como única dependencia base;
  extras OpenCL/Vulkan independientes de CUDA. Importación y cálculo CPU con
  módulos GPU y `examples` bloqueados; no se descargaron nuevas dependencias.
- Comparación AST contra el wheel CUDA anterior: las trece funciones de cálculo,
  preparación y entrada conservan su estructura, salvo el acceso al backend por
  API pública. No se cambiaron recurrencias, tolerancias ni chequeos de convergencia.
- Ruff y formato pasan (73 archivos). Ejemplos CPU ejecutados. Tras cerrar la
  auditoría de documentación/metadatos, **28 pruebas enfocadas** adicionales pasan.
- La primera ejecución OpenCL encontró una caché SQLite no escribible fuera del
  workspace. Se repitió con `PYOPENCL_NO_CACHE=1`; es una configuración de pruebas,
  no una modificación del runtime ni una medición de rendimiento.
- Un test subprocess de wheel inicialmente cargó el paquete histórico del entorno;
  se corrigió para usar el mismo origen de paquete que su proceso padre.
- No se repitieron benchmarks de rendimiento ni precisión adversa en esta separación.
  Los resultados siguientes corresponden a sus snapshots; no se anuncian speedups nuevos.
  Otro hardware/SO y CI remota siguen pendientes.

Artefactos: `.artifacts/architecture-scope`, con logs, wheel/sdist y manifiesto de
fuentes. Los modos matemáticos, autotuning, layouts y ejecutor nativo descritos en
la dirección arquitectónica siguen siendo diseño pendiente.

## Etapa anterior: Vulkan por capacidad y CUDA inicial — 2026-09-12

La implementación CUDA y la nueva política de reutilización Vulkan se documentan
en [resultados actuales](cuda-initial-results.md) y [alcance CUDA](cuda.md).
OpenCL/Vulkan siguen siendo opciones independientes. CUDA aún no incorpora
fusión, normas estables, solvers ni matmul por lotes.

- **1.019 pruebas pasan, una omitida**, con OpenCL/Vulkan/CUDA obligatorios y
  CLBlast nativo. La omitida requiere dos dispositivos OpenCL.
- Wheel desde sdist instalado sin dependencias GPU: **311 pasan, 672 omitidas**.
  Importación desde site-packages; runtime CUDA probado además sin CuPy.
- **208 configuraciones aplicadas**, siete muestras por modo, cero errores o
  ausencias. CUDA diferencia plan sin fusión de lowering fusionado.
- **158 configuraciones de precisión adversa**, cero excepciones, 40 alertas
  numéricas. Se registran 52 casos norma/solver fuera del alcance CUDA inicial.
- Control Vulkan alternado en cuatro procesos: 1.432 reservas/liberaciones a
  cero, mismos dispatches/errores; aumenta la caché retenida de 1 a 3 MiB.
- Ruff y formato pasan. Sintaxis Python 3.10 analizada, runtime Python 3.12.14.
  Wheel/sdist: `.artifacts/dist-cuda-initial`. Otro hardware/SO y CI remota pendientes.

## Etapa anterior: solvers, aplicaciones y preparación CUDA — 2026-09-08

- **965 pasan, 1 omitida** en la suite completa de runtime y adaptadores,
  incluidas 144 pruebas de formas irregulares y cuatro regresiones del pool Vulkan. La omitida requiere dos dispositivos OpenCL.
- Wheel final construido desde sdist e instalado sin PyOpenCL/Vulkan: **294
  pasan, 635 omitidas** por backends opcionales ausentes. Se verificó la importación
  desde site-packages, fuera de `src`. Distribuciones en `.artifacts/dist-next-applications`.
- **63 configuraciones de solvers**, siete muestras por modalidad; cero errores
  de ejecución o backends ausentes. Datos, tolerancias e iteraciones conservados.
- **126 casos adversos OpenPacho GPU**, tres muestras: 28 indicadores, sin
  excepciones/ausencias. Son los mismos indicadores del subconjunto anterior.
- Lowering separado por backend, grupos Vulkan en un command buffer y planes
  de solver entre observaciones. CUDA todavía no es un backend disponible.
- **144 configuraciones aplicadas finales** de covarianza/ridge (72 alineadas
  y 72 irregulares), siete muestras por modo, sin errores ni backends ausentes;
  preparación y precisión visibles. El control del pool aísla su política con
  mediciones de reservas y errores numéricos.
- [Resultados actuales](solver-applications-results.md),
  [suite aplicada](application-measurement-plan.md) y [camino CUDA](cuda-backend-plan.md).

## Etapa anterior: Vulkan, GEMM y estabilidad — 2026-09-07

- **809 pasan, 1 omitida**, con OpenCL/Vulkan obligatorios y CLBlast nativo.
  La omitida requiere dos dispositivos OpenCL.
- Wheel construido desde sdist, instalación CPU sin backends GPU e importación
  desde site-packages: **249 pasan, 527 omitidas**.
- **319 configuraciones de rendimiento y 48 de preparación**, siete muestras
  por modalidad, sin errores de validación ni backends ausentes.
- **264 configuraciones de precisión**, tres muestras: 68 indicadores diagnósticos,
  cero excepciones. Incluye limitaciones numéricas de las referencias; no son
  68 bugs automáticamente ni se ocultan bajo la marca de ejecución correcta.
- Ruff y formato pasan. Distribuciones: `.artifacts/dist-numerics-vulkan`.
- [Resultados y gaps actuales](numerics-vulkan-results.md),
  [precisión](numerical-accuracy.md) e [investigación de CLBlast](clblast-techniques.md).
  AMD/Intel y CI remota siguen sin validación en esta sesión.

## Etapa anterior: map-reduce, preparación y BLAS batched — 2026-09-06

- **625 pasan, 1 omitida**, con OpenCL/Vulkan obligatorios y CLBlast nativo.
  La omitida necesita dos dispositivos OpenCL.
- Wheel desde sdist e instalado sin dependencias GPU: **208 pasan, 401 omitidas**.
- **319 configuraciones comparativas y 48 de preparación**: siete muestras por
  modo, sin errores de validación. Controles intercalados adicionales para tiling,
  batching y bindings de solvers; perfiles antes/después con origen de código.
- Ruff y formato pasan. Wheel/sdist: `.artifacts/dist-remaining-suite`.
- [Implementaciones, resultados mixtos y gaps restantes](remaining-suite-results.md).
  No se valida hardware AMD/Intel ni se resuelve todavía la norma para valores extremos.

## Etapa anterior: ejecución y staging Vulkan — 2026-09-06

- Suite CPU/OpenCL/Vulkan con backends GPU obligatorios: **530 pasan, 1 omitida**
  por requerir dos dispositivos OpenCL. Incluye 118 pruebas Vulkan.
- Wheel construido desde sdist e instalado sin dependencias GPU: **194 pasan,
  320 omitidas** por ausencia de los backends opcionales.
- Se repitieron **319 configuraciones**, siete rondas por modo, sin errores de
  validación. Se añadieron controles de transferencias Vulkan con once muestras.
- Ruff y formato pasan. Artefactos de distribución en `.artifacts/dist-vulkan-execution`.
- Ver [cambios, resultados y límites](vulkan-execution-results.md). Hardware/SO es
  un track secundario; la ruta implementada es portable y sigue validada solo en
  la GPU NVIDIA y Windows locales.

Las etapas siguientes conservan sus resultados históricos.

## Etapa anterior: gaps y optimización — 2026-09-06

- **518 pruebas pasan, 1 omitida**, con OpenCL y Vulkan obligatorios, CLBlast
  nativo y adaptadores de benchmark. La omitida requiere dos dispositivos OpenCL.
- **37 pruebas adicionales pasan** seleccionando CLBlast en la suite matricial:
  broadcasting por lotes, productos irregulares y residuo de CG.
- Wheel construido desde sdist e instalado sin dependencias GPU: **193 pasan,
  309 omitidas** por ausencia esperada de backends opcionales.
- Ruff y formato pasan. Wheel/sdist en `.artifacts/dist-gaps`.
- **319 configuraciones de rendimiento**, siete rondas por cada modo, sin
  errores de validación; además perfiles separados, eventos OpenCL, controles
  de transferencias y sensibilidad al orden de ejecución.
- CLBlast GEMV se prueba con destinos reciclados NaN/Inf; los solvers conservan
  su frecuencia de chequeo. Se reprodujo y documentó por separado el gap numérico
  preexistente de subdesbordamiento de normas.

Ver [gaps, resultados y límites](gaps-and-optimization.md). AMD/Intel y la matriz
de CI remota siguen pendientes. Las secciones siguientes son snapshots históricos.

## Primera implementación

Fecha: 2026-09-06. Windows 11, Python 3.12.14, NumPy 2.5.3,
PyOpenCL 2026.1.4. Dispositivo OpenCL: NVIDIA GeForce RTX 5060 Ti,
driver 616.64, precisión doble disponible.

| Verificación | Resultado |
| --- | --- |
| Suite completa con OpenCL obligatorio | **140 pruebas pasan; 1 omitida** porque requiere dos dispositivos OpenCL físicos. |
| Instalación limpia desde wheel, sin PyOpenCL | **73 pruebas pasan; 68 omitidas** por ausencia esperada de OpenCL. Se verificó que el import procede del paquete instalado, no de `src`. |
| Ejemplo de migración con NumPy y GPU | Correcto; resultado de distancias `[2.2360678, 7.071068]`. |
| Ejemplo de gradiente conjugado en GPU | Convergencia en 2 iteraciones; solución `[0.09090909, 0.6363636]`, residuo medido 0. |
| Residencia de datos | Pruebas impiden descargas durante aritmética compatible; en los solvers solo se descargan escalares. |
| Build | Wheel y sdist generados; wheel construido desde sdist. Se inspeccionó su contenido. |
| Instalación editable | Correcta en `.venv`, con dependencias de desarrollo y OpenCL. |
| Ruff | Lint y formato pasan. |
| Dependencias | `pip check` sin incompatibilidades. |
| Sintaxis Python 3.10 | Análisis sintáctico correcto; esto no sustituye ejecutar pruebas en Python 3.10. |

El compilador del driver NVIDIA emite un aviso sobre el atributo `noinline` de
kernels compilados correctamente. La suite no suprime esos avisos. Se usó
`PYOPENCL_NO_CACHE=1` durante la mayoría de las pruebas por las restricciones del
entorno de ejecución; también se verificó la ejecución con la caché normal habilitada.

## Medición ilustrativa

Matrices 256×256, float32, semilla 47, 5 repeticiones, mediana de tiempo de pared.
El tiempo OpenCL incluye despacho Python y sincronización, con inputs ya residentes;
no incluye la compilación inicial ni transferencias. Los tres casos se validaron contra NumPy.

| Operación | OpenCL residente | NumPy directo |
| --- | ---: | ---: |
| Suma elemento a elemento | 0.137 ms | 0.0155 ms |
| Producto matricial | 0.129 ms | 0.191 ms |
| Reducción sum | 0.136 ms | 0.0242 ms |

La transferencia inicial de los dos inputs tomó aproximadamente 0.154 ms. La primera
ejecución de cada operación, incluida compilación, tomó entre 61 y 112 ms. Estos
resultados de una ejecución local no implican una ventaja universal, ni reproducen
los experimentos del paper. Los archivos JSON y JUnit locales se conservaron en
`.artifacts`, excluido únicamente de este checkout.

## Pendiente de comprobar

- GPUs AMD e Intel, un dispositivo sin fp64 y sistemas distintos de Windows.
- Ejecución remota de la matriz CI y de PoCL: el workflow está preparado, no ejecutado aquí.
- Rendimiento sostenido, mayores tamaños y condiciones numéricas adversas.
- Compatibilidad de aplicaciones NumPy completas fuera del subconjunto documentado.
- Vulkan está verificado en la GPU local; faltan otros fabricantes. No hay publicación en PyPI.


## Compilador y Vulkan — 2026-09-06

Después de incorporar compile como función/decorador, planes cacheados y Vulkan:

- Suite CPU/OpenCL/Vulkan y adaptadores de benchmark: **318 passed, 1 skipped**.
  OpenCL y Vulkan obligatorios, CLBlast nativo habilitado. La prueba omitida requiere
  dos dispositivos OpenCL; este equipo expone uno.
- 65 pruebas Vulkan específicas, incluyendo API pública, captura y tres solvers
  float32/64; 42 casos de compilador CPU/OpenCL (incluidos escalares 0-D).
- Wheel instalado sin dependencias GPU: **122 passed, 186 skipped**.
  Ejemplo `examples/compiled_numpy.py` validado en CPU, OpenCL y Vulkan; CLI
  de descubrimiento Vulkan verificada.
- Ruff y formato pasan; wheel y sdist construidos con `python -m build --no-isolation`.
- 235 configuraciones de benchmark, 7 rondas residentes y 7 con transferencias,
  sin errores. Incluyen cuBLAS, CuPy, nvmath-python y solvers hasta convergencia.
  Ver [resultados y metodología](execution-results.md).

La arquitectura no promete NumPy completo; las limitaciones de
[compile](compilation.md) y [Vulkan](vulkan.md) son parte del contrato.
Las cifras históricas anteriores corresponden a etapas anteriores del runtime.
