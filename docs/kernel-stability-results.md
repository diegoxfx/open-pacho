# Selección de kernels y estabilidad de benchmarks — 2026-09-13

Esta ronda convierte las propuestas de optimización en controles medibles. Mantiene
la prioridad de ejecución/transferencias Vulkan y las fronteras pendientes de
CUDA; no sustituye ese trabajo por un autotuner general ni modifica los solvers.

## Finalización, reposo y calentamiento

Son tres etapas distintas. Sincronizar la cola garantiza la finalización del
trabajo de nuestra aplicación. Descargar resultados solo se incluye en el reloj
de extremo a extremo; las validaciones descargan fuera del reloj residente.
Una espera temporal por sí sola no garantiza la finalización de una cola GPU.

El nuevo `benchmarks.isolated` ejecuta un proveedor por proceso y espera su salida
antes de comprobar el estado de la GPU. El worker ya sincroniza todas las muestras
y termina sus descargas de validación. El siguiente proveedor recibe una pausa
mínima de 10 segundos y una comprobación de reposo con tres lecturas consecutivas,
separadas por dos segundos: utilización GPU/memoria ≤5 %, temperatura estable
dentro de 2 °C y, después del primer proveedor, no superior a su referencia inicial
más 2 °C. Se puede fijar además una temperatura máxima; aquí se usó 52 °C.

Se registran UUID, bus PCI, temperatura, utilización, frecuencias, potencia, pstate
y motivos de reducción de frecuencia. Una señal térmica activa impide continuar.
Si no se alcanza el criterio en 120 segundos, **el siguiente worker no arranca**.
No se confunde timeout con aprobación. Sin telemetría, solo `--allow-unverified`
permite una pausa fija explícitamente etiquetada como no verificada; no permite
ignorar una GPU caliente u ocupada observada previamente.

El recolector actual usa `nvidia-smi` solo en benchmarks. El runtime y sus extras
siguen siendo independientes. `wait_ready` acepta un proveedor de lecturas, para
añadir telemetría AMD/Intel posteriormente. En equipos con varias GPUs, el usuario
debe asociar la GPU observada con los dispositivos seleccionados en cada API:
el runner no verifica automáticamente esa correspondencia por UUID. Aquí todas
las rutas ejecutaron sobre la única RTX 5060 Ti observada.

Después del reposo, cada variante recibe al menos tres llamadas y 0,25 segundos
de calentamiento, fuera del cronómetro. Primera compilación, reposo y rendimiento
caliente quedan separados. El orden inicial se aleatoriza y se invierte en la
segunda ronda. La pausa evita arrastrar trabajo/temperatura excesiva; no garantiza
frecuencias constantes ni ausencia de ruido del SO. Los motivos de potencia o
pstate se registran sin forzar clocks, ventiladores o límites de consumo.

En el control final de variantes, los seis reposos cumplieron el criterio: cinco
duraron aproximadamente 10,4 s y uno necesitó **16,6 s**. Las temperaturas de las
ventanas aceptadas estuvieron entre 40 y 42 °C. No se descartan muestras lentas
para hacer parecer mejor un backend. Las [definiciones de telemetría de NVIDIA](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
explican que utilización y motivos de frecuencia son observaciones con ventanas
de muestreo, no un certificado de estabilidad de toda la ejecución.

## Qué se está optimizando

El GEMM actual ya usa tiles de salida de 32×32, bloques de 64 threads y memoria
compartida para dos tiles de 32×16. Cada thread acumula **4×4 resultados en registros**:
ya implementa la idea de asignar varios elementos a un thread.

| Técnica | Criterio y coste que debemos medir |
| --- | --- |
| Tiling | Reutilización frente a memoria compartida, registros, ocupación y coste de tiles parciales. Mayor tile no implica mayor rendimiento. |
| Más elementos por thread | Reutilización e instrucciones frente a presión de registros, spills y paralelismo. Variar bloque por thread y tile es una decisión conjunta. |
| Tipos vectoriales | Verificar alineación, stride, accesos de cola y código emitido. `float4` puede ayudar a cargas/almacenamientos sin implicar cuatro operaciones aritméticas simultáneas por thread. |
| Bancos de memoria | Modelar lecturas y escrituras: arreglar conflictos de una fase puede empeorar otra. Lecturas de la misma dirección pueden difundirse; no son necesariamente conflictos. |
| Transposición/packing | Distinguir flags/vistas, reordenación dentro del tile y copia global. Medir preparación y reutilización antes de materializar un operando. |
| Padding | Contabilizar bytes y límites antes de reservar o lanzar; datos de relleno y accesos deben ser válidos. Separar padding local de inflar matrices globales. |
| Eliminar condiciones | Solo retirar comprobaciones demostrablemente redundantes; conservar guardas de bordes y participación uniforme en barreras. Medir el código del compilador, que puede predicar o eliminar condiciones. |

Para una copia global reutilizada R veces, el punto de amortización exige
`T_preparación + R*T_kernel_preparado < R*T_kernel_original`. La selección debe
incluir memoria extra, uploads/readback pertinentes y validez de la preparación
ante mutaciones. No se implementó en esta ronda una copia global automática.

OpenCL define `float2/4/8/16` como vectores de valores float; `float16` no significa
FP16. Su existencia en el lenguaje no garantiza la misma instrucción máquina o
ventaja en otra API. Ver [tipos vectoriales OpenCL](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html#vector-data-types).
La [guía CUDA de memoria compartida y registros](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)
detalla conflictos, broadcasts y costes de recursos; el banco físico y el código
generado deben verificarse en el hardware correspondiente.

## Experimento controlado

`benchmarks.kernel_variants` mide cinco formas float32: 32×16×32, 33×17×65,
257×129×513, 1025×513×769 y 1024³. Diez variantes por forma, tres backends y dos
rondas producen **300 configuraciones**, con 21 muestras por variante. Todas
pasan las comprobaciones numéricas y conservan los 16 valores centinela al final
de la salida. La salida se inicializa con NaN para detectar escrituras omitidas.

Se comparan el kernel actual, una base generada sin índices de batch, padding
izquierdo/ambos, transposición local con/sin padding, bloque de registros 2×2,
tile K=32 y especialización de bordes. Dos controles de bordes separan el cambio
del generador experimental del cambio en el generador real del runtime.

Las variantes declaran 2.048–8.192 bytes de memoria compartida y comprueban el
límite del dispositivo. El presupuesto lógico de datos se comprueba antes de
reservar las matrices. No mide pico real de VRAM ni memoria del driver, compilador
o cachés del asignador. No se rellena ni transpone globalmente ningún operando.

El reloj mide lanzamiento y sincronización con salida preasignada: **no equivale
a una llamada pública completa ni a tiempo puro de kernel**. Las descargas,
comprobaciones, compilación, calentamiento y telemetría quedan fuera. Los errores
se comparan contra los inputs float32 representados, recalculados en NumPy float64.
No es una prueba de precisión adversa ni cubre todos los patrones de memoria.

## Selección de la optimización

Eliminar únicamente condiciones redundantes en el generador real para 1024³ da:

| Backend | Guardado → tiles completos, ronda 1 (ms) | Ronda 2 (ms) |
| --- | ---: | ---: |
| OpenCL | 0,790 → 0,477 | 0,778 → 0,465 |
| CUDA custom | 0,433 → 0,429 | 0,433 → 0,430 |
| Vulkan | 0,473 → 0,468 | 0,491 → 0,478 |

Solo se promueve la variante OpenCL, donde el ahorro es grande y repetido. La
selección automática requiere GEMM float32 elegible y **M/N múltiplos de 32, K
múltiplo de 16**. Mantiene tiles, memoria, orden aritmético y soporte de lotes.
Las otras formas conservan sus guardas y CUDA/Vulkan conservan sus shaders.
No requiere un flag del usuario ni cambia dtype o modos matemáticos.

Las variantes de padding/transposición no se habilitan indiscriminadamente:
varias empeoran los GEMM grandes Vulkan/CUDA. El bloque por thread 2×2 tampoco
compensa su pérdida de reutilización en esos casos. Un cambio menor del código
de índices también puede cambiar lo que optimiza cada compilador; de ahí la
base generada adicional. Los resultados completos, incluidos los perdedores,
quedan en los JSON.

El control posterior mediante `a @ b`, con asignación de salida y las mismas
rutas públicas, ejecutó **88 configuraciones** (22 casos × dos variantes × dos
rondas), sin errores. Reutiliza el mismo backend y alterna solo el componente
GEMM anterior/actual extraído del wheel histórico. Las métricas numéricas de cada
pareja coinciden. Las operaciones sin cambios son controles del ruido.

| OpenCL float32 | Residente anterior → actual, ronda 1 / ronda 2 (ms) | Completo, ronda 1 / ronda 2 (ms) |
| --- | ---: | ---: |
| 512×256×768 | 0,156 → 0,129 / 0,159 → 0,128 | 0,988 → 0,947 / 0,980 → 1,012 |
| 1024³ | 0,837 → 0,534 / 0,843 → 0,534 | 3,245 → 2,899 / 3,183 → 2,819 |
| 1025×513×769, guardas conservadas | 0,400 → 0,402 / 0,402 → 0,401 | 1,845 → 1,848 / 1,939 → 1,929 |

1024³ mejora aproximadamente **36–37 % en pared residente y 11 % en ejecución
completa**. En 512×256×768 el ahorro residente es 17–20 %, pero la ejecución
completa es mixta. La selección conserva el mismo coste de memoria GPU y no
elimina transferencias. Una mejora del kernel no basta para prometer una mejora
de aplicación en cada llamada pequeña o con pocos cálculos por byte.

## Reproducción

Desde el repositorio, con extras, drivers y compiladores de los backends elegidos:

```powershell
python -m benchmarks.isolated --worker kernel_variants --backends opencl,vulkan,cuda-custom --rounds 2 --repeats 21 --max-temperature 52 --output .artifacts/variantes-nuevas
python -m benchmarks.isolated --backends opencl --control-wheel <wheel-anterior> --control-component gemm --rounds 2 --repeats 21 --output .artifacts/control-publico-nuevo
```

Usar directorios nuevos y cargas GPU secuenciales. Los controles de wheel GEMM
alternan temporalmente el generador anterior/actual en el mismo proceso y contexto,
fuera de los relojes; no permiten uso concurrente del runner en ese proceso.
Las sustituciones se restauran incluso si la ejecución falla. No alteran el API
que utiliza la aplicación.

Artefactos: `.artifacts/kernel-stability/variants-final`, `public-control` y logs.
La primera carpeta `variants` fue exploratoria y precede al control adicional
del generador; no sustituye la ronda final. Hardware local: RTX 5060 Ti, Windows
11, driver 616.64. Las conclusiones no certifican otro fabricante, driver o SO.

Verificación final: **1.122 pruebas pasan y una se omite** por necesitar dos
dispositivos OpenCL. Incluye los tres backends obligatorios y CLBlast nativo,
controles de bordes alineados/parciales y broadcasting de lotes. Los **42 casos
adversos OpenCL** conservan las mismas nueve alertas previas, sin excepciones;
no se anuncian resueltos los problemas de cancelación. No se repitió esa suite
adversa para los shaders predeterminados CUDA/Vulkan, que no cambiaron.
Después de ajustar la propagación de la GPU observada desde el padre al worker,
las **25 pruebas del protocolo y controles** pasan. Esto mantiene coherente la
telemetría solicitada; no añade asociación automática con el dispositivo de cada API.

Wheel desde sdist: **327 pruebas CPU pasan, 727 omitidas**, con importación desde
el wheel y módulos GPU/ejemplos bloqueados en la comprobación de independencia.
Se verificaron extras separados, NumPy como dependencia base y exclusión de
ejemplos/solvers/benchmarks del wheel. Ruff y formato pasan (88 archivos); sintaxis
Python 3.10 analizada, ejecución en Python 3.12. Solo `_gemm.py` cambió en el
runtime respecto al wheel anterior. Los 12 controles de fuentes guardadas son
idénticos a los anteriores. No se publicó una nueva versión.
