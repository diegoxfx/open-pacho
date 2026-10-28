# Benchmark de la hipótesis de transferencia

## GEMM rectangular, cuBLAS vectorial y layouts compilados

Tres workers comparan cada cambio con dos controles de la ruta anterior sobre
los mismos inputs/contexto. Usan 30 muestras por defecto; `--repeats` debe ser
múltiplo de seis, para equilibrar el orden de los tres modos. Los resultados
incluyen precisión y estado numérico separado de finalización del experimento.

```powershell
python -m benchmarks.isolated --worker rectangular_gemm --backends opencl,cuda-custom --rounds 2 --repeats 30 --output .artifacts/benchmarks/rectangular
python -m benchmarks.isolated --worker compiled_layouts --backends opencl-clblast --rounds 2 --repeats 30 --output .artifacts/benchmarks/layouts
python -m benchmarks.isolated --worker blas_vectors --backends cuda-cublas --control-wheel <wheel-anterior-confiable> --rounds 2 --repeats 30 --output .artifacts/benchmarks/blas-vectors
```

`rectangular_gemm` desactiva solo la selección nueva de tiles; incluye vecinos
excluidos y controles cuadrados/batched. `compiled_layouts` desactiva el hook de
layouts al preparar cada control, incluyendo covarianza centrada. `blas_vectors`
reemplaza únicamente `CUDABackend.matmul` con el método del wheel anterior; incluye
float32/64 y GEMM como control negativo. Los overrides quedan fuera del reloj y
se restauran incluso ante errores. No ejecutar estos workers simultáneamente.
Ver [decisiones y resultados](../docs/three-iterations-results.md).

`benchmarks.suite` mide la hipótesis original con CuPy/CUDA como control externo.
Los runners posteriores también comparan el backend CUDA nativo de OpenPacho,
cuando se selecciona. Los benchmarks se ejecutan desde la raíz del repositorio.

## Preparación

```sh
python -m pip install -e ".[opencl,benchmark,dev]"
python -m pip install "cupy-cuda13x[ctk]"
python -m pytest benchmarks/test_suite.py
```

El extra CUDA necesita un driver compatible. Para otra versión de CUDA elige el
wheel apropiado según [CuPy](https://docs.cupy.dev/en/stable/install.html); no instales
dos variantes CuPy en el mismo entorno. El comando de la suite por defecto exige
OpenCL y CUDA; si un backend no está disponible, selecciónalo explícitamente con
`--backends opencl`, `--backends cuda` o `--backends ""` para medir CPU solamente.

## Ejecutar

En PowerShell, después de activar el entorno:

```powershell
$env:OPENBLAS_NUM_THREADS = "16"
$env:NVIDIA_TF32_OVERRIDE = "0"
$env:CUPY_CACHE_DIR = "$PWD/.artifacts/benchmarks/cupy-cache"
python -m benchmarks.suite --profile smoke --rounds 3 --output .artifacts/benchmarks/smoke
python -m benchmarks.suite --rounds 11 --output .artifacts/benchmarks/full
python -m benchmarks.suite --solver-sizes 2048,4096,8192 --rounds 7 --max-batch 1 --skip-transport --output .artifacts/benchmarks/solvers-large
python -m benchmarks.cold_start --rounds 3 --output .artifacts/benchmarks/cold.json
python -m benchmarks.report .artifacts/benchmarks/full/raw.json --extra .artifacts/benchmarks/solvers-large/raw.json --cold .artifacts/benchmarks/cold.json --output .artifacts/benchmarks/report
```

La suite rechaza sobrescribir un `raw.json` existente: usa un directorio nuevo para
otro ensayo. `checkpoint.json` conserva el avance. Cada error se registra y la
ejecución termina con código no cero si algún caso falla. La prueba `smoke` usa
tamaños pequeños y todos los tipos de operaciones antes del ensayo completo.

`PYOPENCL_NO_CACHE=1` puede usarse cuando el entorno impide escribir la caché del
usuario. La caché en memoria del contexto sigue activa; esta opción influye en la
primera compilación, no debe confundirse con transferencia ni con tiempo residente.
No ejecutes otras pruebas CPU/GPU simultáneamente con los benchmarks.

## Aplicaciones y alcance

Los solvers actuales se importan de `examples/solvers.py`, fuera del paquete.
Se conservan recurrencias, tolerancias y preparación documentada. Sus hashes
forman parte del registro de fuentes. Ejecutar desde la raíz del repositorio.
`LegacyAdapter` conserva deliberadamente los solvers del wheel histórico que
se está comparando; no sustituye ese algoritmo por el ejemplo actual.

Las optimizaciones nuevas pertenecen al compilador/backend y deben beneficiar
código equivalente del usuario. Ver [reglas y prioridades](../docs/architecture-evolution.md).

## Diseño predefinido

- 122 configuraciones: float32/64; vectores entre 1024 y 16.777.216 elementos;
  matmul cuadrado entre 16² y 2048² más cuatro formas rectangulares;
  matvec y transposición hasta 4096²; tres tamaños por solver.
- Extensión: 18 configuraciones de solvers con 2048, 4096 y 8192 variables,
  7 rondas y una llamada por muestra. La matriz mayor ocupa 512 MiB en float64.
- NumPy directo con BLAS de 1, 8 y 16 hilos; el análisis usa el más rápido medido
  por configuración y conserva todas las mediciones originales.
- OpenPacho/OpenCL y CuPy/CUDA: datos grandes residentes, subida+cálculo+descarga,
  subida por separado y descarga por separado.
- Subida y descarga puras con buffers existentes, desde 4 KiB hasta 256 MiB.
- Secuencias de 1, 10 y 100 pasos con una subida/descarga de arrays grandes. Los
  escalares Python de OpenPacho sí causan pequeñas subidas por paso.
- 11 rondas aleatorias por configuración; bloques adaptativos de hasta 64 llamadas,
  con sincronización después de **cada** llamada. Cada muestra guarda el promedio
  por llamada de ese bloque. No se mide solo el enqueue.
- Warmup y validación quedan fuera del reloj. El tiempo de las operaciones incluye
  temporales, despacho y destrucción de resultados. Se conservan los pools nativos
  de CuPy; OpenPacho usa sus asignaciones normales de `cl.Buffer`.
- Solvers equivalentes con el mismo dtype, tolerancia y residuo real. No se usa
  `numpy.linalg.solve` como baseline cronometrado. Se registra convergencia e
  iteraciones y se valida contra una solución conocida.
- TF32 deshabilitado en estos ensayos; no se obtiene velocidad a cambio de cambiar
  silenciosamente la precisión de float32.

La semilla, orden, versiones, BLAS, telemetría y hashes de fuentes quedan en
`raw.json`. La generación de gráficos usa `comparisons.csv`, que también conserva
el baseline CPU elegido, medianas y dispersión. Los intervalos bootstrap son
descriptivos del ensayo y no establecen garantías universales.

## Cómo interpretar

1. Una GPU que pierde con datos residentes no pierde por la transferencia de los
   arrays grandes: hay que investigar kernels, asignadores, despacho y sincronización.
2. Ganar residente y perder end-to-end indica que los costes de entrada/salida y
   preparación borran esa ventaja. No implica que el DMA sea el único coste.
3. Una carga con mayor cálculo por byte puede amortizar las copias, incluso en una
   sola llamada. Una secuencia puede reutilizar los mismos datos.
4. El resultado CuPy no demuestra que CUDA sea intrínsecamente superior a OpenCL:
   las dos bibliotecas tienen algoritmos, bibliotecas internas y asignadores diferentes.
5. El arranque de proceso y primera operación se reportan por separado: no equivalen
   al rendimiento sostenido de un programa que ya inicializó su GPU.

## Comparar optimizaciones en la misma sesión

`optimization.py` contrasta NumPy (1/8/16 hilos), kernels OpenCL propios,
fusión explícita y CLBlast opcional. También puede cargar un wheel anterior
con un nombre de módulo aislado, sin reinstalarlo ni cambiar el entorno actual.

```powershell
$env:OPENPACHO_CLBLAST_LIBRARY = "C:\ruta\clblast.dll"
python -m benchmarks.optimization --output .artifacts/benchmarks/optimization-nueva --clblast --legacy-wheel C:\ruta\open_pacho-anterior.whl --rounds 7
```

La carpeta de salida debe ser nueva. `raw.json` conserva muestras, hashes,
validaciones, iteraciones y tiempos residentes/con transferencias; `REPORT.md`
resume las medianas con transferencias. La referencia anterior se extrae dentro
de la carpeta de salida. Se omite `--clblast` si la biblioteca nativa no está
instalada, y `--legacy-wheel` si no se dispone de un paquete anterior.

En esta comparación `opencl` ya incluye pool de memoria, escalares por valor y
solvers con actualizaciones fusionadas. `opencl-fused` añade el decorador a las
expresiones genéricas; `opencl-clblast` sustituye sus productos flotantes. CLBlast ya usa GEMV y DOT dedicados cuando sus símbolos están disponibles;
los informes anteriores describen el código de su propia etapa. Los resultados históricos no se sobrescriben.

Para ejecutar la suite de conformidad sobre CLBlast, configura la ruta anterior y:

```powershell
$env:PACHO_BLAS = "clblast"
$env:PACHO_REQUIRE_OPENCL = "1"
python -m pytest
```

## Control de la frontera Vulkan

Requiere el [extra, driver y compilador Vulkan](../docs/vulkan.md); la instalación
OpenCL/CuPy de la suite original no los incorpora.

`python -m benchmarks.vulkan_dispatch --repeats 21 --output <archivo-nuevo.json>`
intercala rutas wrapped/direct del mismo backend con kernels, algoritmos y precisión
iguales. Mide operaciones residentes, aplicaciones completas y copias; float32/64,
rectangulares/impares y solvers externos hasta el residuo acordado. `--quick`
reduce el alcance para verificar el runner. Repetir con `--seed 4782` cambia el
orden, no los inputs.

La instrumentación de API y timestamps se instala solo en una invocación aparte,
restaura funciones al terminar y espera antes de destruir queries. Los intervalos
GPU no son tiempos puros de shader; los tiempos de API incluyen driver/esperas.
Se registran asignaciones nuevas y caché, no pico total de VRAM.
Ver [resultados y límites](../docs/vulkan-dispatch-results.md).

`--control descriptors` compara `sets`/`push`, dejando la frontera CFFI tipada
igual en ambos. Registra maxPushDescriptors, conteos y timestamps; exige la
extensión push en el control solicitado. El control `commands` conserva sets
en ambas rutas para aislar los wrappers. Ver [nueva ronda](../docs/backend-sharing-results.md).

## Operaciones compartidas entre backends

El conjunto predeterminado requiere OpenCL, CLBlast nativo, Vulkan, CUDA nativo
y CuPy. Instalar solo los backends necesarios siguiendo [OpenCL](../README.md),
[Vulkan](../docs/vulkan.md) o [CUDA](../docs/cuda.md); para CLBlast configurar la
biblioteca nativa como en la comparación anterior. Limitar `--backends` a los
disponibles. Las ausencias quedan en el JSON y dan
código de salida no cero; no se omiten silenciosamente.

```powershell
python -m benchmarks.backend_sharing --repeats 21 --output .artifacts/operations-new.json
```

Mide cadenas elementwise, transposición/reversión, GEMV, GEMM y lotes regulares/
broadcast, float32/64 y formas irregulares. Intercala proveedores y publica pared
residente, duración de la llamada host, ejecución completa, primera llamada,
errores y muestras/p10/p90. La duración host incluye esperas internas y API;
no representa tiempo puro de Python ni utilización CPU.

Para contrastar un wheel histórico sin instalarlo, ejecutar el mismo módulo en
otro proceso con `PYTHONPATH` apuntando al wheel. El JSON registra el origen y los
hashes del paquete realmente importado. Repetir con `--seed 942` cambia el orden,
no los datos. Las operaciones históricamente ausentes se etiquetan `unsupported`;
no cuentan como ejecuciones validadas ni como tiempos cero. Los archivos de salida
deben ser nuevos y las cargas GPU deben ejecutarse secuencialmente.

`--control-wheel <wheel-anterior>` alterna métodos anteriores/actuales sobre el
mismo contexto y datos residentes: elementwise/gather en OpenCL, matmul en CUDA.
Seleccionar un único backend OpenCL o CUDA con `--backends`. Las demás funciones
son las del runtime actual y el cambio de método queda fuera del reloj. Este
control reduce diferencias entre contextos; no equivale a cargar todo el paquete
histórico. El runner registra el wheel, su hash y el alcance del control.

`register_gemm` y `push_descriptor_limit` describen capacidades del backend,
no una traza de qué kernel ejecutó cada fila. El control Vulkan registra además
la modalidad y los comandos efectivos. La estrategia CUDA se verifica mediante
pruebas de llamadas nativas y las condiciones de selección del runtime.

Para comparar el uso habitual de cada API, ejecutar un proceso por proveedor,
secuencialmente, por ejemplo `--backends vulkan` y luego `--backends opencl`, con
archivos distintos. La intercalación de varias APIs en un proceso puede perturbar
mucho sus tiempos en este driver Windows. Registrar también ese control si se
investiga coexistencia; no confundirlo con rendimiento aislado o límite físico.

## Reposo observado y experimentos de kernels

`python -m benchmarks.isolated --backends opencl,vulkan,cuda-custom --rounds 2 --output <directorio-nuevo>`
automatiza procesos secuenciales y reposo observado antes de cada uno: pausa mínima
de 10 s, temperatura/utilización, tres lecturas y timeout que impide arrancar el
siguiente worker. Registra clocks, potencia y señales térmicas; no fuerza hardware.
El worker vuelve a calentar fuera del reloj al menos tres llamadas y 0,25 s.

`--worker kernel_variants` compara candidatos de tiles, padding, layout local y
guardas. `backend_sharing` añade ahora GEMM 512×256×768 y 1024³, para 22 casos.
`--control-component gemm --control-wheel <wheel>` controla solo el generador de
GEMM sobre un backend OpenCL/CUDA, conservando las demás rutas actuales. Los
overrides son temporales y se restauran al terminar; ese control requiere un
proceso sin llamadas concurrentes al runtime.

El recolector actual es NVIDIA y `--gpu-id` elige la GPU observada, no el dispositivo
del backend. Verificar su correspondencia si hay varias GPUs. Sin telemetría,
`--allow-unverified` permite únicamente una pausa etiquetada como no verificada;
nunca convierte una lectura caliente/ocupada en aprobación por timeout.
Ver [protocolo, restricciones y resultados](../docs/kernel-stability-results.md).

## Osciladores acoplados

`python -m benchmarks.oscillators --backend numpy --profile quick --repeats 12 --output <json-nuevo>`
evalúa 64 pasos velocity-Verlet y separa recurrencia discreta, solución modal y
energía. `--profile all` añade N=257/2049 a N=31, float32/64. Los proveedores
admitidos son NumPy, OpenCL custom/CLBlast, Vulkan, CUDA custom/cuBLAS y CuPy;
OpenPacho compara eager con un paso preparado reutilizado. Las muestras incluyen
la aceleración inicial en cada recorrido, 65 GEMV, y parten del estado original.
Los procesos GPU requieren reposo y exclusividad externos. Ver el
[protocolo y los resultados](../docs/oscillators-results.md).

## Experimento CUDA float64

`python -m benchmarks.cuda_fp64_gemm --profile all --repeats 24 --output <json-nuevo>`
compara el GEMM directo actual con un candidato compartido 16×16, sin cambiar
el runtime. Requiere un dispositivo exclusivo y una puerta de reposo observada
externa antes del proceso. Conserva dos controles, las seis permutaciones de
orden, muestras, validación bit a bit y aplicaciones con planes independientes.
`--control-report <json-anterior>` reutiliza las fuentes directas capturadas.
Ver [alcance, evidencia y decisión](../docs/cuda-fp64-gemm-results.md).

## Contracción float64

`python -m benchmarks.fp64_contraction --backend cuda-custom --profile all --repeats 24 --output <json-nuevo>`
compara redondeo separado y FMA en dos geometrías CUDA. `--backend opencl`
compara su programa actual con la misma fuente y `FP_CONTRACT OFF`. Los módulos
son propiedad del benchmark; no se instalan en las cachés del runtime. Se guardan
fuentes, opciones efectivas, PTX/binarios y oráculos de ambas políticas, además
de referencias exactas de error y dos controles de rendimiento. Requiere un
proceso GPU exclusivo y reposo externo. No cambia ninguna política predeterminada.
Ver [protocolo y evidencia](../docs/fp64-contraction-results.md).

## Staging y generación aleatoria

`python -m benchmarks.isolated --worker staging --backends vulkan --control-wheel <wheel-anterior> --rounds 2 --repeats 31 --output <directorio-nuevo>`
compara los métodos de transferencia de un wheel local confiable con los actuales
sobre el mismo backend/contexto. Sin wheel, compara modos transitorio/readback
del código actual. Alterna bloques con calentamiento idéntico y separa diagnóstico.

`python -m benchmarks.isolated --worker random_generation --backends numpy-1t,opencl,vulkan,cuda-custom,cupy --rounds 2 --repeats 21 --output <directorio-nuevo>`
mide resultado residente, entrega al host y NumPy seguido de subida. Usa las pausas
observadas del runner; cada proveedor corre en un proceso separado. Los algoritmos
son distintos (Philox OpenPacho, default_rng NumPy, RandomState CuPy); el informe
registra momentos, rangos y, para OpenPacho, error contra las mismas transformaciones
Philox en CPU. Uniforme NumPy incluye su conversión a float32; random y las variantes
estándar normal/exponencial usan el dtype nativo de NumPy. No mide calidad estadística
exhaustiva ni promete reproducción de secuencias entre proveedores.
