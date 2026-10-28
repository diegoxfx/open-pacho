# Map-reduce, preparación y BLAS batched

Snapshot histórico. La [continuación actual](numerics-vulkan-results.md) añade
optimización del ejecutor Vulkan, GEMM con registros y estabilidad/diagnóstico numérico.

Etapa del 2026-09-06, posterior a [ejecución y staging Vulkan](vulkan-execution-results.md).
Equipo local: RTX 5060 Ti 16 GB, Ryzen 7 5700X3D, Windows 11, driver 616.64.
Solo este equipo tiene validación física. La especialización por familia de
hardware/SO permanece como un [track secundario](hardware-specialization.md).

## Implementaciones

1. **Map-reduce Vulkan.** Fusiona el productor elemental con `sum` y con las
   reducciones que forman `mean`/normas. Mantiene dtype del productor, acumulación
   solicitada, broadcasting, ejes arbitrarios, `keepdims`, booleanos y vacíos.
   Usa parciales compactos para reducciones grandes; no crea el array completo
   del productor. Un productor compartido entre consumidores distintos o devuelto
   como salida sigue materializándose. GEMM/GEMV conservan su frontera.
2. **Llamadas compiladas preparadas.** `f_compilada.prepare(*ejemplos)` fija la
   especialización y devuelve un callable con validación de bindings. Evita el
   binding general en el caso posicional completo y la búsqueda en la LRU. Los
   escalares dinámicos siguen cambiando; inputs nuevos deben conservar forma,
   dtype y contexto. No retiene los arrays de ejemplo ni comparte outputs vivos.
   No genera un ejecutor nativo ni un grafo persistente de comandos GPU.
3. **Sistemas preparados.** `solvers.prepare(A, method="cg" | "jacobi" | "sd")`
   toma una copia propia y valida la matriz una vez. Cada `.solve(b)` conserva
   validación de b/x0, tolerancias y residuo real en cada iteración. Jacobi reutiliza
   la diagonal validada. La precisión queda fijada; una promoción exigida por b
   se rechaza. Los solvers GPU ordinarios también preparan sus llamadas compiladas
   una vez por resolución, en lugar de repetir búsquedas por iteración.
4. **CLBlast batched nativo.** Usa S/DgemmStridedBatched para offsets positivos
   regulares y S/DgemmBatched para otros patrones disponibles. Broadcasting usa
   offsets sobre los buffers originales. Si faltan símbolos opcionales, conserva
   GEMM del mismo proveedor; un error nativo se propaga. Continúa la protección
   existente de salida GEMV reciclada con NaN/Inf.
5. **GEMM OpenCL con tiling.** Ruta portable 16×16 si el dispositivo admite el
   grupo y memoria local; 8×8 con menores capacidades; kernel anterior para casos
   pequeños o capacidad insuficiente. Preserva el broadcasting y productos
   incompletos en bordes. Permite contracción multiplicación-suma como el GEMM
   anterior; no activa fast-math ni cambia la fusión elementwise.

Los ejemplos completos y los límites están en [compilación](compilation.md) y
[ejecución y rendimiento](performance.md). El usuario sigue escogiendo backend
y proveedor explícitamente.

## Cómo se mide

La batería amplia conserva 319 configuraciones: 172 de operaciones/pasos, 63 de
solvers completos y 84 de escala/batching. Cada modo tiene siete muestras con
sincronización; residente excluye subida/bajada y extremo a extremo las incluye.
Incluye NumPy con 1/8/16 hilos, OpenCL custom/CLBlast, Vulkan, CuPy y referencias
directas cuBLAS/nvmath en los casos admitidos. No son 319 workloads distintos:
el número incluye implementaciones, tamaños y dtypes. No se incorporan PyTorch/JAX
en esta etapa ni un backend CUDA de OpenPacho.

El control nuevo `benchmarks.prepared` mide 48 configuraciones de método, tamaño,
dtype y backend, con cinco modos: resolución ordinaria, preparada, preparación
por separado y ambos recorridos de una sola resolución desde host. Cada resultado
se contrasta con el solver ordinario, su historial y un residuo calculado en
float64. Usa rtol=1e-5/1e-8 para float32/64 y maxiter=1000; la batería amplia usa
rtol=1e-4/1e-7 y maxiter=400. Se compara cada variante dentro de su propio control,
sin equiparar tiempos entre tolerancias distintas. NumPy en este control es el backend de OpenPacho con un hilo; no debe
confundirse con NumPy nativo de la batería amplia.

Los perfiles antes/después usan la rueda congelada de la etapa anterior y la
fuente actual en procesos separados. El perfil de energía distingue tres rutas:
API compilada, API preparada con validaciones y ejecución privada del plan sin
esas validaciones. Esta última es solo un control diagnóstico.

Las medianas históricas no prueban causalidad por sí solas: también cambian rutas
sin modificaciones, como CuPy. No se fijaron clocks ni se controló completamente
el estado del driver/planificador Windows. Se conservan resultados desfavorables
y controles intercalados dentro del mismo proceso para separar parte de esa variación.

## Resultados y atribución

Medianas en milisegundos. Los controles locales alternan las variantes dentro del
mismo proceso y conservan once muestras por variante tras calentamiento.

| Control residente | Ruta anterior | Ruta nueva | Lectura |
| --- | ---: | ---: | --- |
| CLBlast, 64 matrices de 32×32 | 2.028 | 0.236 | Strided-batched, 8.6× menos tiempo que el bucle GEMM |
| CLBlast, 16 matrices de 128×128 | 0.977 | 0.242 | Strided-batched, 4.0× |
| CLBlast, broadcasting (8,1,64,64) × (1,4,64,64) | 2.219 | 0.490 | Batched con offsets, 4.5× |
| OpenCL custom, GEMM 1024² float32 | 4.811 | 1.860 | Sin tiling frente a tile 16×16 |
| OpenCL custom, GEMM 2048² float32 | 24.886 | 18.506 | 1.34×; todavía lejos de BLAS optimizado |
| OpenCL custom, GEMM 1024² float64 | 8.526 | 8.590 | Sin mejora relevante |

La variante inicial de tiling con contracción deshabilitada dio 18.976 ms en
float64 1024². Se descartó esa política para GEMM: el camino anterior ya permitía
contracción. El cambio no afecta la prohibición de contracción en expresiones
elementwise fusionadas. Los controles se conservan en `tile-control*.json`.

En la batería amplia, GEMM custom 1024² float32 pasó de 5.544 a 1.794 ms residente,
pero 2048² pasó de 16.358 a 18.265 ms. CLBlast con broadcasting irregular pasó de
1.105 a 1.661 ms, aunque el control intercalado favorece la ruta nativa. No se
presentan estas mediciones históricas como una mejora uniforme ni se atribuye
toda variación al cambio de kernel.

**Map-reduce sí reduce trabajo verificable.** En el perfil de energía Vulkan
2048×32, los dispatches y asignaciones lógicas bajan de dos a uno. En CG del perfil
(dos iteraciones), los dispatches bajan de 43 a 31 y las llamadas `empty` de 51 a
32; permanecen nueve descargas y diez sincronizaciones contadas, incluida la
sincronización externa del perfil. Son llamadas al backend, no asignaciones
nativas nuevas: el pool puede reutilizar almacenamiento.

El perfil separado de energía Vulkan pasa de 0.544 a 0.409 ms. En la batería
amplia, energía 32768×32 pasa de 2.390 a 0.840 ms residente, frente a 3.413 ms de
NumPy de un hilo. Sin embargo, con carga y descarga Vulkan tarda 5.503 ms frente
a 3.325 ms de NumPy. La mejora de ejecución no elimina el coste de transferencia.

**Preparar el ejecutor no resuelve el cuello de botella principal.** Para energía
OpenCL 2048×32, la API compilada da 0.099 ms, la preparada con validación 0.072 ms
y el plan privado sin validación 0.066 ms. Vulkan da 0.409/0.441/0.391 ms: no hay
ganancia medible de la API preparada en esa ronda. El control intercalado de CG
compara llamadas cacheadas frente a preparadas, ambas validadas: OpenCL da
2.318/2.327 ms y Vulkan 12.824/11.073 ms. No justifica reescribir todo el compilador
antes de reducir el coste de ejecución del backend.

**Los sistemas preparados tienen una frontera de utilidad concreta, todavía
modesta.** Con 2048 variables float32 y rtol=1e-5:

| Backend / método | Ordinario residente | Preparado residente | Preparar A una vez |
| --- | ---: | ---: | ---: |
| OpenCL custom / CG | 20.362 | 18.173 | 0.824 |
| OpenCL custom / Jacobi | 23.756 | 23.938 | 1.073 |
| CLBlast / CG | 23.380 | 24.503 | 0.538 |
| Vulkan / CG | 65.042 | 62.144 | 1.079 |
| Vulkan / Jacobi | 88.671 | 85.958 | 2.421 |
| Vulkan / descenso más pronunciado | 141.201 | 141.728 | 1.226 |

No se anuncia un número universal de reutilizaciones para amortizar: cuando el
ahorro es cercano a la variación observada, dividir por esa diferencia produce
una estimación inestable. Los cinco modos completos están en `prepared.json`.

La batería amplia de Jacobi 2048 float32, con su rtol=1e-4, sigue dando **9.443 ms
en NumPy de ocho hilos, 22.308 en OpenCL custom y 76.424 en Vulkan**, con datos
residentes. La corrección y la reducción de dispatches no equivalen a haber
cerrado este gap; también hubo regresiones de medianas frente a la etapa anterior.

## Verificación

- **625 pruebas pasan, una omitida** por necesitar dos dispositivos OpenCL.
  Incluye límites de índices Vulkan, redondeo de productores, ejes y vacíos,
  ownership/bindings preparados, historial de convergencia, variantes tiled y
  fallback, ABI y errores de CLBlast y salida reciclada con NaN.
- **319 configuraciones comparativas y 48 de preparación**, todas con siete
  muestras por modo y validaciones completas, sin errores registrados.
- Wheel construido desde sdist e instalado sin dependencias GPU: **208 pruebas
  pasan, 401 omitidas** por ausencia de backends opcionales.
- Ruff y formato pasan. Distribución en `.artifacts/dist-remaining-suite`;
  datos, tabla completa, figura y manifest en `.artifacts/benchmarks/remaining-suite`.

## Gaps que quedan

- Los solvers residentes siguen perdiendo frente a NumPy en varios casos.
  Preparar bindings no elimina llamadas al driver ni las observaciones host de
  convergencia. El siguiente trabajo de ejecución debe medir envío/grabación
  Vulkan y un ejecutor nativo pequeño, conservando recursos en vuelo y locks.
- Falta fusionar GEMV con actualizaciones/reducciones. CLBlast no impide añadir
  rutas custom, pero esta etapa no implementa epílogos BLAS ni un solver GPU
  completamente autónomo.
- La preparación de sistemas evita validaciones repetidas; no es una factorización
  ni un precondicionador y no garantiza una mejora de tiempo. Añade una copia de A.
- Transferencias y launch overhead siguen dominando casos de baja intensidad
  aritmética. Ningún cambio hace desaparecer PCIe ni vuelve rentable toda operación
  pequeña en GPU. Las pérdidas de ejecución residentes son un gap separado.
- La norma todavía usa suma de cuadrados: el subdesbordamiento puede dar falsa
  convergencia; desbordamiento y coeficientes extremos también necesitan una ruta
  escalada. Esta deuda numérica no queda resuelta por fusión ni preparación.
- AMD/Intel, dispositivos sin fp64 y otros SO requieren validación física.
  CUDA backend y matrices dispersas continúan fuera de esta etapa.

## Reproducción

Con OpenCL, CLBlast, Vulkan/glslang y referencias CUDA configurados:

```powershell
python -m benchmarks.execution --output .artifacts/benchmarks/remaining-suite/operations.json
python -m benchmarks.execution --solvers-only --output .artifacts/benchmarks/remaining-suite/solvers.json
python -m benchmarks.scale --output .artifacts/benchmarks/remaining-suite/scale.json
python -m benchmarks.prepared --output .artifacts/benchmarks/remaining-suite/prepared.json
python -m benchmarks.gaps --backends opencl,opencl-clblast,vulkan --output .artifacts/benchmarks/remaining-suite/after-profile.json
python -m benchmarks.gaps_tables --root .artifacts/benchmarks/remaining-suite --operations operations.json --solvers solvers.json --scale scale.json --output .artifacts/benchmarks/remaining-suite/report
```

Los JSON privados conservan muestras, errores y procedencia; el informe CSV/Markdown
conserva todas las configuraciones. Los experimentos iniciales de tiling se guardan
aparte y no se presentan como el estado final.
