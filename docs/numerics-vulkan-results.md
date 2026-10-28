# Vulkan, GEMM y estabilidad numérica

Snapshot histórico. La [continuación](solver-applications-results.md) agrupa
pasos de solver y command buffers, amplía aplicaciones y prepara la interfaz CUDA.

Iteración del 2026-09-07. Continúa la [suite anterior](remaining-suite-results.md).
RTX 5060 Ti 16 GB, Ryzen 7 5700X3D, 32 GB RAM, Windows 11, driver 616.64.
Python 3.12.14, NumPy 2.5.3, PyOpenCL 2026.1.4, Vulkan Python 1.3.275.1,
glslang 16.5.0, CLBlast 1.7.0, CuPy 14.2.0 y nvmath-python 1.0.0.
La validación física sigue limitada a este equipo.

**Hay mejoras importantes dentro del runtime y de los kernels. Vulkan ya supera
a CLBlast en el GEMM residente 2048² float32 medido, pero sigue perdiendo en
transferencias y solvers. La precisión ahora forma parte visible del resultado.**

## Implementaciones

1. **Menos construcción de objetos Vulkan.** LRU de 128 plantillas de dispatch
   con fuente, ABI y geometría; estructuras CFFI de descriptores conservadas por
   lease y reenlazadas después de completar su uso. Se reutilizan estructuras de
   envío, barreras y copia bajo el lock. Los recursos en vuelo siguen retenidos;
   no se reduce el número de submits mediante esta mejora por sí sola.
2. **GEMM con bloques de registros.** OpenCL custom y Vulkan float32 calculan
   32×32 salidas por grupo de 64 hilos, con 4×4 acumuladores por hilo y tiles K=16.
   Conservan broadcasting y bordes. La selección comprueba forma y capacidades;
   otras precisiones/tamaños mantienen las rutas anteriores. La
   [investigación de CLBlast](clblast-techniques.md) explica la técnica y qué se pospone.
3. **GEMV y residuo fusionados.** El compilador puede convertir el patrón privado
   `b - A @ x` en una operación con epílogo para OpenCL custom/Vulkan y dtypes
   flotantes iguales. No se cambia el proveedor CLBlast. Productos compartidos,
   salidas públicas y casos no elegibles mantienen la frontera. No es fusión
   general de cualquier expresión alrededor de matmul.
4. **Normas y coeficientes estables.** Una primitiva de norma escalada evita
   formar cuadrados extremos. CG y descenso más pronunciado usan direcciones
   normalizadas; se elimina el uso de `r·r` para esos coeficientes. Se conserva
   dtype, residuo verdadero en cada iteración, tolerancia, validación e información
   de convergencia. No garantiza que cualquier problema extremo sea representable.
5. **Precisión observable.** Tiempos acompañados de error absoluto, relativo L2,
   recuento de no finitos y referencia. Solvers añaden residuo independiente y
   error hacia atrás. Una suite adversa usa Decimal80 sobre los inputs realmente
   representados. Ver [método y reproducción](numerical-accuracy.md).

## Controles que aíslan las mejoras

GEMM: once muestras intercaladas dentro del mismo proceso. La variante anterior
desactiva solamente la ruta de registros. Medianas residentes sincronizadas:

| Backend / float32 | Tiling local anterior | Local + registros | Mejora |
| --- | ---: | ---: | ---: |
| OpenCL 512² | 0.324 ms | 0.247 ms | 1.31× |
| OpenCL 2048² | 12.092 ms | 6.126 ms | 1.97× |
| Vulkan 512² | 0.477 ms | 0.331 ms | 1.44× |
| Vulkan 2048² | 11.262 ms | 2.930 ms | 3.84× |

En 2048² ambas variantes obtienen error relativo L2 8.099e-7 y absoluto máximo
1.517e-4 frente a float64. Son métricas iguales en estos inputs, sin promesa de
igualdad bit a bit universal. No se habilitó TF32 ni fast-math.

El control de ejecución Vulkan alterna tres procesos de cada versión, con 31
muestras residentes por operación. Suma de 64K elementos pasa de 0.328–0.332 a
0.223–0.235 ms; SAXPY de 0.331–0.382 a 0.218–0.238 ms; energía reducida total de
0.439–0.445 a 0.286–0.324 ms. Los perfiles cuentan 36–41 % menos llamadas Python.
Su tiempo inclusivo contiene esperas: no se interpreta como utilización de CPU.

El control del residuo GEMV intercala fusión/resta separada sobre el mismo plan.
En 2048², OpenCL pasa de 0.163 a 0.077 ms y Vulkan de 0.348 a 0.239 ms. En 8192²,
las ganancias son menores: 0.790→0.739 y 1.008→0.907 ms, respectivamente. La
diferencia máxima entre resultados fue cero en los seis controles 512/2048/8192.
Durante la implementación se corrigió una propagación de `precise` en GLSL que
alteraba la contracción del GEMV previo; hay regresiones de cancelación y NaN/Inf.

## Comparación completa actual

319 configuraciones de rendimiento: 172 operaciones, 63 solvers y 84 de escala.
Cada modalidad tiene siete muestras, con orden alternado entre implementaciones,
sin errores de validación ni referencias ausentes. Se separan primera ejecución,
datos residentes y carga+cálculo+descarga. Los diagnósticos se calculan fuera del
cronómetro, pero pueden afectar cachés/estado térmico entre muestras; el tiempo de
pared también incluye Python, driver y espera. No son tiempos de kernel aislados.

GEMM 2048² float32, medianas; se muestran rutas compiladas de OpenPacho:

| Implementación | Residente | Carga + cálculo + descarga | Error relativo L2 vs float64 |
| --- | ---: | ---: | ---: |
| NumPy, 16 hilos | 29.258 ms | 33.179 ms | 3.077e-7 |
| OpenCL custom | 6.902 ms | 14.848 ms | 8.099e-7 |
| OpenCL + CLBlast | 3.929 ms | 11.830 ms | 8.099e-7 |
| Vulkan | 3.047 ms | 15.759 ms | 8.099e-7 |
| CuPy | 1.419 ms | 10.180 ms | 4.698e-7 |
| cuBLAS directo | 1.429 ms | 10.059 ms | 4.698e-7 |
| nvmath-python | 1.517 ms | 10.229 ms | 4.698e-7 |

Vulkan se acerca en cálculo, pero su descarga/staging sigue pesando. Esto también
se ve en GEMV 8192² float32 (matriz de 256 MiB): OpenCL 0.867/39.867 ms,
Vulkan 1.054/98.134 ms y CuPy 0.852/53.706 ms, residente/completo. NumPy 1 hilo
marca 8.996/8.462 ms. La GPU aprovecha la residencia, pero subir la matriz en cada
llamada invierte la comparación. Vulkan todavía está lejos del coste de transferencia
que logra OpenCL sobre el mismo PCIe: no es correcto atribuir toda esa diferencia al bus.

Solvers completos, 2048 variables float32, `rtol=1e-4`, residuo real cada iteración:

| Implementación | Jacobi, 37 iter. | Descenso más pronunciado, 30 iter. | CG, 13 iter. |
| --- | ---: | ---: | ---: |
| NumPy, 8 hilos | 9.648 ms | 12.722 ms | 8.390 ms |
| OpenCL custom | 8.839 ms | 21.198 ms | 9.120 ms |
| OpenCL + CLBlast | 12.079 ms | 22.438 ms | 11.841 ms |
| Vulkan | 21.533 ms | 41.966 ms | 20.424 ms |
| CuPy | 10.228 ms | 17.757 ms | 10.501 ms |

Los residuos relativos independientes quedan cerca de 8.907e-5 / 9.568e-5 /
7.086e-5. El error de solución relativo respecto a la solución generadora es
aproximadamente 8.874e-5 / 3.490e-4 / 1.712e-4. La solución generadora no es una
referencia exacta del sistema después de redondear el RHS.

La suite anterior registraba Vulkan Jacobi/SD/CG en 2048 float32 a
valores superiores; eso es contexto histórico, no un control causal: esta etapa
también cambia la recurrencia y la norma. Los comparadores NumPy/CuPy ahora usan
la misma recurrencia normalizada; CuPy obtiene la norma del solver mediante
cuBLAS NRM2. No se atribuye el cambio histórico entero al ejecutor.

Hay resultados mixtos que se conservan. La energía compilada Vulkan sobre
2048×32 pasó de 0.712 ms en la suite histórica a 1.663 ms; sobre 32768×32,
de 0.840 a 1.272 ms. Un control adicional del mismo cálculo en tres procesos por
versión y 31 muestras dio 0.369–0.401→0.223–0.236 ms y
0.405–0.420→0.265–0.276 ms, con igual error máximo 1.526e-5 frente a NumPy.
No se reproduce la regresión en aislamiento; tampoco se borra la observación
de la suite mixta ni se considera resuelta su sensibilidad al entorno de medición.

Los 48 casos de preparación también tienen siete muestras por modalidad y
validación. La preparación no asegura acelerar: CG OpenCL float32 2048 da
10.274 ms ordinario y 13.477 ms preparado en esta ronda; Vulkan da 22.124 y
21.737 ms. El contrato de snapshot/validación es útil, pero aún no es una
factorización reutilizable ni un ejecutor autónomo en GPU. Este benchmark usa
`rtol=1e-5`, por lo que no debe mezclarse con los tiempos de la tabla anterior.

## Qué muestran los errores

264 configuraciones adversas, tres muestras por caso: **68 indicadores fuera del
umbral diagnóstico, cero excepciones y cero backends ausentes**. No son 68 bugs
demostrados; el umbral fijo ayuda a investigar y no contempla el condicionamiento
de cada operación. La suite conserva fallos de las referencias y de OpenPacho.

| Caso | Evidencia | Interpretación |
| --- | --- | --- |
| Normas float32 de orden 1e-25/1e25 | OpenPacho: error relativo 4.511e-8 / 2.664e-8; las funciones norm nativas NumPy/CuPy dan cero/Inf | Defecto de algoritmo corregido con escalado; no describe cuBLAS NRM2. |
| Normas y solvers float64 de orden 1e±200 | Los casos de OpenPacho quedan bajo su umbral diagnóstico | No se forman cuadrados extremos ni se exige cambiar dtype. |
| Suma/DOT/GEMM con cancelación adversa | 40 indicadores entre implementaciones y dtypes | El orden de acumulación y precisión importan; compensación o acumulación más ancha son decisiones futuras medibles. |
| CG, Hilbert 8 float32, Vulkan | Residuo relativo Decimal 5.455e-6; error de solución 0.9663 | Cumplir tolerancia residual no garantiza una solución precisa; condición estimada 1.499e9. |
| CG, Hilbert 8 float64, Vulkan | Residuo relativo Decimal 5.487e-12; error de solución 1.498e-4 | Doble precisión ayuda, pero el condicionamiento y criterio de parada siguen importando. |
| Suma del menor subnormal float32 | Vulkan/CuPy: error relativo 1; OpenCL/NumPy lo preservan en este control | Requiere distinguir modo de ejecución, driver y posibles algoritmos alternativos. |

El total se distribuye en 40 casos de cancelación, 18 de Hilbert, 8 de normas
nativas NumPy/CuPy y 2 de suma subnormal. No se activan cambios ocultos de
precisión para hacer desaparecer indicadores.

La consulta local `VkPhysicalDeviceFloatControlsProperties` informa
`shaderDenormPreserveFloat32=false` y `shaderDenormPreserveFloat64=false`.
Así, este driver no ofrece el modo estándar `DenormPreserve 32` para corregir la
suma subnormal. El backend tampoco consulta esos controles hoy; en otro
dispositivo que los anuncie habría una ruta de mejora. No equivale a demostrar
que el silicio no pueda operar subnormales mediante otra API o una emulación.
Ver [propiedades Vulkan](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceFloatControlsPropertiesKHR.html)
y [modos SPIR-V](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_float_controls.html).
La investigación y consulta quedan en `.artifacts/float-controls`.

## Gaps y siguiente prioridad

| Gap | Qué controlamos | Qué limita la mejora | Siguiente paso |
| --- | --- | --- | --- |
| Solvers residentes | Fusión, reducción y ejecución del plan | Cada decisión host conserva una sincronización; no es PCIe de matrices | Medir agrupación de escalares y ejecutor nativo pequeño preservando chequeos por iteración. |
| Transferencias Vulkan | Staging, copias host, asignación y envío | El enlace local Gen3×8 limita payload; no explica toda la diferencia con OpenCL | Perfilar ruta de transferencias grandes y reutilización antes de especializar por familia. |
| GEMM custom vs CLBlast/CUDA | Reutilización de registros, vectorización, unroll y geometría | Compilador/driver, presión de registros, FP64 del hardware | Controles portables de vectorización/unroll; packing solo cuando amortice. |
| Cancelación y condicionamiento | Algoritmo de acumulación, normas, diagnóstico y futura estrategia de solver | Precisión finita e información perdida en inputs | Evaluar compensación y acumulación opcional; no prometer que un residuo certificado implique forward error pequeño. |
| Subnormales Vulkan | Consulta de capacidades y selección explícita de modos donde existan | Este driver no ofrece DenormPreserve32 | Conservar caso de regresión/diagnóstico; investigar alternativas solo con coste/valor medidos. |
| Resultados sensibles al entorno | Protocolo, aislamiento y controles intercalados | Cachés, clocks y planificación del SO | Mantener resultados completos y controles; no convertir ruido aislado en una regla de selección. |

Hardware/SO sigue como [track secundario](hardware-specialization.md). No hay
backend CUDA nuevo ni dispersas, ni selección silenciosa de proveedor. AMD/Intel,
otras versiones del driver y CI remota no se consideran verificadas aquí.

## Verificación y artefactos

- **809 tests pasan, 1 omitido** por requerir dos dispositivos OpenCL. Backends
  GPU obligatorios y CLBlast nativo. Incluye rango extremo, cancelación, límites,
  broadcasting, planes y recursos en vuelo.
- Wheel construido desde sdist e instalado en entorno sin PyOpenCL/Vulkan:
  **249 pasan, 527 omitidos**, importando desde site-packages fuera de `src`.
- Ruff y formato pasan. Distribuciones finales: `.artifacts/dist-numerics-vulkan`.
- JSON y logs completos: `.artifacts/benchmarks/numerics-vulkan`;
  `operations.json`, `solvers.json`, `scale.json`, `prepared.json`, `accuracy.json`.
- Tablas CSV/Markdown con precisión: subdirectorio `report`; casos adversos en
  `accuracy.md`. Figuras: `report/gaps.png` y `report/precision.png`, también SVG.
- Controles de registros/energía: `.artifacts/numerics-vulkan`; controles del
  ejecutor/residuo: `.artifacts/vulkan-execution-next`. Los JSON registran origen.

Los resultados float64 de la suite grande se comparan contra float64: esa parte
no certifica exactitud. Decimal80 se reserva para los casos adversos pequeños y
también tiene precisión finita. Se identifican esas referencias en cada fila.
