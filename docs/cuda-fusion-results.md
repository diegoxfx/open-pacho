# Fusión CUDA y selección del siguiente margen de mejora

Medición local del 2026-09-13, Windows, Python 3.12 y RTX 5060 Ti, driver 616.64.
La fusión reduce aproximadamente a la mitad la ejecución de los veinte pasos
de ridge frente al plan CUDA anterior, sin cambiar el algoritmo. No produce una
mejora uniforme de extremo a extremo ni paridad general con CuPy.

## Por qué rotó la prioridad

La última optimización de [finalización Vulkan](routing-completion-results.md)
dio mejoras pequeñas o mixtas. Eso no agota el margen de Vulkan: sus costes de
envío y transferencia siguen presentes. Sin embargo, CUDA conservaba un coste
eliminable más directo: cada nodo elementwise del plan lanzaba una primitiva y
materializaba su resultado. La fusión ya probada en otros backends tenía mayor
potencial inmediato que seguir ajustando la misma espera Vulkan.

El criterio es frecuencia de uso, coste eliminable, mejora residente/completa,
repetibilidad, precisión y esfuerzo. Se rota entre frentes de optimización, no
se declara terminado un backend por una ronda marginal. La corrección prevalece
si aparece una regresión funcional.

## Implementación

[`CUDALowering`](../src/openpacho/_cuda_lowering.py) implementa la factoría
`backend.create_lowering()` ya existente. La selección sucede al preparar el
plan; el compilador común no añade condiciones por nombre de backend.

- Los grupos elementwise se convierten en un kernel NVRTC. Broadcasting,
  escalares por valor y los redondeos de cada operación conservan su contrato.
- Map-reduce evalúa la expresión dentro de la reducción y guarda solo parciales.
  Mantiene el árbol CUDA previo: 128 hilos, bloques de hasta 4096 términos y
  reducción posterior de parciales cuando hace falta.
- El payload preparado conserva código/geometría, sin retener entradas ni valores
  escalares de llamadas anteriores. La gestión de dependencias, salidas públicas
  y temporales compartidos sigue en el compilador común.
- Matmul continúa usando el proveedor explícito custom/cuBLAS. No se fusiona
  dentro de cuBLAS ni se introduce un epílogo GEMV CUDA en esta etapa.

`@pacho.compile`, `pacho.compile(función)` y `pacho.fuse` acceden al lowerer.
La ejecución eager conserva su comportamiento. No cambian los algoritmos de
covarianza/ridge ni los solvers externos; tampoco las dependencias opcionales.
CUDA aún carece de normas escaladas, acumuladores bool y graphs.

Se conservan `--fmad=false`, `--ftz=false`, división y raíz precisas; la fusión
no autoriza fast-math. Las barreras de reducción son alcanzadas por todos los
hilos, incluidos los de las colas y reducciones vacías. Contratos consultados:
[opciones NVRTC](https://docs.nvidia.com/cuda/nvrtc/) y
[sincronización de bloques CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

## Protocolo y control

[`benchmarks/fusion.py`](../benchmarks/fusion.py) mide nueve casos en float32/64:
cadenas de 33 y 1.048.577 elementos; energía 1025×1023; energía por filas 257×129
y 17×4099; centrado por filas 2049×33; pérdida cuadrática 2049×1023; covarianza
1025×65; y ridge 2049×129 durante veinte pasos idénticos. Se usan operadores y
métodos comunes del array, con las mismas entradas por semilla.

El control CUDA alterna el orden de dos modos por caso:

- **primitive-plan:** prepara sin lowerer, como el wheel anterior.
- **fused-plan:** prepara con `CUDALowering`, en la misma instancia y contexto.

El runner verifica por AST que `_run`, `_compile`, `_activate`, `empty`,
`elementwise`, `reduce`, `matmul` y `synchronize` son iguales a los del wheel
anterior. SHA256 de su módulo CUDA:
`3a87b365f08bc8dbd7dfd58fa3be5e95acc0d39ba628790d3aefb43fc6a3265f`.
Es un control de lowering con las mismas primitivas, no una ejecución integral
del paquete antiguo. La factoría se restaura incluso ante una excepción.

Los procesos son secuenciales, con orden inicial aleatorio y reverso en la
segunda ronda. Antes de cada proceso se exigen al menos diez segundos, tres
lecturas consecutivas con utilización GPU/memoria ≤5 %, temperatura dentro de
2 °C de la referencia y ≤52 °C, sin ralentización térmica observada. Después
hay calentamiento de al menos tres llamadas y 0,25 s por modo/alcance, fuera de
los relojes. Se sincroniza antes y después de cada muestra.

Hay 21 muestras por configuración y alcance, dos rondas CUDA y dos por cada
referencia GPU. **144 configuraciones CUDA y 218 de referencias** completaron
la medición. NumPy a un hilo solo tiene una ronda: la segunda no alcanzó el
reposo exigido en 120 s y no se ejecutó. No se relajó el criterio. Las ventanas
aceptadas terminaron a 43–47 °C, tras esperas de 10,4–24,9 s. Esto no garantiza
clocks constantes ni elimina actividad del sistema durante las muestras.

El alcance **residente** excluye subida, preparación de entradas y descarga;
ridge reutiliza la transpuesta preparada. El **completo** incluye subida,
preparación, cálculo y descarga, con kernels/planes ya calentados. En ambos
se mide tiempo de pared, incluidos Python, driver y esperas. La primera llamada
queda separada en los JSON; puede reutilizar kernels compilados por otros casos
del proceso y no representa siempre un arranque frío independiente.

## Antes y después

Milisegundos; cada celda muestra las medianas de **ronda 1 / ronda 2**. Los
JSON y CSV conservan todas las muestras, p10/p90, primera llamada y errores.

| CUDA custom, float32 | Residente anterior | Residente fusionado | Completo anterior | Completo fusionado |
| --- | ---: | ---: | ---: | ---: |
| Cadena 1.048.577 | 0,292 / 0,290 | 0,098 / 0,086 | 2,650 / 2,639 | 2,432 / 2,426 |
| Energía 1025×1023 | 0,529 / 0,374 | 0,192 / 0,168 | 1,931 / 1,905 | 1,957 / 1,681 |
| Energía por filas 257×129 | 0,281 / 0,353 | 0,096 / 0,083 | 0,531 / 0,540 | 0,539 / 0,334 |
| Pérdida cuadrática | 0,429 / 0,439 | 0,209 / 0,211 | 1,964 / 1,958 | 1,730 / 1,729 |
| Covarianza | 0,559 / 0,550 | 0,708 / 0,442 | 0,733 / 0,743 | 0,850 / 0,850 |
| Ridge, veinte pasos | 12,149 / 12,316 | 5,948 / 5,806 | 12,913 / 12,851 | 6,555 / 6,666 |

Ridge custom float64 pasa de **12,133 / 13,135** a **6,016 / 6,116 ms** residentes;
el completo pasa de 12,968 / 14,126 a 6,770 / 7,252 ms. Con cuBLAS, ridge float32
pasa de **12,592 / 12,566** a **6,422 / 6,261 ms** residentes. Su completo es más
variable: 15,243 / 13,089 frente a 10,374 / 6,858 ms. No se oculta esa variación.

La energía y las cadenas muestran que eliminar lanzamientos puede ahorrar mucho
con datos residentes y poco cuando predominan las copias. No se interpreta la
diferencia entre alcances como una medición aislada del coste PCIe.

### Seguimiento de covarianza

La pérdida completa de custom float32 en ambas rondas motivó otro control,
con el mismo contexto, 31 pares alternados por alcance y tres bloques. Superó
el mismo criterio de reposo tras 10,4 s, a 47 °C. No cambió el runtime:

| Bloque | Residente anterior → fusionado | Completo anterior → fusionado |
| --- | ---: | ---: |
| 1 | 0,561 → 0,440 ms | 0,747 → 0,633 ms |
| 2 | 0,963 → 0,703 ms | 0,761 → 0,636 ms |
| 3 | 0,555 → 0,444 ms | 0,749 → 0,635 ms |

La pérdida inicial **no se reprodujo en este control alternado**. La dispersión
sigue siendo visible, especialmente en el segundo bloque. Los datos iniciales
permanecen en el informe; no se atribuye su causa a un kernel ni se añade una
excepción basada en el nombre del workload. Covarianza custom float64 y cuBLAS
mejoraron también en el control principal.

## Referencias y gaps que permanecen

Milisegundos residentes, float32, medianas de las dos rondas. Esta tabla compara
las rutas medidas; no es un ranking universal ni una cota física del hardware.

| Implementación | Cadena 1.048.577 | Energía 1025×1023 | Ridge 2049×129 |
| --- | ---: | ---: | ---: |
| NumPy, 8 hilos | 1,813 / 1,869 | 1,616 / 1,585 | 0,789 / 0,797 |
| OpenCL custom, compilado | 0,133 / 0,130 | 0,114 / 0,112 | 5,464 / 5,290 |
| OpenCL CLBlast, compilado | 0,129 / 0,155 | 0,131 / 0,114 | 6,387 / 6,564 |
| Vulkan, compilado | 0,196 / 0,171 | 0,224 / 0,232 | 7,624 / 4,983 |
| CUDA custom, fusionado | 0,098 / 0,086 | 0,192 / 0,168 | 5,948 / 5,806 |
| CUDA cuBLAS, fusionado | 0,086 / 0,089 | 0,161 / 0,175 | 6,422 / 6,261 |
| CuPy eager | 0,085 / 0,082 | 0,100 / 0,104 | 5,211 / 4,700 |
| CuPy fuse | 0,029 / 0,046 | 0,411 / 0,413 | — |

[`cupy.fuse`](https://docs.cupy.dev/en/stable/user_guide/kernel.html#kernel-fusion)
se compara con las mismas funciones elementwise/reducción que admite esta suite.
En las cadenas ayuda; en esta energía grande es peor que CuPy eager. Ambos modos
se conservan. Ridge y covarianza usan la ruta eager CuPy común; no se afirma que
representen su mejor implementación posible.

En completo, NumPy de ocho hilos sigue ganando ridge en este tamaño:
**1,141 / 1,315 ms**, frente a 6,555 / 6,666 ms de CUDA custom y 5,179 / 4,957 ms
de CuPy. La energía completa OpenCL custom da 1,565 / 1,571 ms y CuPy eager
1,509 / 1,502 ms; las grandes ganancias residentes no se trasladan por entero.
La cadena de solo 33 elementos sigue siendo claramente de CPU: aproximadamente
2–3 μs, frente a 85–86 μs CUDA custom fusionado y 23 μs CuPy fuse.

No se vuelve a medir nvmath Matmul en esta ronda: la implementación de matmul no
cambió. Continúa como referencia de los informes matriciales. No se incorporaron
PyTorch/JAX ni se modificaron los solvers para favorecer un backend.

## Memoria, lanzamientos y error numérico

El diagnóstico se ejecuta fuera de las muestras de tiempo. Cuenta lanzamientos
custom y bytes solicitados mediante `empty`; **no** mide memoria máxima viva,
asignaciones nuevas del driver ni lanzamientos internos de cuBLAS.

| CUDA custom float32 | Lanzamientos anteriores → fusionados | Bytes solicitados anteriores → fusionados |
| --- | ---: | ---: |
| Energía 1025×1023 | 4 → 2 | 8.389.628 → 1.028 |
| Covarianza 1025×65 | 6 → 5 | 567.320 → 567.060 |
| Ridge, veinte pasos | 140 → 80 | 379.440 → 348.480 |

En energía se evita materializar dos matrices completas; quedan 256 parciales
float32 y una salida. Ridge gana principalmente por menos operaciones/envíos,
no por eliminar grandes transferencias. En cuBLAS, los kernels propios de ridge
pasan de 100 a 40, además de los cuarenta productos BLAS que conserva el algoritmo.

Se compara con la expresión NumPy calculada en float64. Para salidas float64,
esa referencia es **de igual precisión**, no una solución exacta. Máximo error
relativo L2 observado sobre todos los casos de cada grupo:

| Ruta actual | float32 | float64 |
| --- | ---: | ---: |
| CUDA custom fusionado | 4,36×10⁻⁷ | 8,20×10⁻¹⁶ |
| CUDA cuBLAS fusionado | 8,37×10⁻⁸ | 3,19×10⁻¹⁶ |
| OpenCL custom | 4,34×10⁻⁷ | 8,37×10⁻¹⁶ |
| Vulkan | 4,31×10⁻⁷ | 8,27×10⁻¹⁶ |
| CuPy, ambos modos | 1,80×10⁻⁷ | 3,06×10⁻¹⁶ |

Estos máximos no son garantías para otras entradas. El máximo absoluto CUDA
float32 es 0,00489, correspondiente a la escala de resultados acumulados; ambos
errores se conservan por caso. Tests adicionales verifican cancelación, subnormales,
NaN/Inf, cero con signo, estrechamiento float64→float32→float64, promoción antes
de cada operación y ausencia de FMA oculta. La reducción ordinaria sigue sin ser
una norma estable. Los veinte pasos de ridge no miden tiempo hasta convergencia.

## Próxima selección

1. **Preparación/envío CUDA y reducciones por eje.** La cadena fusionada aún
   tarda aproximadamente 2–4 veces lo de CuPy fuse. Separar preparación de
   argumentos, activación del contexto, lanzamiento y GPU permitirá escoger el
   siguiente cambio. Evitar asumir que todo el hueco es tiempo de kernel.
2. **Materialización por coste.** La fusión de una operación pequeña con una
   salida grande por broadcasting puede repetir trabajo. Medir expresiones y
   geometrías, no introducir reglas por nombre de aplicación.
3. **Vulkan y kernels matriciales.** Mantener la investigación de envío/copias,
   GEMV/DOT y layouts. Su prioridad sube cuando un perfil muestre más ahorro
   eliminable que los frentes anteriores; no seguir microajustando una espera
   por inercia. Las regresiones verificadas se atienden antes de ampliar alcance.

## Reproducción y verificación

Desde un entorno con los extras instalados y rutas locales configuradas para
CLBlast/glslang, usando una carpeta de salida nueva:

```powershell
python -m benchmarks.isolated --worker fusion --backends cuda-custom,cuda-cublas --rounds 2 --repeats 21 --max-temperature 52 --control-wheel .artifacts/cuda-fusion/before.whl --output .artifacts/cuda-fusion/control
python -m benchmarks.isolated --worker fusion --backends numpy-1t,numpy-8t,opencl,opencl-clblast,vulkan,cupy --rounds 2 --repeats 21 --max-temperature 52 --output .artifacts/cuda-fusion/references
```

El wheel de control es un artefacto local anterior, no una versión publicada.
El paquete registra hashes de fuentes; los manifiestos conservan política,
lecturas, comandos y fallos de reposo. La asociación automática entre selección
del backend y UUID de telemetría continúa pendiente. La GPU observada fue
`GPU-16858217-5c55-717b-c7dc-045e1345b3de`; no se extrapola a otros dispositivos.

Artefactos en `.artifacts/cuda-fusion`: `summary.csv` reúne las 362 configuraciones;
`control/` y `references/` conservan los JSON originales. El seguimiento alternado
está separado en `covariance-control.json` y su script, con seis configuraciones
adicionales. `environment.json` registra versiones y la comprobación de que los
algoritmos de aplicaciones/solvers no cambiaron respecto al sdist anterior.
La [verificación actual](verification.md) registra pruebas y wheel.
