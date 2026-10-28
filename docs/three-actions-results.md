# Binding, contexto CUDA y productos vectoriales — 2026-09-18

Esta iteración compara tres cambios contra el wheel final de la ronda anterior.
La mayor ganancia esperada corresponde al DOT largo custom, que antes acumulaba
en un solo hilo. El binding es una mejora transversal; la gestión de contexto
tiene un efecto menor y dependiente del plan. Son comparaciones con la versión
anterior de OpenPacho, no una afirmación de superar CuPy, cuBLAS o NumPy.

## Implementación

1. **Binding posicional del compilador.** Los nombres y la elegibilidad de la firma
   se resuelven al construir la función compilada. Si llegan todos los parámetros
   por posición, se evita reconstruir `inspect.BoundArguments`. Los kwargs,
   defaults omitidos y parámetros keyword-only conservan la ruta general.
   Permanecen la validación, la LRU, las claves de forma/dtype/backend/estáticos,
   el reenlace escalar y los errores. No exige `prepare` ni cambiar el algoritmo.
   La primera captura de una variante sí enlaza la firma; los hits evitan ese trabajo.
2. **Contexto CUDA anidado y scope selectivo.** El lock protege la profundidad
   de activación. Una llamada anidada consulta el contexto actual del hilo antes
   de omitir push/pop; si otra biblioteca lo cambió, activa/restaura el contexto
   correspondiente. Se restaura también ante excepciones. El compilador resuelve
   `execution_scope_min_steps` una vez: CUDA solicita al menos dos pasos; Vulkan
   conserva su comportamiento. No se añaden esperas ni se cambia el límite de
   64 operaciones pendientes. La gestión sigue el
   [contrato del Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
3. **GEMV/DOT cooperativos CUDA custom.** `_cuda_gemv.py` adapta el algoritmo
   OpenCL: 128 hilos leen términos de una fila y reducen en memoria compartida.
   Admite filas de matrices y lotes contiguos, float32/64 e índices irregulares.
   DOT de más de 4096 términos utiliza parciales de 1024 y una reducción GPU.
   La ruta se selecciona para vector derecho y K >= 256; las filas menores
   conservan el kernel anterior. CuBLAS conserva su proveedor explícito.

El primer ensayo habilitaba cooperación desde K=64 y scope en todos los planes.
La preparación adicional no compensó en varias operaciones cortas. Se descartó
esa selección indiscriminada; K=256 y dos pasos son criterios conservadores
locales, no un crossover universal ni autotuning por dispositivo.

## Método y trazabilidad

Windows, Python 3.12, RTX 5060 Ti, driver 616.64. Se ejecuta un backend por proceso,
con 31 muestras por alcance, orden de variantes rotado/invertido y dos semillas.
Entre procesos se exige reposo observado: al menos diez segundos, tres lecturas
consecutivas, utilización de GPU/memoria <=5%, temperatura <=52 °C y dentro del
techo relativo de reposo. Cada variante se calienta fuera del reloj durante al
menos 0,25 segundos y tres llamadas. No garantiza estado térmico estacionario.

`benchmarks.three_actions` mide baseline, binding aislado, contexto aislado,
GEMV aislado y todos juntos cuando aplica. Cada variante tiene su propio plan,
para que el scope enlazado durante captura sea correcto. Seleccionar el control
queda fuera del reloj. El control carga solo los métodos anteriores desde un
wheel local, preservando las identidades de clases/allocations actuales y
verificando los otros módulos de runtime mediante hashes.

El alcance **residente** incluye llamada Python y sincronización, con entradas
ya subidas. **Completo** añade subida, preparación y descarga. La primera llamada
se registra separada. Ridge conserva veinte iteraciones de la misma función de
dominio, no representa tiempo hasta convergencia. No se modifican los solvers.

Los perfiles con eventos y contadores son diagnósticos instrumentados aparte:
sus duraciones son inclusivas y no se suman ni se restan del tiempo total para
deducir un supuesto coste puro de Python o de GPU.

## Resultados medidos

Las ocho ejecuciones válidas contienen **660 configuraciones y 492 comparaciones
contra baseline**, con 31 muestras por alcance. Las cifras siguientes agregan
sin ponderar las reducciones porcentuales de las medianas por caso y ronda;
no representan una mezcla universal de aplicaciones.

| Cambio aislado / backend | Menos tiempo residente | Menos tiempo completo |
| --- | ---: | ---: |
| Binding, OpenCL custom | 8,05% | 2,34% |
| Binding, Vulkan | 4,21% | 2,09% |
| Binding, CUDA/cuBLAS | 5,85% | 1,21% |
| Binding, CUDA custom | 7,96% | 2,25% |
| Todos los cambios, CUDA/cuBLAS | 6,32% | 1,32% |
| Todos los cambios, CUDA custom | 8,60% | 2,04% |

Para ver el cambio matricial sin diluirlo entre operaciones que no usan GEMV:

| CUDA custom, todos los cambios | Antes residente | Ahora residente | Antes completo | Ahora completo |
| --- | ---: | ---: | ---: | ---: |
| DOT 1.048.577, f32 | 15,96–16,69 ms | 0,198–0,201 ms | 12,70–12,75 ms | 1,674–1,686 ms |
| DOT 1.048.577, f64 | 54,89–55,28 ms | 0,188–0,190 ms | 32,96–33,05 ms | 2,923–2,939 ms |
| GEMV 129×8193, f32 | 0,205–0,207 ms | 0,077–0,078 ms | 1,049–1,053 ms | 0,923–0,925 ms |
| GEMV 129×8193, f64 | 0,597–0,598 ms | 0,086–0,101 ms | 1,983–2,046 ms | 1,562–1,601 ms |

Los intervalos son las dos medianas observadas, no intervalos de confianza.
Los alcances se calientan/miden por separado: no se debe restar residente de
completo para estimar transferencia; relojes y estado del dispositivo pueden
diferir entre alcances. Las grandes razones de mejora corresponden a reemplazar
nuestro DOT serial anterior, no a una victoria sobre otra biblioteca.

El scope aislado ahorra una mediana de **2,41% custom / 2,62% cuBLAS** en los
casos de varios pasos seleccionados (ridge, loss, covarianza y centrado). En el
agregado completo es marginal: −0,74% custom / +0,26% cuBLAS residente. En ridge
cuBLAS f32, el diagnóstico reduce los pares push/pop de **243 a 25** por llamada;
persisten las consultas del contexto actual y otros costes del ejecutor.

Las fronteras cortas K=256/257 todavía presentan resultados mixtos: la reducción
cooperativa y su preparación pueden añadir unos microsegundos. El umbral no
garantiza una mejora para cada forma. Ridge tampoco mejora uniformemente: en la
segunda ronda custom f32 el tiempo completo pasó de 5,85 a 7,46 ms, mientras que
el residente bajó de 4,97 a 4,88 ms; f64 residente pasó de 6,60 a 6,68 ms. Estos
datos permanecen en el informe aunque contradigan la mejora agregada.
Se intentó un seguimiento de ridge f32 con solo baseline/selección final y 51
pares por alcance, pero no llegó a medir: la actividad de memoria superó el gate
de reposo durante los 300 segundos permitidos. `ridge-followup.json` conserva la
telemetría y el estado incompleto. La regresión puntual sigue sin explicación
aislada; no se considera ruido demostrado ni una mejora confirmada de ridge.

## Precisión y selección

No cambian dtypes, `fmad=false`, `ftz=false`, división/raíz precisas ni el modo
cuBLAS. Binding y contexto conservan resultados exactamente iguales al control
en las comparaciones. GEMV/DOT cambian el orden de suma: no se promete identidad
bit a bit ni una mejora numérica universal.

El caso float32 129×8193 de la semilla 941 hizo visible un error previo: el kernel
serial excede rtol=5e-5/atol=5e-6 frente a NumPy y a la referencia float64. La nueva
reducción pasa el mismo umbral. Se añadió una regresión con esos inputs; el runner
conserva los indicadores de fallo de cada variante histórica, sin ampliar tolerancias
ni ocultarlos. Para float64, NumPy float64 es una comparación de igual precisión,
no un oráculo de mayor precisión.

Todas las **168 configuraciones con la selección final** pasan los umbrales
registrados frente a ambas referencias. Los tres fallos registrados son el mismo
caso serial anterior en baseline/binding/contexto, no tres problemas distintos.
El máximo error relativo L2 de las salidas seleccionadas es 4,36e-7 en f32 y
2,83e-14 en f64. No equivale a una garantía para cualquier input.

En GEMV 129×8193 f32, el error relativo de la semilla 941 baja de 1,59e-6 a
1,84e-7. Hay un contraejemplo a «paralelizar siempre mejora precisión»: el DOT
largo f64 de la semilla 942 pasa de 2,93e-16 a 2,11e-14 frente a NumPy float64;
ambos cumplen la tolerancia, pero la nueva suma se aleja más de esa referencia.

## Límites y siguiente prioridad

La reducción del coste de contexto es marginal en el agregado y no es un techo
físico. No merece desplazar kernels o copias con mayor margen. La vía posicional
tampoco acelera llamadas por nombre ni elimina la validación por invocación.
Los casos cortos siguen dominados por Python/driver; las transferencias siguen
limitando el beneficio completo. No hay pinned memory, solapamiento ni graphs
CUDA nuevos en esta ronda.

La siguiente prioridad matricial es GEMV/DOT específicos de cuBLAS y aplicaciones
con transpuestas reutilizables; el proveedor actual todavía usa GEMM para casos
vectoriales. Las filas cortas y la geometría cooperativa necesitan más mediciones
antes de autotuning. Para Vulkan, la mejora presente es el binding común; no se
atribuye a un nuevo kernel ni se declara agotado el margen del ejecutor.

## Verificación

La suite completa da **1.250 pruebas pasando y una omitida** por necesitar un
segundo dispositivo OpenCL, con OpenCL/Vulkan/CUDA obligatorios y CLBlast nativo.
Otras **89 pruebas** validan los benchmarks. El wheel construido desde sdist pasa
**391 pruebas CPU, con 860 omisiones GPU**, cargando el runtime empaquetado y
comprobando independencia de imports opcionales. Ruff/formato y sintaxis Python
3.10 pasan. Ver [detalle de verificación](verification.md).

## Reproducción

Los datos locales quedan en `.artifacts/three-actions`, incluyendo intentos
incompletos. `paired/` fue detenido por la comparación numérica del baseline;
`paired-v2/` es exploratorio y termina con un error del runner al intentar cerrar
OpenCL, que no expone `close()`. La limpieza ahora usa `synchronize()` en ese caso
y guarda el informe incluso si limpiar falla; ambas rutas tienen pruebas CPU.

`final-paired/` contiene la primera ejecución CUDA custom seleccionada. La tanda
se detuvo al no alcanzar reposo antes del siguiente backend; no se relajó el gate.
`remaining/` y `custom-repeat/` completan las mediciones con espera máxima de 300
segundos, conservando todos los criterios de reposo. Un ajuste posterior solo de
formato en `_cuda_gemv.py` tiene equivalencia AST comprobada y registrada.

Para repetir desde un wheel local anterior compatible y una carpeta nueva:

```powershell
python -m benchmarks.isolated --worker three_actions --backends cuda-custom,cuda-cublas,opencl,vulkan --rounds 2 --repeats 31 --max-temperature 52 --max-wait-seconds 300 --control-wheel .artifacts/three-actions/before.whl --output .artifacts/three-actions/reproduction
```

El umbral de reposo, las muestras, el error, el dispositivo y los hashes quedan
en JSON. La asociación automática backend/UUID y otros fabricantes/SO siguen
pendientes. La versión continúa en 0.1.0a1 sin publicación.
