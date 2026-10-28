# Extremos residentes y normalización — 2026-10-22

OpenPacho añade `min` y `max`, los alias `amin`/`amax`, métodos de arrays y
dispatch NumPy. Se ejecutan en NumPy, OpenCL, Vulkan y CUDA, tanto eager como
mediante `compile`. Esto permite normalización por ejes y recetas con
desplazamiento por el máximo sin descargar arrays para obtener ese máximo.

La utilidad está ligada al [plan de aplicaciones](application-measurement-plan.md):
normalización, transformaciones previas y futuras redes densas. Las recetas
`softmax` y `logsumexp` viven en `examples/normalization.py`; no se añaden
algoritmos de dominio al runtime ni dependencias obligatorias.

```python
import numpy as np
import openpacho as p
from examples.normalization import softmax, logsumexp


@p.compile
def normalize(x):
    return softmax(x), logsumexp(x)


x = p.asarray([[10000, 10001, 9999]], dtype=np.float32, backend="opencl")
weights, log_total = normalize(x)
host_weights = p.asnumpy(weights)  # observación explícita al final
```

Estas recetas requieren entradas flotantes finitas y un eje no vacío. No hacen
un chequeo de datos mediante descargas; NaN e infinitos siguen la aritmética
ordinaria. `logsumexp` conserva el eje reducido para broadcasting. También se
mantienen los límites existentes de cada backend: por ejemplo, Vulkan no admite
`exp`/`log` float64 en esta implementación GLSL. El ejemplo de verificación usa
float32 en todos los backends.

## Contrato

- `axis=None`, entero positivo/negativo o tupla de ejes; `keepdims` y escalares.
- Dtype de entrada preservado: float32/64, bool e int32/64/uint32/64 donde el
  backend los admite. CUDA sigue limitado a float32/64 y bool.
- Los enteros nunca pasan por una representación flotante para compararse.
- NaN se propaga; los infinitos participan normalmente. La carga útil del NaN
  no tiene una garantía de bits entre backends.
- En ceros de signos distintos, min prefiere `-0` y max prefiere `+0`. Si todos
  los ceros tienen el mismo signo, lo conserva. Esta regla asociativa también
  rige los parciales; puede diferir de la elección de signo de NumPy.
- Reducir un eje vacío falla explícitamente, incluso cuando la salida también
  sería vacía. Una salida vacía con ejes reducidos no vacíos es válida.
  `axis=()` conserva forma, valores y dtype.
- Sin `initial`, `where`, `out`, `nanmin`/`nanmax` ni dtype de acumulación.
  `fuse` continúa siendo solo elementwise; las reducciones usan `compile`.

## Implementación y límites

`_reductions.py` comparte el neutro y la combinación de extremos entre kernels
eager, acumulación por lane, árbol cooperativo y combinación de parciales. No
se reutilizan `fmin`/`fmax` con una semántica distinta para NaN. Los límites
enteros se construyen en su tipo, incluido el mínimo int64 sin overflow.

El IR distingue `sum`, `min` y `max`; la caché de cuerpos Vulkan también incluye
la operación. `explain` la muestra en pasos `reduce`/`map_reduce`. Los productores
elementwise privados elegibles se fusionan con extremos, sin temporal completo.
Productores compartidos o devueltos mantienen las fronteras existentes.

La geometría de las reducciones existentes se conserva. En particular, una
reducción compilada sin productor elementwise puede usar la primitiva eager.
Este lote demuestra funcionalidad y residencia; no promete aceleración ni
compara tiempos con NumPy, CuPy o bibliotecas de redes neuronales.

Las rutas de suma mantienen sus expresiones y reglas anteriores. CUDA habilita
min/max booleanos, pero la suma con acumulador bool continúa sin soporte;
sumar entradas bool con acumulador float32/64 explícito sigue siendo válido.

## Verificación

**267 pruebas dirigidas pasan y cuatro se omiten** por los enteros no admitidos
en CUDA: 48 casos nuevos y 219 regresiones/controles dirigidos. OpenCL, Vulkan y
CUDA fueron obligatorios; no se ejecutó toda la suite ni se reconstruyó el wheel.

La cobertura nueva comprueba ejes, keepdims, dispatch NumPy, captura, parciales
de más de 4096 elementos, NaN/infinito, ceros de ambos signos, extremos enteros,
identidad bool, vacíos, errores explícitos, caché de operaciones y rebinding de
arrays/escalares preparados. El productor `where` seguido de min/max sobre
`(3,8193)` usa dos lanzamientos por llamada en cada GPU: parciales y combinación,
con upload/download prohibidos durante la ejecución.

Softmax y logsumexp se comparan con una referencia float64 para entradas
positivas/negativas de magnitud 10.000, eager y compiladas. Los pesos se mantienen
finitos y suman uno dentro de la tolerancia. Las regresiones cubren sum/mean,
promoción, redondeo antes de acumulación, parciales, salidas compartidas y ABI
escalar del lowerer. El harness de este último ahora prueba las tres operaciones.
QA independiente: **APPROVE**, con 2.072 comprobaciones adicionales en NumPy
sobre dtypes, formas, ejes, vacíos, NaN, ceros y ejecución eager/compilada.

Entorno: Windows, RTX 5060 Ti/driver 617.14, Python 3.12.14, NumPy 2.5.3.
Se usa `--capture=sys`, TEMP accesible y compilador GLSL explícito. El QA también
ejecutó accidentalmente una selección GPU; no coincidió con otras pruebas o
mediciones GPU activas. Los informes están en
`.artifacts/iterations-2026-10-22/extrema-*.xml`: 202 regresiones aprobadas en la
primera ejecución y 65 pruebas finales tras corregir CUDA y actualizar el harness.
Se conserva también la evidencia de los fallos iniciales; no se ocultan mediante
omisiones. El runtime mantiene NumPy como única dependencia base.
