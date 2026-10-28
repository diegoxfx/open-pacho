# Aplicaciones de ejemplo

OpenPacho ofrece operaciones de arrays/matrices en GPU. Los algoritmos de dominio
viven aquí y no se instalan en el wheel ni se exportan como `openpacho.solvers`.
El repositorio y el sdist incluyen los ejemplos para reproducir pruebas y benchmarks.

Desde la raíz del repositorio, después de instalar el paquete:

```sh
python examples/numpy_migration.py --backend numpy
python -m examples.solve_system --backend numpy
python -m examples.solve_system --backend opencl
```

`solvers.py` contiene CG, Jacobi, descenso más pronunciado y `PreparedSystem`.
Usa la API pública, incluido `get_backend(array)` para conservar exactamente el
contexto/proveedor de los inputs. CUDA admite estos ejemplos mediante sus normas
estables float32/64, tanto eager como compiladas y con los proveedores custom y
cuBLAS. No existe una capacidad «solver» en el runtime; los algoritmos permanecen
en este directorio.

Se conservan las recurrencias normalizadas, validaciones, tolerancias y residuo
real en cada iteración de la versión anterior. CG/descenso requieren matrices
SPD; Jacobi exige diagonal no nula e iteración convergente. No se demuestran esas
propiedades ni se promete exactitud de la solución a partir de un residuo pequeño.

Estos ejemplos ya contienen fases `compile`/`prepare` explícitas. Los benchmarks
deben declarar esa preparación y mantener comparaciones eager y compiladas cuando
proceda; no son una prueba de que cualquier código NumPy gane sin adaptación.
Las optimizaciones nuevas deben ser reutilizables desde operaciones ordinarias
en el compilador/backend, sin reconocer el nombre de un solver. Un cambio
algorítmico se evalúa por separado y se aplica también a las referencias pertinentes.

`normalization.py` muestra softmax y logsumexp con desplazamiento por el máximo,
usando min/max y las operaciones existentes sin transferencias intermedias.
Las recetas requieren entradas flotantes finitas y un eje no vacío; no validan
datos mediante descargas. `logsumexp` conserva el eje reducido para broadcasting.
Pueden capturarse con `compile` y respetan los límites de funciones/dtypes de cada
backend; no son una API de redes neuronales en el runtime. Ver los
[resultados de extremos](../docs/extrema-results.md).

`dense_network.py` contiene `predict`, `loss_and_gradients` y `sgd_step`: dos capas
afines, activación tanh y pérdida cuadrática media dividida por dos. Los gradientes
son explícitos y el paso devuelve pérdida previa y nuevos parámetros, sin modificar
los originales. Las funciones admiten NumPy directo y arrays OpenPacho/CuPy; el
caller proporciona arrays flotantes del mismo dtype y backend. Se valida metadato,
sin descargar datos. Tanh float64 continúa sin soporte en Vulkan.

```python
import numpy as np
import openpacho as p
from examples.dense_network import predict, sgd_step

rng = np.random.default_rng(47)
shapes = ((31, 7), (31, 3), (7, 11), (11,), (11, 3), (3,))
host = tuple((0.2 * rng.standard_normal(shape)).astype(np.float32) for shape in shapes)
x, y, w1, b1, w2, b2 = (p.asarray(a, backend="opencl") for a in host)
step = p.compile(sgd_step).prepare(x, y, w1, b1, w2, b2, 0.1)
loss, w1, b1, w2, b2 = step(x, y, w1, b1, w2, b2, 0.1)
prediction = predict(x, w1, b1, w2, b2)
print(loss.item())  # observación explícita de la pérdida anterior al paso
```

Las diferencias finitas verifican todos los parámetros del caso pequeño. El
[informe de red densa](../docs/dense-network-results.md) separa inferencia, un paso
SGD, preparación, residencia y transferencias. No constituye un framework de
entrenamiento ni promete convergencia con cualquier tasa.

`oscillators.py` implementa velocity-Verlet para osciladores acoplados con masas
unitarias, `q'' = -K @ q`. La matriz de rigidez debe ser simétrica definida positiva;
el caller proporciona esa propiedad y un paso estable. `verlet_step` devuelve
posición, velocidad y aceleración nuevas; `integrate` conserva la aceleración
entre pasos y ejecuta exactamente `steps + 1` productos matriz-vector. `energy`
devuelve un escalar residente para diagnósticos fuera del bucle.

```python
import numpy as np
import openpacho as p
from examples.oscillators import integrate, verlet_step

K, q, v = (
    p.asarray(a, backend="opencl")
    for a in (
        np.array([[2.0, -0.5], [-0.5, 1.0]], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.25], dtype=np.float64),
    )
)
dt = 1 / 32
step = p.compile(verlet_step).prepare(K, q, v, -(K @ q), dt)
q, v, a = integrate(K, q, v, dt, 64, step=step)
```

Se prepara un paso y se conserva el bucle Python; compilar el integrador completo
desenrollaría sus pasos. El ejemplo no consulta escalares ni descarga arrays en
cada iteración. Su [informe](../docs/oscillators-results.md) distingue error frente
a la recurrencia discreta, error temporal frente a la solución modal y energía.
