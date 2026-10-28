# Preparación de argumentos CUDA

Ronda local del 2026-09-13, Windows / Python 3.12 / RTX 5060 Ti, driver 616.64.
Se conserva la fusión anterior y se reduce el trabajo host de cada lanzamiento.
En las 72 comparaciones controladas mejora el tiempo residente: la mediana de
las mejoras porcentuales por caso es **13,9 %**. En completo esa mediana es
**3,3 %**, con nueve resultados peores. No equivale a una aceleración universal.

## Decisión y cambio

El perfil de la [etapa anterior](cuda-fusion-results.md) señalaba preparación y
envío como frentes pendientes. Un perfil adicional confirmó que `_run` seguía
dedicando tiempo a construir arrays NumPy de escalares/punteros en cada llamada.
El cambio está acotado a ese trabajo, sin modificar kernels ni algoritmos.

[`KernelArguments`](../src/openpacho/_cuda_arguments.py) crea una vez el vector
`void**` y el almacenamiento host de sus valores. Cada kernel compilado conserva
su vector; en cada lanzamiento se actualizan los valores bajo el lock del backend.
No se guardan referencias a arrays o escalares del llamador. Las asignaciones GPU
siguen retenidas por `_pending` hasta completar el stream, independientemente
del almacenamiento de argumentos. La LRU de 128 módulos y `close()` liberan
también estas plantillas.

Se preservan tamaños ABI, ceros con signo y la resolución de tipos previa. Los
punteros de dispositivo usan 64 bits; los punteros host del vector usan el tamaño
nativo. Un cambio inesperado de cantidad o tipo de parámetros falla antes de
lanzar. La validación detectó que NumPy 2.5.3 no permite asignar `np.bool_` a
`ctypes.c_ubyte`: la versión final usa `ctypes.c_bool`, con representación de un
byte comprobada. Esto no modifica las rutas float32/64 medidas.

El driver copia los valores apuntados por `kernelParams` durante el lanzamiento,
por lo que la siguiente llamada puede reutilizar ese almacenamiento. Se mantiene
separada la vida del buffer GPU apuntado.
[Contrato de parámetros CUDA](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html),
[preparación en CUDA bindings](https://nvidia.github.io/cuda-python/cuda-bindings/13.3.2/overview.html).

No cambian los contextos, el stream, la frecuencia de sincronización, las copias,
la política matemática ni la elección custom/cuBLAS. Los parámetros alpha/beta
de las llamadas cuBLAS conservan su implementación anterior. El API público y
los extras de instalación permanecen iguales.

## Control de medición

[`benchmarks/cuda_dispatch.py`](../benchmarks/cuda_dispatch.py) compara el `_run`
del wheel anterior con el actual, en **la misma instancia/contexto y con la misma
función compilada**. Verifica los otros métodos CUDA por AST, salvo `_compile`,
cuyo cambio añade el campo de almacenamiento a la caché. Los demás módulos de
runtime, incluidos generadores y lowerer, se comprueban por hash contra el wheel.
No se ejecuta íntegramente el paquete histórico: el control aísla empaquetado.

Se reutilizan los nueve casos de fusión, en float32/64: cadenas de 33 y 1.048.577
elementos, energía 1025×1023, energía por filas 257×129 y 17×4099, centrado por
filas 2049×33, pérdida cuadrática 2049×1023, covarianza 1025×65 y ridge 2049×129.
Ridge mantiene los mismos veinte pasos; no mide tiempo hasta convergencia.

- Dos rondas por proveedor, **31 pares alternados** por caso/alcance. El orden
  de los procesos se invierte en la segunda ronda y el de las variantes cambia
  en cada par. Primera llamada y calentamiento quedan fuera de las muestras.
- Residente: entradas/preparación ya disponibles, cálculo y sincronización.
  Completo: subida, preparación, cálculo y descarga, con kernels/planes calientes.
  Ambos son tiempos de pared con Python/driver/esperas, no solo ejecución GPU.
- Reposo observado antes de cada proceso: al menos 10 s, tres lecturas con
  utilización GPU/memoria ≤5 %, temperatura dentro de 2 °C de la referencia y
  ≤52 °C, sin ralentización térmica observada. Después, calentamiento de al menos
  tres llamadas y 0,25 s por modo/alcance. Las ventanas aceptadas terminan a
  46–48 °C tras 10,4 s. No garantiza clocks constantes durante las muestras.
- **144 configuraciones CUDA y 92 referencias** completadas. Las referencias
  actuales son NumPy de ocho hilos y CuPy eager/fuse. OpenCL/Vulkan se verifican
  funcionalmente; sus tiempos anteriores se conservan como históricos.
- Diagnóstico con instrumentación y eventos separado de las muestras de tiempo.
  Se conservan todos los samples, p10/p90, primera llamada y errores.

La primera tentativa falló en la validación del wheel por una ruta incorrecta
del runner, antes de medir. Se corrigió y su log permanece en `control/`; las
mediciones válidas están en `paired/`. La corrección final de la constante ABI
bool se hizo después de esta tanda exclusivamente flotante; el seguimiento y
la suite completa sí usan esa corrección. Los hashes originales se conservan.

## Resultados antes y después

Milisegundos, medianas **ronda 1 / ronda 2**. El control anterior ya incluye fusión.
No deben encadenarse porcentajes con otras rondas tomadas en momentos distintos.

| Caso | Residente anterior | Residente actual | Completo anterior | Completo actual |
| --- | ---: | ---: | ---: | ---: |
| cuBLAS, cadena grande f32 | 0,086 / 0,087 | 0,073 / 0,073 | 2,441 / 2,531 | 2,422 / 2,498 |
| cuBLAS, energía f32 | 0,162 / 0,179 | 0,140 / 0,151 | 1,653 / 1,645 | 1,636 / 1,625 |
| cuBLAS, centrado por filas f32 | 0,225 / 0,229 | 0,187 / 0,189 | 0,723 / 0,568 | 0,683 / 0,536 |
| Custom, ridge f32 | 7,679 / 6,300 | 6,125 / 5,183 | 6,991 / 6,900 | 5,673 / 5,743 |
| Custom, ridge f64 | 6,946 / 8,097 | 5,149 / 6,896 | 7,594 / 9,369 | 6,365 / 8,034 |
| cuBLAS, ridge f32 | 6,961 / 5,891 | 6,105 / 5,410 | 7,111 / 6,675 | 8,455 / 6,128 |
| cuBLAS, ridge f64 | 5,981 / 6,170 | 5,371 / 5,504 | 6,772 / 6,982 | 6,269 / 6,453 |

Las 72 mejoras residentes están entre **7,7 % y 31,3 %**. Las mediciones completas
mejoran en 63 de 72 comparaciones, con resultados mixtos en cadenas grandes,
pérdida cuadrática, covarianza f64 y energía por filas f64. No se atribuyen
automáticamente esas diferencias a PCIe ni a un kernel.

### Seguimiento del resultado adverso cuBLAS ridge f32

La pérdida completa del 18,9 % de la primera ronda motivó tres bloques adicionales,
con 51 pares alternados por alcance, reposo observado y el mismo algoritmo:

| Bloque | Residente anterior → actual | Completo anterior → actual |
| --- | ---: | ---: |
| 1 | 6,124 → 5,660 ms | 6,757 → 6,260 ms |
| 2 | 6,311 → 5,861 ms | 6,759 → 6,181 ms |
| 3 | 6,123 → 5,599 ms | 6,893 → 6,347 ms |

La pérdida no se reprodujo en esos bloques; tampoco se descarta el dato inicial.
Las mediciones completas muestran variación incluso en CuPy: ridge f32 dio
4,806 / 7,320 ms. No hay evidencia suficiente para una excepción por workload.

## Qué queda por optimizar

Un control **solo de CPU** para cuatro argumentos, con 31 tandas alternadas de
2000 llamadas, baja de **6,0 a 0,70 μs** por empaquetado. No incluye driver,
binding del compilador, asignación, kernel ni sincronización. El coste de este
subpaso ya es pequeño; seguir afinándolo ofrece menos margen que otros frentes.

La cadena grande f32 tarda 73 μs residente con CUDA/cuBLAS y **30 / 33 μs** con
CuPy fuse. Sus tiempos completos quedan próximos por las copias: CUDA 2,422 /
2,498 ms frente a CuPy fuse 2,440 / 2,434 ms. En energía, CuPy eager sigue siendo
mejor que su fuse: 0,137 / 0,099 ms frente a 0,401 / 0,406 ms residentes.
No se escoge únicamente el modo de CuPy que favorece al comparador.

NumPy sigue ganando ridge en este tamaño: **1,111 / 1,093 ms completos**. El
objetivo de esta ronda es quitar coste de abstracción, sin presentar una mejora
relativa propia como superioridad sobre CPU u otras bibliotecas.

El diagnóstico conserva el número de lanzamientos: ridge custom usa 80 kernels
propios; cuBLAS usa 40 kernels propios y los cuarenta productos del algoritmo.
En ridge cuBLAS aparecen **243 pares push/pop de contexto** en la llamada
instrumentada, incluidas sincronizaciones y reciclaje. Reducir esa repetición
merece un control separado, con restauración del contexto y concurrencia probadas.

Los eventos de la cadena grande CUDA/cuBLAS delimitan unos 28 μs en el primer
diagnóstico, frente a 73 μs de pared sin instrumentar. **No se resta una cifra
de la otra para afirmar un coste Python exacto**: los intervalos de eventos pueden
incluir huecos de envío/cola y la instrumentación cambia la ejecución. Son
indicadores de que aún hay margen, no una cota física de NVIDIA.
[Semántica de eventos CUDA](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html).

La prioridad siguiente rota fuera del empaquetado de argumentos: controlar
activaciones de contexto/envío y binding del plan; comparar su ahorro posible
con GEMV/DOT, reducciones por eje y layouts. Vulkan/OpenCL mantienen esos frentes
abiertos. No se cambia una ruta por inercia ni se debilita el contrato para ganar.

## Precisión y verificación

Los resultados de ambas variantes son exactamente iguales en todos los casos
controlados. Frente a NumPy calculado en float64, los máximos relativos L2 son:

| Ruta actual | float32 | float64 |
| --- | ---: | ---: |
| CUDA custom | 4,36×10⁻⁷ | 8,20×10⁻¹⁶ |
| CUDA cuBLAS | 8,37×10⁻⁸ | 3,19×10⁻¹⁶ |
| CuPy, ambos modos | 1,80×10⁻⁷ | 3,06×10⁻¹⁶ |

Float64 se compara con una referencia de igual precisión, no con una solución
exacta. El máximo absoluto CUDA float32 es 0,00489; los errores se conservan por
caso junto a los tiempos. No son garantías para entradas diferentes.

La suite completa con OpenCL/Vulkan/CUDA obligatorios y CLBlast nativo da
**1200 pruebas pasando y una omitida** por necesitar dos dispositivos OpenCL.
Las regresiones nuevas cubren bytes/tamaños ABI, ceros con signo, reenlace sin
retención de inputs, tipos/cantidad incorrectos, lanzamiento fallido y reintento,
lock, liberación con expulsión del módulo y resultados booleanos en la GPU.
La validación existente cubre concurrencia, precisión adversa, colas/rectangulares,
RNG, vida de buffers y restauración de contexto. Ver [empaquetado y controles](verification.md).

## Artefactos y reproducción

`.artifacts/cuda-dispatch/summary.csv` reúne las 236 configuraciones principales;
`paired-gains.csv` registra las 72 comparaciones y diferencias de pares.
`paired/` y `references/` conservan JSON, muestras, comandos y telemetría. El
seguimiento está en `ridge-followup.json` (seis configuraciones adicionales),
su script y su gate; el diagnóstico CPU en `packing_profile.py/json`.
Los perfiles cProfile son diagnósticos instrumentados, no benchmarks GPU.

El wheel local anterior tiene SHA256
`013d623c9d2a0504d1a4e6177fa83e5e0953519dc18d418788cc9c95eeead97b`.
Con extras instalados, usar carpetas de salida nuevas:

```powershell
python -m benchmarks.isolated --worker cuda_dispatch --backends cuda-custom,cuda-cublas --rounds 2 --repeats 31 --max-temperature 52 --control-wheel .artifacts/cuda-dispatch/before.whl --output .artifacts/cuda-dispatch/paired
python -m benchmarks.isolated --worker cuda_dispatch --backends numpy-8t,cupy --rounds 2 --repeats 31 --max-temperature 52 --output .artifacts/cuda-dispatch/references
```

La asociación automática backend/UUID de telemetría sigue pendiente. El hardware
observado es el mismo de la ronda anterior; Linux y otros fabricantes no fueron
verificados físicamente. La versión sigue siendo 0.1.0a1, sin publicación.
