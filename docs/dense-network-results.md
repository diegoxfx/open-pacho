# Un cliente de red densa con gradientes explícitos — 2026-10-22

`examples/dense_network.py` implementa inferencia y un paso SGD de una red de
regresión de dos capas afines con activación tanh intermedia. Es un cliente de las
operaciones existentes: no amplía el runtime, añade autodiff ni instala un
framework de redes. Su finalidad es comprobar y medir conjuntamente GEMM,
broadcasting, activaciones, reducciones, transpuestas y temporales.

La familia procede del [plan de aplicaciones](application-measurement-plan.md).
El código de dominio usa operaciones públicas y funciona directamente con NumPy,
CuPy y OpenPacho. `compile` captura las funciones exteriores; no contiene llamadas
compiladas anidadas, descargas ni cambios de backend. El wheel sigue excluyendo
los ejemplos y NumPy continúa siendo la única dependencia obligatoria.

## Contrato matemático

Para X de forma (n,d), W1 (d,h), b1 (h,), W2 (h,o), b2 (o,) y Y (n,o):

```text
H  = tanh(X @ W1 + b1)
P  = H @ W2 + b2
L  = sum((P - Y)^2) / (2*n*o)
D2 = (P - Y) / (n*o)
dW2 = H.T @ D2             db2 = sum(D2, axis=0)
D1  = (D2 @ W2.T) * (1-H*H)
dW1 = X.T @ D1             db1 = sum(D1, axis=0)
```

`predict` devuelve P. `loss_and_gradients` devuelve L y los cuatro gradientes.
`sgd_step` devuelve **la pérdida anterior al paso** y cuatro parámetros nuevos:
W1−rate·dW1, b1−rate·db1, W2−rate·dW2, b2−rate·db2. Todos los gradientes utilizan
los parámetros originales; ni los inputs ni sus arrays de parámetros se modifican.
La pérdida promedia también la dimensión de salida, no solo las filas del batch.

Las dimensiones deben ser positivas y los arrays compartir dtype float32/64 y
backend. Se comprueban formas y dtypes mediante metadatos. Las entradas finitas y
una tasa positiva finita son responsabilidad del caller; no se observan datos GPU
para validarlas. El ejemplo no demuestra que una tasa arbitraria haga converger
el entrenamiento. Vulkan conserva su error explícito para tanh float64.

## Protocolo de medición

`benchmarks/dense_network.py` usa las mismas funciones de dominio en todos los
comparadores. Cada proceso selecciona un backend: NumPy con 1/8 hilos, OpenCL
custom/CLBlast, Vulkan, CUDA custom/cuBLAS o CuPy. OpenPacho tiene filas eager y
compiladas; las compiladas preparan el plan una vez y lo reutilizan.

El perfil completo contiene tres formas (n,d,h,o): (31,7,11,3),
(4096,128,256,32) y (4095,127,255,31), en float32/64, para inferencia y un paso SGD.
Son datos sintéticos de semilla 47, entradas y pesos escalados por dimensión,
bias no nulo y targets fijos. La tasa es 0,1. Cada llamada SGD parte de los mismos
parámetros originales: no mide una trayectoria cuyo estado cambie entre variantes.

La inferencia solo sube X y los cuatro parámetros; no carga targets que no utiliza.
El paso SGD sube los seis arrays y descarga pérdida y cuatro parámetros nuevos.
Las transpuestas de activaciones, pesos y X forman parte del cálculo; no se
ocultan en una preparación externa. Se registra:

- Subida inicial, preparación de plan y primera ejecución sincronizada, que puede
  compilar kernels del dispositivo.
- Calentamiento de al menos tres llamadas y 0,1 s.
- Doce muestras residentes por implementación: asignaciones, cálculo y sincronización.
- Doce muestras completas: subida, cálculo con plan reutilizable/eager y descarga
  de todas las salidas declaradas. La preparación se informa por separado.
- Muestras de subida y primera descarga fuera de los relojes anteriores, bytes
  de entradas/salidas, versiones, fuentes, hilos y dispositivo. El **pico de memoria
  no está medido**; los bytes no se presentan como un sustituto.

Las dos implementaciones de cada proceso alternan en órdenes balanceados. Los
diagnósticos quedan fuera del reloj e incluyen cada salida antes y durante las
repeticiones. Para SGD también se evalúan los gradientes y la predicción; el control
de gradientes usa ejecución compilada cuando la fila medida es compilada.

La referencia float64 convierte los mismos datos ya redondeados, sin generar una
entrada distinta. Para filas float64 se identifica como `sameprecision`: no es
un oráculo de precisión superior. Los umbrales fijados antes de medir son
rtol=2e-4, atol=2e-6 para float32 y rtol=2e-10, atol=2e-12 para float64. Se registran
errores absolutos y relativos, dtypes, formas y valores no finitos. Un rechazo
numérico conserva los tiempos; un fallo de ejecución queda en su fila y permite
intentar las restantes. `attempts_complete`, `execution_complete` y
`all_numerically_accepted` expresan condiciones distintas.

```powershell
.venv/Scripts/python.exe -m benchmarks.dense_network --backend numpy --profile quick --repeats 12 --output .artifacts/benchmarks/dense-numpy.json
.venv/Scripts/python.exe -m benchmarks.dense_network --backend opencl --profile all --repeats 12 --output .artifacts/benchmarks/dense-opencl.json
```

Los comandos individuales requieren archivos nuevos y no aplican por sí mismos
la puerta de reposo GPU del supervisor. El retorno no cero preserva el JSON de
errores o rechazos numéricos; no significa que se hayan perdido las filas restantes.

## Verificación funcional

Las diferencias finitas centrales de todos los parámetros de una red pequeña
float64 verifican de forma independiente los gradientes, el factor n·o y la
actualización. También se comprueban activaciones saturadas, preservación de
inputs, float32/64, dimensiones distintas, una sola fila y rebinding de arrays/tasa
en el plan compilado. La pérdida disminuye en un caso con tasa pequeña; no se
extrapola esa observación a cualquier problema.

Pasan 24 pruebas CPU del ejemplo y ocho pruebas nativas: OpenCL, Vulkan,
CUDA custom y CUDA cuBLAS, incluyendo el rechazo esperado de tanh float64 Vulkan.
Las 13 pruebas CPU del runner verifican referencias sobre datos redondeados,
descarga de todas las salidas, parámetros originales entre llamadas, modos
balanceados, preparación real en NumPy y conservación de fallos. No se ejecutó
la suite completa ni se cambió el runtime.
## Resultados locales

Se completaron dos rondas con orden de backends invertido, más dos procesos
suplementarios CuPy. Los primeros dos intentos CuPy fallaron al crear la caché
fuera del directorio permitido; se conservan sus 24 filas de error. Los reintentos
usan `CUPY_CACHE_DIR` dentro del workspace, sin cambiar inputs, código ni tolerancias.
Al ejecutarse después de la cola original, sus posiciones no están intercaladas
con los otros backends. No se sustituyen los informes originales.

Hay **288 filas ejecutadas y aceptadas**, con **6.912 muestras residentes/completas**.
Se conservan también 24 filas float64 Vulkan rechazadas por tanh no soportado y
las 24 filas de fallo inicial de caché CuPy. Todos los informes terminaron sus
intentos; los que incluyen fallos no anuncian ejecución completa ni aceptación global.
Los hashes de inputs por caso coinciden entre comparadores y rondas, y las fuentes
cargadas de runtime, ejemplo y runner se mantuvieron constantes.

Las 14 puertas GPU se cumplieron sin relajar la política: mínimo diez segundos,
tres lecturas consecutivas de utilización GPU/memoria ≤5%, máximo 52 °C y dos
grados sobre la temperatura base. La espera observada fue 10.48–73.08 s
y el inicio 40–42 °C. Esto describe el reposo antes de cada proceso,
no una medición de temperaturas de cada kernel. Entorno: Windows, RTX 5060 Ti,
driver 617.14, Python 3.12.14 y NumPy 2.5.3; las demás versiones e información
de bibliotecas/hilos están en los JSON.

Los máximos errores absolutos observados, incluyendo replay y todos los
comparadores ejecutados, fueron:

| Dtype | Predicción | Pérdida | Gradientes | Parámetros nuevos |
| --- | ---: | ---: | ---: | ---: |
| float32 | 3,66e-7 | 4,69e-9 | 1,19e-8 | 2,92e-8 |
| float64 | 6,67e-16 | 1,39e-17 | 2,61e-17 | 1,39e-17 |

Se redondean hacia arriba para resumirlos; no son cotas para otras entradas.
Los 23.328 diagnósticos de las filas aceptadas cumplieron los umbrales publicados.

Las tablas muestran **milisegundos**, intervalo de las dos medianas, para la forma
regular (4096,128,256,32). R = residente; C = completo. Preparación y primera
llamada están separadas. No son intervalos de confianza ni un ranking universal.

### float32

| Implementación | Inferencia R | Inferencia C | SGD R | SGD C |
| --- | ---: | ---: | ---: | ---: |
| numpy-1t | 7.378–7.418 | 7.526–7.593 | 15.169–15.325 | 15.111–15.302 |
| numpy-8t | 4.709–4.772 | 4.920–4.985 | 8.904–9.125 | 9.009–9.211 |
| opencl-eager | 0.364–0.380 | 1.067–1.112 | 2.194–2.359 | 3.149–3.597 |
| opencl-compiled | 0.266–0.267 | 0.884–0.893 | 1.237–1.447 | 2.153–2.659 |
| opencl-clblast-eager | 0.517–0.545 | 1.393–1.596 | 2.550–2.551 | 3.435–3.497 |
| opencl-clblast-compiled | 0.308–0.317 | 1.007–1.274 | 1.214–1.228 | 2.165–2.189 |
| vulkan-eager | 0.727–0.932 | 1.360–1.447 | 3.901–4.292 | 5.170–5.491 |
| vulkan-compiled | 0.769–0.788 | 1.302–1.414 | 1.674–1.802 | 2.694–2.834 |
| cuda-custom-eager | 0.621–0.621 | 1.436–1.455 | 3.323–3.353 | 4.422–4.487 |
| cuda-custom-compiled | 0.298–0.300 | 1.082–1.091 | 1.486–1.501 | 2.554–2.589 |
| cuda-cublas-eager | 0.692–0.795 | 2.009–2.247 | 3.463–3.557 | 4.553–4.613 |
| cuda-cublas-compiled | 0.500–0.509 | 1.600–1.717 | 1.290–1.300 | 2.342–2.368 |
| cupy-eager | 0.308–0.423 | 1.009–1.340 | 1.103–1.127 | 1.767–1.783 |

### float64

| Implementación | Inferencia R | Inferencia C | SGD R | SGD C |
| --- | ---: | ---: | ---: | ---: |
| numpy-1t | 22.302–22.811 | 22.618–22.952 | 36.916–37.801 | 36.868–37.602 |
| numpy-8t | 16.474–16.983 | 17.319–17.640 | 24.931–26.244 | 24.482–25.740 |
| opencl-eager | 1.098–1.101 | 2.487–2.598 | 2.771–2.867 | 4.212–4.671 |
| opencl-compiled | 1.070–1.077 | 2.422–2.445 | 2.679–2.684 | 3.995–4.227 |
| opencl-clblast-eager | 1.145–1.156 | 2.500–2.503 | 2.745–2.750 | 4.040–4.169 |
| opencl-clblast-compiled | 1.116–1.117 | 2.492–2.594 | 2.618–2.634 | 4.029–4.060 |
| vulkan-eager | sin soporte | sin soporte | sin soporte | sin soporte |
| vulkan-compiled | sin soporte | sin soporte | sin soporte | sin soporte |
| cuda-custom-eager | 2.226–2.230 | 3.767–3.884 | 5.066–5.093 | 6.714–6.729 |
| cuda-custom-compiled | 2.178–2.181 | 3.686–3.705 | 4.917–4.923 | 6.441–6.525 |
| cuda-cublas-eager | 1.374–1.394 | 2.873–3.047 | 3.349–3.487 | 4.867–4.944 |
| cuda-cublas-compiled | 1.322–1.351 | 3.011–3.093 | 2.830–2.835 | 4.345–4.360 |
| cupy-eager | 1.296–1.350 | 2.639–2.714 | 2.680–2.727 | 3.938–4.045 |

### Formas impares y caso pequeño

Para (4095,127,255,31), estas razones comparan la mediana compilada con la eager
del mismo proceso. Se muestran ambas rondas, incluidos resultados mixtos:

| Backend/dtype | Inferencia R | Inferencia C | SGD R | SGD C |
| --- | ---: | ---: | ---: | ---: |
| opencl/float32 | 0.676–0.690 | 0.824–0.836 | 0.626–0.673 | 0.717–0.794 |
| opencl/float64 | 0.981–0.984 | 0.986–0.991 | 0.964–0.970 | 0.945–0.990 |
| opencl-clblast/float32 | 0.610–0.652 | 0.846–0.873 | 0.467–0.495 | 0.635–0.667 |
| opencl-clblast/float64 | 0.977–0.980 | 0.979–1.008 | 0.925–0.938 | 0.938–0.983 |
| vulkan/float32 | 0.748–0.816 | 0.831–0.831 | 0.387–0.460 | 0.492–0.575 |
| cuda-custom/float32 | 0.484–0.509 | 0.757–0.757 | 0.349–0.361 | 0.432–0.498 |
| cuda-custom/float64 | 0.977–0.982 | 0.986–0.998 | 0.968–0.983 | 0.959–0.991 |
| cuda-cublas/float32 | 0.467–0.512 | 0.767–0.777 | 0.303–0.380 | 0.448–0.528 |
| cuda-cublas/float64 | 0.962–0.974 | 0.968–0.982 | 0.781–0.860 | 0.886–0.904 |

En la inferencia pequeña float32, NumPy tarda 0.026–0.028 ms completos;
las GPU 0.416–1.042 ms. El coste de despacho/transferencias importa:
los resultados grandes no justifican mover automáticamente cargas pequeñas a GPU.
El backend sigue siendo una elección explícita del usuario.

## Qué cambia la prioridad

La fusión y preparación existentes reducen claramente el coste del paso SGD
float32 en estos casos. Las ganancias float64 suelen ser menores; hay rutas de
inferencia con resultado completo mixto, incluido cuBLAS. Los tiempos de CuPy y
las dos rondas también muestran variación: no se atribuyen diferencias pequeñas
a una mejora demostrada ni se declara paridad universal entre bibliotecas.

El cliente revela un frente concreto: CUDA custom float64 queda por detrás de
cuBLAS/OpenCL. La inspección confirma que sus GEMM float64 usan el kernel directo,
mientras OpenCL usa un tile con memoria compartida. Esto justifica descomponer
el flujo y probar una variante acotada; **la comparación de aplicaciones aún no
cuantifica el tiempo exclusivo del kernel ni demuestra una aceleración posible**.
No se modifican kernels ni selección de proveedor en este lote.

Los informes, muestras, diagnósticos y supervisores se conservan en
`.artifacts/iterations-2026-10-22/dense-network/` y su directorio padre. Las tablas
provienen de esos informes; se conservan el caso pequeño, las formas impares,
los fallos explícitos y las distribuciones de todas las muestras.

QA independiente: **APPROVE**. Reejecutó las 37 pruebas CPU, inspeccionó las ocho
pruebas nativas y reconcilió las 140 celdas de las tablas, muestras, diagnósticos,
puertas de reposo y hashes. Confirmó que el runtime permanece idéntico al lote
anterior. No ejecutó GPU durante la revisión.
