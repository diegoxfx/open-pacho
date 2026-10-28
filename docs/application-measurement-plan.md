# Medir aplicaciones además de primitivas

Plan y primera suite del 2026-09-08. Los solvers anteriores usan almacenamiento
denso de una matriz tridiagonal SPD con diagonal 2.5 y bandas -1, condición ≤9.
Ese control sirve para repetir resultados e iteraciones, pero no representa la
diversidad de problemas densos, condicionamientos ni matrices de aplicaciones.
Tampoco es evidencia para priorizar soporte disperso, que sigue fuera de alcance.

## Primera ampliación implementada

`benchmarks/applications.py` contiene dos patrones aplicados con datos sintéticos
deterministas. No se presentan como resultados sobre datasets de producción.

| Carga | Cálculo | Qué revela |
| --- | --- | --- |
| Covarianza muestral | Centrado por media de columnas; `center.T @ center / (n-1)` | Reducción por eje, broadcasting, materialización de transpuesta y GEMM rectangular. |
| Regresión ridge | 20 pasos de `w -= rate * (Xt @ (X @ w-y) + lambda*w)` | Reutilización de matriz y transpuesta preparada, dos GEMV rectangulares por paso y actualización fusionada. |

Formas completas: 4096×128 y 16384×256, float32/64. Semilla 47. Ridge usa
`rate=0.25/n`, `lambda=0.1`, datos normales escalados por `1/sqrt(d)` y RHS con
ruido. Los veinte pasos son un **control de trabajo fijo**, no una afirmación de
convergencia ni el mejor optimizador para ese problema. Se publica objetivo final,
norma del gradiente y error respecto a la misma trayectoria float64.

Cada implementación recibe los mismos inputs. Se conserva NumPy directo con
1/8 hilos, OpenCL custom/CLBlast, Vulkan, CUDA custom/cuBLAS opcional y CuPy;
OpenPacho tiene controles eager
y compiled. La transpuesta de ridge se materializa una vez y se cronometra como
preparación. El modo residente reutiliza esa preparación; el modo completo incluye
subida, preparación, cálculo y descarga. En covarianza, el centrado y la transpuesta
del array centrado forman parte de cada cálculo. No se esconde ese coste.

```powershell
.venv/Scripts/python.exe -m benchmarks.applications --quick --backends "" --output .artifacts/benchmarks/applications-cpu.json
.venv/Scripts/python.exe -m benchmarks.applications --repeats 7 --output .artifacts/benchmarks/applications.json
.venv/Scripts/python.exe -m benchmarks.applications --irregular --repeats 7 --output .artifacts/benchmarks/applications-irregular.json
```

El informe registra primera llamada/compilación, preparación, residencia,
transferencias y diagnósticos fuera del cronómetro. La referencia float64 de los
casos float64 es de igual precisión, no un oráculo. Las métricas de precisión
adversa con Decimal siguen complementando estas mediciones a escala.

El perfil `--irregular` usa 4095×127 y 4097×129 en ambas precisiones y cargas.
Permite comparar dimensiones impares a ambos lados de las potencias de dos del
caso 4096×128. `tests/test_irregular_workloads.py` añade 144 combinaciones:
GEMM rectangular alrededor de tiles 16/32/64/128, tamaños pares no potencia de dos,
GEMV, reducciones por ambos ejes y broadcasting por lotes impares. Los solvers
cuadrados de tamaños 31/65/129 usan matrices densas SPD diferentes de la familia
tridiagonal, comparan sistema ordinario/preparado y verifican residuo independiente.
No se pasa una matriz rectangular a un solver cuya API exige un sistema cuadrado.

Desde 2026-09-12, `--backends` también admite `cuda-cublas,cuda-custom`.
Son rutas del backend CUDA inicial; `compiled` captura el plan y fusiona los grupos
elementwise/map-reduce admitidos mediante `CUDALowering`. Matmul y BLAS permanecen
separados. Actualmente CUDA también admite normas estables y los ejemplos de
solvers; las entradas bool pueden reducirse con acumulador flotante explícito,
pero la acumulación bool y los graphs siguen fuera de alcance. Los JSON identifican
los planes como `compiler_lowering: backend-plan` y registran
las versiones de bibliotecas nativas. La [ronda de fusión CUDA](cuda-fusion-results.md)
documenta ese alcance; la [ronda CUDA inicial](cuda-initial-results.md) conserva las
mediciones base al comparar tiempos y errores.

## Nuevas familias acordadas

Los solvers actuales viven en `examples/solvers.py`, fuera del paquete. Nuevos
algoritmos de dominio seguirán esa separación, con un ejemplo legible y un runner
independiente para cronometrar; no se añade un framework neuronal o físico al runtime.

| Familia propuesta | Primer caso | Qué revela | Validación |
| --- | --- | --- | --- |
| Red densa, implementada | Inferencia de dos capas con tanh y un paso SGD con gradientes explícitos, en `examples/dense_network.py` | GEMM rectangular, bias/broadcasting, activaciones, transpuestas y temporales | Todos los gradientes frente a diferencias finitas en un caso pequeño; errores de predicción, loss, gradientes y parámetros; misma actualización en todas las referencias |
| Osciladores acoplados, implementados | Pasos de velocity-Verlet en sistema armónico denso SPD, con matriz de rigidez reutilizada | GEMV repetido, actualizaciones fusionables, sincronización y acumulación de error | Trayectoria analítica/modal en casos pequeños, error temporal y deriva de energía; mismo dt, pasos y condiciones iniciales |

Ambos admiten tamaños impares/rectangulares en sus operandos apropiados y usan
principalmente operaciones existentes. La simulación densa es un control de
operaciones densas, no una forma eficiente de representar cualquier física local;
no densificar un problema disperso para obtener un benchmark favorable.

Entrenamiento no implica añadir autodiff. Empezar con NumPy directo, OpenCL
custom/CLBlast, Vulkan y CuPy/CUDA donde sus primitivas lo permitan. PyTorch/JAX
son controles adicionales si implementan el mismo trabajo y precisión; separar
compilación/captura y distinguir autodiff de gradientes explícitos. Conservar
rutas eager/compiladas, errores, fallos, transferencias y memoria máxima.
La [primera red densa](dense-network-results.md) dispone de ejemplo, pruebas y runner
independiente; registra bytes pero todavía no instrumenta el pico de memoria.
La [simulación de osciladores](oscillators-results.md) añade un recorrido de pasos
fijos, con un único paso compilado reutilizado. Separa diferencias de backend,
error temporal modal y energía; tampoco mide todavía el pico de memoria.

## Próximas cargas y decisiones que deben justificar

1. **Distancias por bloques para búsqueda de vecinos/clustering.** Comparar
   diferencias fusionadas contra identidad de Gram, incluyendo cancelación al
   restar términos grandes. Medir pico de memoria y coste de bloques; un método
   más rápido que pierde distancias pequeñas no cumple el mismo objetivo.
2. **Transformaciones por lotes.** Muchas matrices pequeñas con broadcasting,
   tamaños irregulares y diferentes cantidades de lotes. Separar despacho por
   matriz de batched real; no descartar el workload porque cada matriz sea pequeña.
3. **Solvers en matrices densas variadas.** Familias SPD generadas con espectro
   controlado, matrices de Gram regularizadas y varios RHS. Medir hasta tolerancia,
   iteraciones, error de solución, residuo y memoria; no inferirlo de veinte pasos.
4. **Pipelines reutilizados.** Centrado/normalización, covarianza y proyecciones con
   factores ya disponibles. Evaluar `mean`, transpose, GEMM y temporales antes de
   añadir una factorización nueva. No anunciar PCA completo sin eigensolver.

La elección de nuevas funciones debe seguir a estos casos. `maximum`/clipping,
reducciones eficientes por eje y matmul con operandos transpuestos pueden aportar
más valor inmediato que ampliar la superficie de NumPy indiscriminadamente.

## Protocolo para decidir prioridades

- Separar problema pequeño, datos que caben en caché y matrices mayores que la
  caché; registrar bytes y memoria máxima junto a tiempo y error.
- Comparar mismo dtype, inputs, algoritmo, número de pasos o criterio de parada.
  Tratar una mejora algorítmica como una comparación adicional identificada.
- Publicar mediana y dispersión, primera ejecución y coste de preparación;
  intercalar implementaciones y conservar casos mixtos o fallidos.
- Medir timestamps GPU y tasas de copia/cálculo sostenidas para construir un
  modelo de límites por carga. No llamar límite físico a la velocidad de CuPy.
- Cuando un caso aplicado revele una regresión, aislar su operación dominante
  antes de cambiar kernels o selección de proveedor.

Los resultados de esta etapa se recogen en
[solvers, aplicaciones y preparación CUDA](solver-applications-results.md).
