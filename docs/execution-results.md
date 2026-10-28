# Compilador, Vulkan y referencias CUDA: resultados locales

**Snapshot histórico anterior a las optimizaciones de GEMV, map-reduce y Vulkan.**
La continuación de las prioridades al final de este documento está en
[gaps y optimización](gaps-and-optimization.md); los tiempos de esta página se
conservan como referencia del estado anterior.
La continuación más reciente está en [map-reduce, preparación y BLAS batched](remaining-suite-results.md).

Plataforma: Windows-11-10.0.26200-SP0. Dispositivos: {'opencl': 'NVIDIA GeForce RTX 5060 Ti', 'opencl-clblast': 'NVIDIA GeForce RTX 5060 Ti', 'vulkan': 'NVIDIA GeForce RTX 5060 Ti'}. La sesión original del 2026-09-06 corresponde a Windows 11, RTX 5060 Ti 16 GB (driver 616.64), Ryzen 7 5700X3D y 32 GB RAM. No se ha validado AMD/Intel.

**235 configuraciones, siete rondas por modo, cero errores de validación.** 172 de operaciones/pasos y 63 de solvers completos. Los JSON conservan todas las muestras; el CSV añade cuartiles.

## Qué cambia y qué sigue costando

100 actualizaciones de 1 048 576 valores float32: 203.974 ms en el mejor NumPy, 33.846 ms en OpenCL directo, 11.280 ms con compile y 6.187 ms en CuPy. Incluyendo transferencias: 209.622 ms NumPy y 13.418 ms OpenCL compilado.

Matmul 2048×2048 float32: 31.670 ms en NumPy, 3.771 ms OpenCL+CLBlast, 1.458 ms cuBLAS y 1.589 ms nvmath. La ventaja principal aquí viene del proveedor matricial: compilar un matmul aislado no lo fusiona ni mejora necesariamente su kernel.

Jacobi completo a 2048 float32: 10.402 ms NumPy, 25.121 ms OpenCL custom y 14.658 ms CuPy. Descenso más pronunciado: 13.354 ms NumPy, 34.135 ms OpenCL y 23.711 ms CuPy. Son tiempos residentes: la transferencia no es la única causa de las diferencias. Hay GEMV/DOT, despacho, asignaciones y sincronización para convergencia. CLBlast por sí solo no resuelve el algoritmo.

Vulkan: matmul 2048 float32 residente 20.447 ms, con transferencias 108.242 ms; 100 actualizaciones grandes, 303.090 ms. Es funcional; faltan pool, fusión y menor coste de envío. Estos números no representan un límite de Vulkan como tecnología.

100 actualizaciones de solo 256 valores: 0.301 ms NumPy y 11.023 ms OpenCL compilado. La corrección se conserva; no se prioriza hacer que GPU gane este caso.

## Método y límites de comparación

- Tiempo de pared sincronizado, con despacho Python y asignación de resultados. Rondas con orden aleatorio de implementaciones. No se usan tiempos de lanzamiento asincrónico como cómputo completo.
- Residente: inputs ya cargados, planes preparados y kernels calentados. Cada repetición reinicia el estado lógico de las iteraciones. La validación y descarga del resultado quedan fuera de este tiempo.
- Con transferencias: nuevos buffers de entrada, cómputo y descarga. Se reutilizan planes/programas; nvmath reenlaza operandos sin planificar otra vez. NumPy usa los arrays host existentes: no se le añade una copia artificial.
- Los JSON separan subida, planificación, primera ejecución y descarga. La primera ejecución es de esa configuración dentro del proceso; puede reutilizar cachés de configuraciones anteriores. No mide un proceso frío independiente ni incluye inicialización del dispositivo. La captura de OpenPacho está incluida en su primera ejecución; la planificación explícita de nvmath figura separada.
- NumPy se mide con BLAS limitado a 1, 8 y 16 hilos. Las tablas muestran la mejor mediana por caso y modo. Las operaciones sin BLAS no se paralelizan por cambiar ese límite; diferencias pequeñas ahí son ruido.
- float32/float64 originales, TF32 deshabilitado; nvmath usa COMPUTE_32F/64F. Se comprueban forma, dtype y valores en cada ronda. No se exige identidad bit a bit.
- `updates100`: x=(1-tasa)*x+tasa*b, cien pasos. `energy`: sum((2*(a*b+c))²,axis=-1), arrays tamaño×32. Jacobi20 y gradient20 usan 20 pasos de una matriz SPD densa dominante; gradient20 usa tasa fija.
- Los casos `_solve` ejecutan los solvers públicos hasta convergencia, con residuo real en cada iteración, rtol=1e-4/1e-7 para float32/64 y maxiter=400. El descenso más pronunciado usa búsqueda exacta de paso. NumPy/CuPy ejecutan la misma recurrencia mediante operaciones nativas de `benchmarks.suite.native_solver`. Los coeficientes escalares pueden tener rutas/precisión intermedia diferentes. Se valida además el residuo host contra la tolerancia (margen de 2 % por orden de reducción). Los inputs son sistemas tridiagonales almacenados densamente; no es una prueba sparse.
- CUDA es la plataforma. La referencia concreta cuBLAS usa bindings directos; CuPy matmul también puede usar cuBLAS y nvmath utiliza cuBLASLt. No son tres dispositivos independientes. cuBLAS/nvmath se comparan solo en matmul; no se inventa un solver de alto nivel para esas interfaces.
- No se midieron PyTorch/JAX, uso máximo de memoria, energía, todas las formas rectangulares ni portabilidad de rendimiento entre fabricantes. Esta batería cubre el alcance actual; no constituye todos los workloads posibles.

## Tiempos residentes

Milisegundos; menor es mejor. OpenPacho usa compile en operaciones/pasos; en `_solve`, el solver público ya compila sus actualizaciones. Los modos eager y compilado completos están en el JSON/CSV.

| Caso / tamaño / dtype | NumPy mejor | OpenCL custom | OpenCL CLBlast | Vulkan | CuPy | cuBLAS | nvmath |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul / 64 / float32 | 0.029 | 0.234 | 0.241 | 1.058 | 0.333 | 0.316 | 0.520 |
| matmul / 512 / float32 | 0.943 | 0.494 | 0.342 | 1.302 | 0.349 | 0.343 | 0.450 |
| matmul / 1024 / float32 | 5.172 | 2.481 | 0.761 | 3.460 | 0.469 | 0.489 | 0.582 |
| matmul / 2048 / float32 | 31.670 | 16.402 | 3.771 | 20.447 | 1.464 | 1.458 | 1.589 |
| matmul / 64 / float64 | 0.038 | 0.260 | 0.259 | 1.011 | 0.335 | 0.377 | 0.397 |
| matmul / 1024 / float64 | 8.530 | 8.620 | 5.883 | 9.357 | 6.886 | 6.893 | 7.047 |
| energy / 8 / float32 | 0.026 | 0.270 | 0.328 | 4.515 | 0.293 | — | — |
| energy / 2048 / float32 | 0.151 | 0.308 | 0.337 | 5.825 | 0.345 | — | — |
| energy / 32768 / float32 | 5.283 | 0.311 | 0.371 | 5.024 | 0.302 | — | — |
| updates100 / 256 / float32 | 0.301 | 11.023 | 10.924 | 270.404 | 7.564 | — | — |
| updates100 / 65536 / float32 | 2.565 | 11.067 | 11.234 | 264.744 | 6.872 | — | — |
| updates100 / 1048576 / float32 | 203.974 | 11.280 | 11.193 | 303.090 | 6.187 | — | — |
| jacobi20 / 512 / float32 | 0.370 | 2.217 | 2.518 | 70.555 | 2.623 | — | — |
| jacobi20 / 2048 / float32 | 3.738 | 6.730 | 2.441 | 73.488 | 3.738 | — | — |
| gradient20 / 512 / float32 | 0.414 | 2.850 | 3.422 | 70.073 | 2.844 | — | — |
| gradient20 / 2048 / float32 | 3.877 | 6.969 | 3.320 | 72.751 | 4.093 | — | — |
| jacobi_solve / 512 / float32 | 1.185 | 26.416 | 26.737 | 349.146 | 11.939 | — | — |
| jacobi_solve / 2048 / float32 | 10.402 | 25.121 | 27.532 | 350.699 | 14.658 | — | — |
| jacobi_solve / 1024 / float64 | 9.134 | 47.483 | 46.741 | 606.478 | 24.415 | — | — |
| sd_solve / 512 / float32 | 1.584 | 32.532 | 35.952 | 558.161 | 24.600 | — | — |
| sd_solve / 2048 / float32 | 13.354 | 34.135 | 34.451 | 538.382 | 23.711 | — | — |
| sd_solve / 1024 / float64 | 14.814 | 62.470 | 67.341 | 1033.909 | 47.033 | — | — |
| cg_solve / 512 / float32 | 0.916 | 19.386 | 19.520 | 323.175 | 13.047 | — | — |
| cg_solve / 2048 / float32 | 8.727 | 17.428 | 19.036 | 311.270 | 10.672 | — | — |
| cg_solve / 1024 / float64 | 6.583 | 29.073 | 31.723 | 530.994 | 20.406 | — | — |

## Carga + cómputo + descarga

Milisegundos; planes y programas calientes.

| Caso / tamaño / dtype | NumPy mejor | OpenCL custom | OpenCL CLBlast | Vulkan | CuPy | cuBLAS | nvmath |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul / 64 / float32 | 0.014 | 0.292 | 0.261 | 3.534 | 0.594 | 0.401 | 0.486 |
| matmul / 512 / float32 | 0.833 | 1.404 | 1.496 | 8.125 | 0.987 | 1.104 | 1.054 |
| matmul / 1024 / float32 | 4.943 | 4.685 | 3.330 | 27.957 | 2.966 | 2.980 | 2.859 |
| matmul / 2048 / float32 | 31.088 | 28.896 | 13.070 | 108.242 | 11.370 | 10.981 | 10.691 |
| matmul / 64 / float64 | 0.020 | 0.247 | 0.253 | 3.430 | 0.421 | 0.390 | 0.463 |
| matmul / 1024 / float64 | 10.346 | 12.464 | 10.471 | 52.349 | 12.765 | 11.779 | 11.831 |
| energy / 8 / float32 | 0.011 | 0.356 | 0.411 | 7.477 | 0.492 | — | — |
| energy / 2048 / float32 | 0.082 | 0.582 | 0.599 | 9.464 | 0.608 | — | — |
| energy / 32768 / float32 | 5.299 | 2.427 | 2.544 | 17.043 | 2.559 | — | — |
| updates100 / 256 / float32 | 0.295 | 10.835 | 11.152 | 267.471 | 7.336 | — | — |
| updates100 / 65536 / float32 | 2.515 | 11.527 | 10.988 | 296.495 | 7.290 | — | — |
| updates100 / 1048576 / float32 | 209.622 | 13.418 | 14.335 | 318.779 | 10.037 | — | — |
| jacobi20 / 512 / float32 | 0.346 | 2.407 | 2.697 | 73.664 | 2.811 | — | — |
| jacobi20 / 2048 / float32 | 2.853 | 9.660 | 5.131 | 90.634 | 4.477 | — | — |
| gradient20 / 512 / float32 | 0.372 | 3.110 | 3.766 | 73.844 | 3.056 | — | — |
| gradient20 / 2048 / float32 | 3.088 | 9.490 | 5.905 | 81.944 | 6.747 | — | — |
| jacobi_solve / 512 / float32 | 1.103 | 26.918 | 26.795 | 358.815 | 11.947 | — | — |
| jacobi_solve / 2048 / float32 | 9.954 | 27.671 | 28.881 | 360.976 | 15.301 | — | — |
| jacobi_solve / 1024 / float64 | 9.054 | 46.408 | 47.406 | 597.689 | 24.489 | — | — |
| sd_solve / 512 / float32 | 1.515 | 31.792 | 34.362 | 554.568 | 24.392 | — | — |
| sd_solve / 2048 / float32 | 12.775 | 48.296 | 41.136 | 539.987 | 27.616 | — | — |
| sd_solve / 1024 / float64 | 14.510 | 60.198 | 64.552 | 1068.319 | 46.568 | — | — |
| cg_solve / 512 / float32 | 0.807 | 18.753 | 19.533 | 330.216 | 12.697 | — | — |
| cg_solve / 2048 / float32 | 8.143 | 21.696 | 21.801 | 328.721 | 16.057 | — | — |
| cg_solve / 1024 / float64 | 6.428 | 30.813 | 32.579 | 544.778 | 19.665 | — | — |

## Prioridad siguiente

1. Reducir sincronizaciones y coste de ejecución en solvers sin cambiar silenciosamente su frecuencia de comprobación o tolerancias; optimizar GEMV/DOT.
2. Fusionar map-reduce y mejorar decisiones sobre temporales compartidos. Medir un ejecutor con menor overhead cuando domine Python.
3. Pool y fusión Vulkan, reutilización de descriptores/comandos y kernels matriciales optimizados. Verificar un segundo fabricante.
4. Ampliar operaciones frecuentes, como matmul por lotes, según casos reales. Mantener CLBlast y kernels custom intercambiables por contrato; CUDA backend y dispersas siguen como trabajo futuro.

## Reproducir

```powershell
$env:PYOPENCL_NO_CACHE='1'
$env:PACHO_VULKAN_COMPILER='<ruta a glslang o glslc>'
$env:OPENPACHO_CLBLAST_LIBRARY='<ruta a CLBlast>'
python -m benchmarks.execution --repeats 7 --output .artifacts/benchmarks/execution/raw.json
python -m benchmarks.execution --solvers-only --repeats 7 --output .artifacts/benchmarks/execution-solvers/raw.json
python -m benchmarks.execution_report --inputs .artifacts/benchmarks/execution/raw.json .artifacts/benchmarks/execution-solvers/raw.json --output docs/execution-results.md --artifacts .artifacts/benchmarks/execution-report
```

Dependencias adicionales: extra benchmark, CuPy y nvmath-python con componentes CUDA compatibles, CLBlast nativo, bindings Vulkan y compilador de shaders. No forman parte de las dependencias mínimas CPU de OpenPacho.

Versiones medidas: numpy 2.5.3, pyopencl 2026.1.4, vulkan 1.3.275.1, cupy-cuda13x 14.2.0, nvmath-python 1.0.0.
Los artefactos locales originales están en .artifacts/benchmarks/execution-2026-09-06, .artifacts/benchmarks/execution-solvers-2026-09-06.
