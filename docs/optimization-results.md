# Primera etapa de optimización — 2026-09-06

Se implementaron fusión elemento a elemento, pool de memoria OpenCL, escalares
por valor, actualizaciones fusionadas de solvers y un adaptador opcional CLBlast.
La elección de backend sigue siendo explícita. La API y sus límites están en
[performance.md](performance.md).

## Comparación final

26 configuraciones, 390 series y 7 rondas por serie, sin errores de ejecución o
validación numérica. Se midieron en la misma sesión NumPy con BLAS de 1/8/16 hilos,
el wheel anterior sin modificar y la versión actual. Se eligió el menor tiempo
mediano de NumPy por caso. Equipo: Ryzen 7 5700X3D y RTX 5060 Ti, Windows;
CLBlast 1.7.0, NumPy 2.5.3 y PyOpenCL 2026.1.4. Solo se validó esta GPU.

Medianas en **milisegundos con subida, cálculo y descarga**, runtime caliente,
entradas y validación fuera del reloj. El NumPy CPU recibe/devuelve datos de host
y no carga con una copia artificial. Cada llamada se sincroniza. No se afirma
significación estadística de diferencias pequeñas a partir de estas medianas.

| Caso | NumPy | OpenPacho anterior | Actual custom | Actual fusionado | Actual CLBlast |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 actualizaciones, 1.048.576 elementos, float32 | 197.50 | 231.58 | 34.08 | **26.18** | — |
| 100 actualizaciones, 1.048.576 elementos, float64 | 378.19 | 258.35 | 35.43 | **28.20** | — |
| sin/exp/cos, 1.048.576 elementos, float32 | 7.66 | 9.16 | 2.95 | **2.68** | — |
| Matmul 2048², float32 | 30.45 | 36.64 | 24.38 | — | **11.17** |
| Matmul 2048², float64 | 63.15 | 86.27 | 61.60 | — | 60.78 |
| Jacobi 4096 variables, float64 | 138.71 | 166.72 | **95.10** | — | 97.16 |
| Jacobi 1024 variables, float32 | **3.12** | 32.27 | 27.01 | — | 29.01 |
| CG 1024 variables, float32 | **2.47** | 22.08 | 24.22 | — | 24.80 |

En las expresiones genéricas, `custom` ya incluye memoria reutilizable y escalares
por valor. `fusionado` agrega `@fuse`. En los solvers, tanto `custom` como `CLBlast`
ya usan las actualizaciones fusionadas. CLBlast cambia los productos flotantes.

## Lo que aporta y lo que queda pendiente

- La secuencia float32 fusionada es aproximadamente **7,54× más rápida que NumPy**
  y **8,85× que el wheel anterior** en este ensayo. El paso de 231,58 a 34,08 ms
  corresponde al conjunto de mejoras de ejecución; la fusión aporta el paso de
  34,08 a 26,18 ms. No se aislaron pool y escalares entre sí.
- CLBlast ofrece **2,73× sobre NumPy** en matmul 2048² float32. La diferencia
  pequeña en float64 no debe interpretarse como una ventaja consolidada.
- CLBlast no gana siempre al kernel propio. El adaptador inicial usa GEMM también
  para vectores. GEMV/DOT dedicados y kernels que combinen producto/actualización
  siguen siendo oportunidades de trabajo.
- Los solvers pequeños siguen perdiendo. CG de 1024 variables incluso empeora
  frente al wheel anterior. El trazado y las comprobaciones son candidatos a
  perfilar; este ensayo no aísla la causa de esa regresión.
- Los solvers se compararon con recurrencias equivalentes y criterios de residuo
  real, en matrices SPD tridiagonales **almacenadas y procesadas densas**. Esto no
  compara contra un solver disperso especializado ni cubre todos los problemas.
- Se usa la precisión solicitada, sin bajar de float64 a float32, y se conserva
  la comprobación del residuo en cada iteración. El redondeo de coeficientes en
  dispositivo puede alterar las iteraciones; se valida convergencia y solución.
- El decorador traza Python en cada llamada y fusiona una expresión, no un bucle
  completo. No cubre reducciones ni matmul. El backend NumPy no compila ese grafo.

## Evidencia y reproducción

El informe completo local
(`.artifacts/benchmarks/optimization-final-2026-09-06/REPORT.md`, no distribuido)
incluye todos los casos y enlaza muestras, tiempos residentes y validaciones.
Los hashes del runtime permanecieron iguales durante la ejecución final. Se
guardaron snapshots del runner y la suite, y el wheel anterior permanece en
`.artifacts/benchmarks/baseline-0.1.0a1.whl`.

Los ensayos exploratorios previos se conservaron por separado: seis configuraciones
de matmul tuvieron inicialmente un error de construcción de dimensiones del
benchmark, corregido antes de la ejecución final. También se corrigió una
desactivación demasiado amplia de FP_CONTRACT: ahora solo se deshabilita en
expresiones fusionadas. Las cifras anteriores proceden exclusivamente de la
ejecución final completa, con cero errores.

Comando: `python -m benchmarks.optimization --output NUEVA_CARPETA --clblast
--legacy-wheel .artifacts/benchmarks/baseline-0.1.0a1.whl --rounds 7`, con
`OPENPACHO_CLBLAST_LIBRARY` apuntando al binario nativo. Véase
[benchmarks/README.md](../benchmarks/README.md).

Verificación de la implementación: 207 pruebas correctas y 1 omitida en la suite
con CPU/OpenCL; la variante CLBlast también pasó la suite completa antes del
ajuste de FP_CONTRACT (que afecta solo los kernels propios). El adaptador además
dispone de pruebas del ABI simulado y 10 casos de producto nativo. La revisión
adicional de broadcasting, vida útil de buffers y residencia no encontró nuevas
regresiones de corrección. El wheel instalado en el entorno CPU independiente
pasó 102 pruebas, con 99 omisiones de casos que requieren GPU/CLBlast.
Las conversiones flotante→entero fuera de rango siguen
siendo una limitación anterior documentada.

Matrices dispersas y precondicionadores quedan como deuda futura.
