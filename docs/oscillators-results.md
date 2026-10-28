# Osciladores acoplados: flujo iterativo sin observaciones por paso

Esta aplicación evalúa un caso distinto de los solvers hasta convergencia:
trabajo fijo, reutilización de la misma matriz y acumulación de error durante una
trayectoria. Se implementa en `examples/oscillators.py`, fuera del wheel, mediante
operaciones públicas. No añade física, autodiff ni un bucle GPU al runtime.

## Modelo y algoritmo

Para masas unitarias se resuelve `q'' = -K @ q`, con K simétrica definida positiva.
El estado incluye posición q, velocidad v y aceleración a. Un paso velocity-Verlet
calcula:

```
vhalf = v + (dt/2)*a
q1 = q + dt*vhalf
a1 = -(K @ q1)
v1 = vhalf + (dt/2)*a1
```

`integrate` calcula `a0 = -(K @ q0)` una vez y lleva la aceleración de un paso al
siguiente. Un recorrido de N pasos usa N+1 GEMV. Cada repetición parte del mismo
estado original. Las funciones no modifican entradas ni consultan valores host.
`energy` calcula `(v·v + q·Kq)/2` como diagnóstico separado.

La compilación prepara únicamente `verlet_step`; el bucle exterior sigue en
Python y reutiliza el plan. Desenrollar todo el recorrido sería otra estrategia
y no se usa en esta comparación. Se sincroniza al medir el recorrido completo,
sin introducir descargas o reducciones de energía entre sus pasos.

La estabilidad exige `abs(dt)*sqrt(lambda_max(K)) < 2`. El ejemplo valida
metadatos y deja al caller las propiedades de K, finitud y estabilidad. Los
tamaños densos de este experimento no representan necesariamente la forma
apropiada de almacenar un sistema físico local o disperso.

## Referencias y comprobaciones

La comparación de backends debe usar exactamente los inputs ya redondeados al
dtype solicitado. La recurrencia NumPy float64 separa diferencias de ejecución
del error temporal: es mayor precisión para inputs float32 e igual precisión
para float64, no un oráculo exacto.

Para un sistema pequeño, si `K = Q diag(omega²) Q.T`, `c = Q.T @ q0` y
`d = Q.T @ v0`, la solución continua es:

```
q(t) = Q @ (c*cos(omega*t) + d*sin(omega*t)/omega)
v(t) = Q @ (-omega*c*sin(omega*t) + d*cos(omega*t))
```

Se compara la recurrencia discreta con esta trayectoria por separado del error
entre backends. La energía inicial se evalúa en t=0 y la velocidad siempre en
instantes enteros. La energía de Verlet puede oscilar; no se exige disminución
monótona ni deriva final cero. La excursión máxima se obtiene de una repetición
diagnóstica fuera de los tiempos.

Las pruebas CPU verifican un paso escalar, convergencia de segundo orden a tiempo
final fijo, excursión energética acotada, reversibilidad, conteo de productos,
inputs inmutables y rebinding del plan. Ocho pruebas nativas verifican eager,
compiled y prepared en OpenCL, Vulkan y CUDA custom/cuBLAS, float32 y float64.
Todas pasan; Vulkan float64 admite este flujo, que no necesita trascendentes.

## Protocolo de medición

Se usan N=31, 257 y 2049, float32/float64, dt=1/32 y 64 pasos: tiempo final 2.
La semilla 47 genera `K = I + R.T @ R / N` en float64; se simetriza antes de
convertir al dtype y se escala el estado inicial por `1/sqrt(N)`. Generación y
referencias usan siempre un hilo BLAS, fuera de los tiempos de integración.
Se conservan hashes de los inputs efectivos para comparar procesos y proveedores.

La matriz suministrada se verifica simétrica, se comprueba Cholesky numérico
en float64 y se exige `dt*sqrt(norm_inf(K)) < 2`. Cholesky es una comprobación
numérica de positividad, no un certificado matemático exacto. N≤257 añade la
referencia modal y una repetición separada de energía con observación por paso;
N=2049 conserva la referencia discreta y la energía final, sin diagonalizar.

Se comparan NumPy con uno y ocho hilos, OpenCL custom/CLBlast, Vulkan, CUDA
custom/cuBLAS y CuPy. Los backends OpenPacho ejecutan eager y un plan preparado
de un paso. Las dos modalidades se intercalan en 12 rondas equilibradas; cada
backend tiene un proceso independiente y se repite en orden inverso.

Preparación, subida inicial, aceleración de ejemplo, primera llamada y
calentamiento se registran por separado. Cada muestra residente calcula de nuevo
la aceleración inicial y realiza los 64 pasos sobre inputs residentes; la muestra
completa incluye subir K/q/v y descargar q/v/a. Las validaciones quedan fuera del
reloj. Se conservan las muestras, errores y fallos; no se elimina un tiempo porque
su resultado sea incorrecto. La energía inicial/final y el error modal también
se evalúan fuera de los tiempos.

Residente y completo son muestras distintas. La variación puede hacer que una
mediana completa sea menor que su mediana residente; su resta no representa un
coste de transferencia y no se usa para descomponer el tiempo.

El benchmark registra bytes de entradas/salidas y no mide el pico de memoria.
Estos tamaños no establecen rendimiento por encima de la caché ni un límite de
ancho de banda. Las condiciones locales siguen siendo RTX 5060 Ti, Windows y
driver 617.14; no se extrapolan a otras GPU.

```powershell
python -m benchmarks.oscillators --backend numpy --profile quick --repeats 12 --output .artifacts/oscillators-new.json
python -m benchmarks.oscillators --backend cuda-custom --profile all --repeats 12 --output .artifacts/oscillators-cuda-new.json
```

Las ejecuciones GPU requieren exclusividad y reposo observado antes de iniciar
cada proceso. Los artefactos locales se conservan en
`.artifacts/iterations-2026-10-22/oscillators/`, con la sesión y los informes
individuales; `oscillators-native.xml` conserva las ocho pruebas nativas.

## Resultados locales

En esta carga, NumPy con ocho hilos es más rápido que todas las rutas GPU medidas,
tanto residente como completa, en ambos dtypes y los tres tamaños. Compilar un
paso sí mejora consistentemente OpenPacho frente a eager, pero esa mejora no
basta para superar la referencia CPU elegida. Para N=2049 float64, CUDA custom
compilado tarda 17.806–18.011 ms residente frente a 58.222–63.942 ms eager;
NumPy ocho hilos tarda 11.752–12.572 ms. El flujo completo mantiene esa conclusión.

La pared GPU cambia poco entre N=31 y N=2049, mientras el coste NumPy crece con
el tamaño. Esto justifica estudiar despacho/reutilización del bucle antes de
añadir otro kernel GEMV; sin eventos GPU específicos de este recorrido no se
atribuye cuantitativamente la diferencia a Python o al driver. Tampoco se amplían
artificialmente las matrices para convertir esta medición en una victoria GPU.

La aplicación aporta un flujo reproducible y una recomendación acotada de backend,
no una modificación del runtime. En los sistemas con referencia modal, el error
temporal de Verlet es mayor que las diferencias observadas entre backends: cambiar
de float32 a float64 no elimina el error de usar este paso temporal.

Las 14 ejecuciones completan 156 filas aceptadas y 3.744 muestras
residentes/completas. Los hashes de inputs, runtime, ejemplo y runner coinciden
entre los procesos. Las tablas muestran rangos de medianas de las dos rondas,
en milisegundos por recorrido de 64 pasos; no son intervalos de confianza.

### float32

| Implementación | N31 residente | N31 completo | N257 residente | N257 completo | N2049 residente | N2049 completo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numpy-1t | 0.717–0.731 | 0.713–0.773 | 1.061–1.062 | 1.062–1.064 | 14.667–14.729 | 14.400–15.191 |
| numpy-8t | 0.713–0.723 | 0.706–0.720 | 1.058–1.058 | 1.058–1.116 | 8.570–9.253 | 8.158–9.168 |
| opencl-eager | 44.913–53.373 | 40.999–48.919 | 42.758–46.266 | 47.239–50.018 | 43.234–47.489 | 43.716–44.575 |
| opencl-compiled | 18.943–26.317 | 19.161–27.352 | 22.377–23.182 | 22.193–24.180 | 19.000–19.837 | 21.921–22.388 |
| opencl-clblast-eager | 42.690–47.087 | 44.814–49.177 | 41.899–49.103 | 41.451–50.105 | 44.875–45.283 | 42.722–45.421 |
| opencl-clblast-compiled | 20.837–23.468 | 20.478–20.989 | 20.513–27.558 | 20.623–27.244 | 20.056–21.831 | 22.622–23.441 |
| vulkan-eager | 62.037–63.766 | 63.346–65.112 | 62.576–73.770 | 64.606–67.502 | 61.263–81.046 | 62.653–78.739 |
| vulkan-compiled | 18.112–19.516 | 18.879–18.955 | 19.106–19.848 | 19.758–20.760 | 17.699–20.978 | 19.224–21.744 |
| cuda-custom-eager | 60.117–62.197 | 59.347–59.387 | 62.184–66.478 | 59.725–68.805 | 58.959–61.645 | 63.191–68.546 |
| cuda-custom-compiled | 17.579–17.621 | 17.947–18.076 | 18.328–18.567 | 18.108–18.978 | 18.106–18.296 | 20.834–23.223 |
| cuda-cublas-eager | 58.558–58.663 | 62.553–63.052 | 64.029–70.443 | 59.620–69.676 | 60.521–63.052 | 61.504–64.686 |
| cuda-cublas-compiled | 18.422–18.929 | 18.660–19.021 | 19.505–21.653 | 19.401–19.606 | 18.971–19.211 | 21.909–22.245 |
| cupy-eager | 14.991–15.194 | 15.284–15.334 | 15.155–16.364 | 15.691–16.003 | 15.402–16.148 | 16.247–17.760 |

### float64

| Implementación | N31 residente | N31 completo | N257 residente | N257 completo | N2049 residente | N2049 completo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numpy-1t | 0.681–0.695 | 0.674–0.688 | 1.328–1.332 | 1.326–1.355 | 27.644–29.461 | 27.412–29.292 |
| numpy-8t | 0.670–0.679 | 0.680–0.682 | 1.288–1.410 | 1.287–1.333 | 11.752–12.572 | 10.995–11.933 |
| opencl-eager | 43.799–46.097 | 45.547–46.023 | 47.316–48.534 | 45.522–47.421 | 43.465–47.175 | 45.213–51.313 |
| opencl-compiled | 20.840–21.582 | 20.494–20.710 | 23.135–26.860 | 19.909–21.993 | 18.885–20.421 | 24.123–27.328 |
| opencl-clblast-eager | 43.055–53.785 | 40.954–52.316 | 41.576–46.281 | 41.020–49.449 | 40.112–44.307 | 45.514–47.912 |
| opencl-clblast-compiled | 20.591–26.980 | 21.364–28.178 | 21.926–25.881 | 20.916–21.185 | 19.894–25.658 | 24.819–26.016 |
| vulkan-eager | 66.628–72.313 | 68.901–71.502 | 64.030–67.724 | 67.771–71.295 | 62.423–70.050 | 64.925–67.081 |
| vulkan-compiled | 19.665–19.811 | 20.448–21.081 | 19.002–19.214 | 18.905–19.239 | 18.873–20.224 | 23.070–24.708 |
| cuda-custom-eager | 59.523–61.556 | 58.802–62.116 | 58.927–62.371 | 64.388–65.933 | 58.222–63.942 | 63.327–67.844 |
| cuda-custom-compiled | 18.549–21.822 | 17.950–18.540 | 17.806–19.734 | 18.130–18.944 | 17.806–18.011 | 22.881–23.427 |
| cuda-cublas-eager | 58.766–59.594 | 59.261–61.730 | 61.607–65.043 | 61.732–74.093 | 59.879–64.922 | 65.140–71.768 |
| cuda-cublas-compiled | 18.889–18.968 | 18.912–19.528 | 18.617–18.890 | 18.906–19.355 | 19.498–20.004 | 24.661–25.143 |
| cupy-eager | 16.182–16.562 | 16.014–16.942 | 15.670–15.939 | 15.782–16.461 | 15.803–17.774 | 18.313–21.487 |

### Diferencias numéricas

Se exigió `rtol=2e-4, atol=2e-6` para float32 y `rtol=2e-10, atol=2e-12`
para float64 frente a la recurrencia NumPy float64 de los mismos inputs.
Los máximos absolutos observados en los diagnósticos guardados son:

| Dtype | Posición q | Velocidad v | Aceleración a |
| --- | ---: | ---: | ---: |
| float32 | 1.539e-07 | 1.370e-07 | 2.709e-07 |
| float64 | 1.665e-16 | 2.515e-16 | 4.718e-16 |

La referencia modal mide otro error: el de discretizar el tiempo con Verlet.
La excursión energética es el máximo valor absoluto de `(E(t)-E(0))/E(0)`
en 65 instantes, a través de los proveedores y de ambas rondas.

| N / dtype | Error temporal máximo q | Error temporal máximo v | Excursión energética observada |
| --- | ---: | ---: | ---: |
| 31 / float32 | 8.607e-05 | 1.372e-04 | 3.938e-04–3.939e-04 |
| 31 / float64 | 8.607e-05 | 1.372e-04 | 3.939e-04–3.939e-04 |
| 257 / float32 | 6.862e-05 | 8.378e-05 | 3.693e-04–3.693e-04 |
| 257 / float64 | 6.862e-05 | 8.378e-05 | 3.693e-04–3.693e-04 |

N=2049 no tiene referencia modal ni excursión por paso en esta medición;
sí conserva la referencia discreta y el cambio de energía final.

Se registran 5.916 llamadas de validación y 12.480
diagnósticos individuales guardados. El calentamiento valida todas sus
llamadas y conserva únicamente el último conjunto de métricas para acotar
el informe; cualquier fallo se retiene. Ninguna fila falló numéricamente.

La verificación dirigida comprende 21 pruebas CPU del ejemplo, ocho nativas y
11 del protocolo de benchmark: 40 aprobadas. No se ejecutó la suite global.

QA independiente: **APPROVE**. Se reejecutaron las 32 pruebas CPU y se comprobaron
las 156 celdas de tiempos, tablas numéricas, hashes, referencias físicas y 12
puertas de reposo. Los seis inputs se recrearon bajo BLAS de un hilo. Los 33
archivos del runtime son idénticos al estado anterior; la revisión nativa se basa
en el XML guardado de ocho casos, sin nuevas ejecuciones GPU durante QA.
