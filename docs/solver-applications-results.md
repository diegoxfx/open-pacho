# Solvers, aplicaciones y preparación de CUDA

Continuación del 2026-09-12: [Vulkan por capacidad y CUDA inicial](cuda-initial-results.md).
Las cifras y el estado de CUDA de este documento corresponden a la iteración histórica siguiente.

Iteración del 2026-09-08, RTX 5060 Ti / Ryzen 7 5700X3D / Windows 11 / driver
616.64. Continúa [Vulkan, GEMM y estabilidad](numerics-vulkan-results.md).
Solo este hardware tiene validación física. Los resultados anteriores permanecen
como históricos; no se sobrescriben para atribuirles el código nuevo.

## Cómo interpretar las victorias anteriores

En la ronda anterior, OpenCL custom ganaba por mediana **8/9 casos residentes
frente a la variante OpenPacho con CLBlast**, y 9/9 incluyendo transferencias.
La excepción residente era CG 1024 float64. Frente a CuPy ganaba 8/9 residentes
y 7/9 completos. Son comparaciones de solvers completos, no una demostración de
superioridad de cada kernel. CLBlast aporta productos BLAS dentro de nuestro solver;
no se está comparando contra un solver propio de CLBlast.

CuPy superaba a descenso más pronunciado custom en 2048 float32; custom ganaba
en 512 float32 y 1024 float64. Los tres casos conservaban iguales iteraciones.
Por tanto, no era una derrota general de SD ni una victoria universal de CG.
Los casos son matrices tridiagonales SPD almacenadas densamente, condición ≤9;
esa familia acotada no representa toda aplicación densa.

También había una victoria en una operación concreta: GEMM Vulkan custom 2048²
float32 residente era 3.047 ms frente a 3.929 ms con CLBlast. OpenCL custom estaba
en 6.902 ms y CuPy en 1.419 ms. Son formas, rutas y modalidades distintas; no se
puede resumirlo como que todos nuestros kernels ya superan a CLBlast.

## Cambios implementados ahora

- **Curvatura sin kernel booleano.** La decisión host lee directamente el escalar
  existente, verifica finitud/positividad y mantiene su valor GPU para actualizar
  en el mismo dtype. Se elimina un dispatch por iteración CG/SD; se conserva la
  observación host y no se acepta un candidato con curvatura inválida.
- **Planes entre observaciones.** GEMV/DOT se agrupan para curvatura; SD incluye
  su normalización. El avance agrupa actualización, residuo verdadero y norma
  escalada, devolviendo candidato y estado residual. No se relaja ninguna
  tolerancia ni se omiten chequeos por iteración.
- **Un command buffer Vulkan por grupo.** Dispatches dependientes de una llamada
  compilada comparten grabación y mantienen barreras. Los descriptor sets y
  buffers siguen retenidos hasta completar la cola. El límite de 64 cuenta
  operaciones, no command buffers, para conservar el límite de recursos en vuelo.
  Sincronización, presión de caché o una excepción cierran el grupo de forma segura.
- **Separación del compilador por backend.** `_lowering.py` concentra código y
  ejecución específicos; `_compiler.py` conserva el grafo y vida de temporales.
  Es la base implementada del [camino CUDA](cuda-backend-plan.md), no un backend
  CUDA disponible todavía.
- **Reutilización Vulkan al cambiar de forma.** El pool expulsa clases de tamaño
  antiguas cuando llega un buffer que necesita espacio. Antes, las primeras 128
  reservas libres podían bloquear indefinidamente la admisión de tamaños nuevos.
  La expulsión solo afecta buffers completados sin referencias vivas; mantiene
  límites de 128 buffers y 128 MiB y separa memoria GPU, subida y readback.

## Solvers: nueva ronda completa

63 configuraciones, siete muestras residentes y completas por configuración.
Mismos datos, dtypes y tolerancias del benchmark anterior. Se alternan las
implementaciones entre rondas y los diagnósticos quedan fuera del cronómetro.
Esta ronda y los controles de fases siguientes preceden a la corrección final
del pool Vulkan; se conservan con esa identificación.

Medianas residentes 2048 float32, en milisegundos:

| Implementación | Jacobi (37 iter.) | SD (30 iter.) | CG (13 iter.) |
| --- | ---: | ---: | ---: |
| NumPy 8 hilos | 9.067 | 14.720 | 8.312 |
| OpenCL custom | 9.056 | 14.491 | 6.812 |
| OpenCL + CLBlast | 14.045 | 20.950 | 8.715 |
| Vulkan | 20.586 | 33.226 | 15.593 |
| CuPy | 10.205 | 17.057 | 8.697 |

En el mismo orden, los residuos relativos independientes de custom son
8.907e-5 / 9.567e-5 / 7.086e-5; los de CuPy son
8.907e-5 / 9.568e-5 / 7.086e-5. El error relativo de solución respecto a la
solución generadora es aproximadamente 8.874e-5 / 3.490e-4 / 1.712e-4 en ambos.
El RHS se redondea al generar el sistema; esa solución no es un oráculo exacto
del sistema redondeado. No se ha acelerado aceptando un residuo mayor.

SD custom ya gana en este caso de la nueva ronda. Vulkan mejora frente a los
41.966 ms históricos de SD y 20.424 de CG, pero conserva aproximadamente 2.3× el
tiempo de custom en ambos. Jacobi Vulkan apenas cambia frente a los 21.533 ms
históricos. No se trata toda diferencia entre sesiones como ganancia causal.

## Controles de ejecución y explicación de SD

Un control separado alterna procesos con el wheel anterior y el código nuevo,
once muestras por método, mismos sistemas 2048 float32. En dos rondas, Vulkan
SD pasa de 34.64–41.05 a 24.62–24.63 ms; CG de 15.85–23.29 a 12.75–12.87 ms;
Jacobi de 19.61–24.80 a 16.41–16.71 ms. Los historiales y residuos se conservan.
Hay variabilidad considerable: estos controles no sustituyen la suite mixta.
OpenCL SD pasa de 15.84–23.39 a 13.09–13.35 ms, mientras CG/Jacobi tienen rangos
solapados entre versiones. Se conserva ese resultado mixto.

Las fases aisladas sincronizadas, 31 muestras, ayudan a localizar costes:

| Fase float32, 2048 | OpenCL custom | CLBlast | Vulkan | CuPy |
| --- | ---: | ---: | ---: | ---: |
| Normalización | 0.040 ms | 0.045 ms | 0.183 ms | 0.033 ms |
| Curvatura + lectura | 0.185 ms | 0.256 ms | 0.343 ms | 0.224 ms |
| Avance + residuo + norma + lectura | 0.175 ms | 0.535 ms | 0.408 ms | 0.341 ms |
| Actualización de dirección CG | 0.045 ms | 0.043 ms | 0.225 ms | 0.086 ms |

Son fases sobre un estado fijo, con sincronización adicional: **no se suman para
reconstruir el tiempo de un solve ni representan tiempo GPU aislado**. Explican
oportunidades verificables, no un reparto exacto de la latencia histórica.

Por código, CG fusiona una actualización de dirección que el comparador CuPy
expresa con varias operaciones; SD solo normaliza un vector en esa posición.
Así, la ventaja de nuestra fusión depende del método. SD además paga GEMV, DOT,
curvatura y residuo durante más iteraciones que CG en estos casos. Esto no hace
inherentemente lento a SD dentro de OpenPacho: los cambios de agrupación y
validación reducen el coste sin modificar la recurrencia. CuPy conserva como
referencia fuerte cuBLAS NRM2 para la norma del solver.

## Aplicaciones y gaps que deben dirigir el siguiente trabajo

La [suite de aplicaciones](application-measurement-plan.md) añade covarianza
muestral y veinte pasos de regresión ridge con matrices rectangulares. Se
publican preparación, transferencia, error, objetivo y norma del gradiente.
Usa datos sintéticos y patrones aplicados; no son datasets de producción ni
pruebas de convergencia de ridge. Los resultados completos quedan en
`.artifacts/benchmarks/next-applications/applications-final.json`.

La ronda aplicada final, después de corregir el pool, terminó con **72
configuraciones**, siete muestras por modalidad y cero errores/ausencias. Para
16384×256 float32:

| Implementación | Covarianza residente/completa | Error L2 covarianza vs float64 | Ridge residente/completo |
| --- | ---: | ---: | ---: |
| NumPy 8 hilos | 6.711 / 6.389 ms | 1.161e-7 | 5.313 / 14.427 ms |
| OpenCL custom compiled | 3.708 / 5.994 ms | 1.907e-6 | 5.667 / 10.413 ms |
| OpenCL + CLBlast compiled | 1.529 / 3.825 ms | 1.907e-6 | 11.777 / 14.443 ms |
| Vulkan compiled | 1.741 / 4.579 ms | 1.906e-6 | 7.386 / 7.961 ms |
| CuPy | 0.876 / 3.925 ms | 1.125e-7 | 4.716 / 6.211 ms |

Esto descubre oportunidades que no mostraban los solvers cuadrados: CLBlast
supera a nuestro OpenCL custom aproximadamente 2.4× en covarianza residente,
mientras custom gana en ridge. Vulkan está a 1.14× CLBlast en covarianza y gana
a NumPy; en ridge queda a 1.30× custom y 1.57× CuPy en modo residente. El backend no tiene una única relación
de rendimiento que pueda extrapolarse a todas las aplicaciones.

El error absoluto máximo de covarianza es 2.269e-8 para custom/CLBlast,
2.223e-8 para Vulkan y 1.116e-9 para CuPy. Hay margen numérico además de tiempo:
un resultado dentro de tolerancia no significa idéntica precisión. En ridge,
los errores L2 de custom/Vulkan/CuPy rondan 6.6–6.9e-8; CLBlast queda en 4.401e-7.
El objetivo final es aproximadamente 7712.505 y la norma del gradiente 998.429
en todas las rutas: los veinte pasos **no han convergido**. La comparación mide
una trayectoria común, no el tiempo necesario para resolver completamente ridge.

La preparación de la transpuesta de ridge cuesta 9.435 ms en NumPy 8 hilos,
0.412 en OpenCL custom, 0.495 en Vulkan y 0.347 en CuPy. Por eso se publica por
separado: residencia/preparación cambia el valor práctico de cada resultado.

Antes de la corrección del pool, esta misma suite registraba ridge Vulkan en
30.091 ms residentes; la ronda final mide 7.386 ms. La comparación entre sesiones
incluye ruido: el control de política siguiente aísla la causa con mayor precisión.

Vulkan todavía necesita reducir coste de envío/lectura escalar y staging grande.
Compartir command buffers reduce gestión de comandos, pero no elimina llamadas
Python/CFFI, barreras, descriptor updates ni esperas host. Las transferencias no
se optimizaron en esta iteración; no se atribuye a PCIe toda su diferencia con
OpenCL. Un ejecutor nativo pequeño y staging reutilizable siguen como candidatos
que deben medirse antes de incorporarse.

Para CuPy, el objetivo es acercarse a su rendimiento con contrato comparable.
Su velocidad no determina nuestro límite físico: usa el mismo dispositivo.
Las bibliotecas optimizadas y su acceso al compilador/driver explican posibles
ventajas de software, pero falta perfilarlas para asignar causas. CuPy puede
delegar operaciones a bibliotecas especializadas; no toda operación tiene la
misma ruta. [Guía de rendimiento CuPy](https://docs.cupy.dev/en/v14.0.0/user_guide/performance.html).
No se atribuye su ventaja publicada a TF32, que está deshabilitado en las comparaciones.

CUDA es el siguiente backend propuesto. HIP/ROCm, Metal y SYCL tienen valor
potencial con sus plataformas de prueba, pero no se implementan simultáneamente.
La [decisión de backends](cuda-backend-plan.md) detalla etapas, contrato y fuentes.

## Dimensiones irregulares y el problema de reutilización Vulkan

Se añadieron **144 pruebas** en NumPy, OpenCL custom, CLBlast y Vulkan, con
float32/64: GEMM rectangular, dimensiones impares alrededor de los tiles,
dimensiones pares que no son potencias de dos, broadcasting por lotes, GEMV,
reducciones por eje y solvers cuadrados SPD de tamaños 31, 65 y 129. Comprueban
precisión, residuo verdadero, dtype, inputs inmutables y reutilización del plan.
Las matrices rectangulares se ejercitan también en covarianza y ridge; Jacobi,
CG y SD conservan su contrato de sistema cuadrado.

La primera ronda irregular reveló ridge Vulkan 4097×129 mucho más lento que
4096×128. Un perfil mostró cientos de `vkAllocateMemory` y `vkFreeMemory` por
cada diez ejecuciones ya calentadas. Cambiar la forma podía dejar el pool lleno
de tamaños anteriores: descartaba todos los nuevos aunque fueran reutilizables.
Eso es un defecto general de admisión del runtime, no una frontera de matrices
impares ni evidencia de una limitación del hardware.

El control `pool_control.py` mantiene el mismo solver/runtime y sustituye solo
la política de reciclaje por la antigua. Procesa 4095×127, 4096×128 y 4097×129
en ese orden, con tres calentamientos y once muestras por forma, incluyendo la
descarga del vector final. El perfil de reservas se toma aparte del cronómetro.

| Forma ridge float32 | Política antigua | Expulsión nueva | Reservas en diez ejecuciones, antes → después |
| --- | ---: | ---: | ---: |
| 4095×127 | 6.539 ms | 7.526 ms | 0 → 0 |
| 4096×128 | 11.621 ms | 6.558 ms | 100 → 0 |
| 4097×129 | 47.911 ms | 6.417 ms | 810 → 0 |

Errores relativos L2 iguales entre políticas: 8.476e-8, 6.474e-8 y 7.769e-8,
respectivamente, frente a la trayectoria float64. El primer caso no mejora;
la reducción de reservas en los siguientes sí demuestra que se elimina la
exclusión permanente de tamaños nuevos. No se extrapola una aceleración universal.

**Límite pendiente:** alternar muchas formas y retener sus arrays puede exceder
la capacidad del pool exacto. Un control intercalado de cinco formas sigue
mostrando churn; expulsar no evita volver a reservar lo que acaba de expulsarse.
Clases de capacidad, subasignación y reducción de temporales en vuelo deben
compararse como siguientes opciones. Aumentar el límite sin medir memoria no
resolvería la causa de manera general.

La ronda final `applications-irregular-final.json` completa otras **72
configuraciones** con siete muestras por modo, sin errores/ausencias. Para
4097×129 float32, medianas de ridge:

| Implementación | Residente | Completo | Error L2 vs trayectoria float64 |
| --- | ---: | ---: | ---: |
| NumPy 8 hilos | 2.575 ms | 3.137 ms | 6.242e-8 |
| OpenCL custom compiled | 5.507 ms | 6.598 ms | 7.424e-8 |
| OpenCL + CLBlast compiled | 6.530 ms | 7.327 ms | 1.948e-7 |
| Vulkan compiled | 7.595 ms | 7.450 ms | 7.766e-8 |
| CuPy | 4.835 ms | 4.972 ms | 7.314e-8 |

Vulkan mejora frente a los 38.030 ms residentes de la ronda irregular anterior,
pero NumPy gana a todas las GPU en este problema. Los modos son muestras
independientes intercaladas: su ruido puede hacer que la mediana completa sea
menor que la residente; no significa que transferir datos tenga coste negativo.
Los JSON conservan muestras y percentiles, incluidas variaciones de las referencias.

## Evidencia reproducible

Los controles de fase/procesos viven en `.artifacts/next-applications`; el
baseline se extrajo de `.artifacts/dist-numerics-vulkan`. Los resultados actuales
están en `.artifacts/benchmarks/next-applications`. `solver-tridiagonal-after-1.json`
es una versión intermedia; `solver-tridiagonal-final-*.json` incorpora los planes
y comandos finales, antes de modificar el pool. `applications*.json` sin sufijo
`final` son rondas anteriores a ese cambio; `applications*-final.json` sí lo incluye.
`pool-old.json` y `pool-new.json` aíslan la política de reciclaje. `solver-before-1.json` es un control exploratorio de otra matriz, que no
se usa en las comparaciones anteriores.

La suite completa de runtime y adaptadores pasa **965 pruebas, con 1 omisión**
por requerir dos dispositivos OpenCL. Incluye 144 casos de formas irregulares,
cuatro regresiones del pool, casos extremos, curvas inválidas, rebinding,
plan independiente del nombre del backend y vida de recursos Vulkan. OpenCL y
Vulkan son obligatorios, y CLBlast nativo está habilitado. Ruff y formato pasan;
la sintaxis se analiza como Python 3.10, sin sustituir una ejecución en esa versión.

El wheel final, construido desde sdist e instalado sin PyOpenCL/Vulkan, pasa
**294 pruebas, con 635 omisiones esperadas** por backends opcionales ausentes.
La importación procede de site-packages fuera de `src`. Distribuciones en
`.artifacts/dist-next-applications`; el manifiesto de fuentes y resultados está
en `.artifacts/benchmarks/next-applications/manifest.json`. Detalles en
[verificación local](verification.md).

La suite adversa se repitió para los tres caminos OpenPacho GPU: **126 casos,
tres muestras, 28 indicadores numéricos, cero excepciones/ausencias**. Se conserva
el conjunto de indicadores de ese mismo subconjunto anterior; no es una reducción
de 68 a 28, porque esta vez no se repitieron los comparadores CPU/CUDA sin cambios.
Los umbrales diagnósticos no son garantías universales de forward error.
