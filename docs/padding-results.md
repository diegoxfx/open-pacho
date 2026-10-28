# Padding opcional: resultados locales — 2026-09-18

**Snapshot histórico.** La [inspección posterior](kernel-inspection-results.md)
detectó y corrigió conversiones float64 innecesarias en la ruta OpenCL sin
padding. Con la base corregida, la ventaja del padding en el caso grande irregular
no se mantiene. Las mediciones siguientes conservan la versión anterior.
La [reevaluación actual](padding-reassessment.md) amplía las formas sobre la base
corregida y conserva la opción por ganancias concretas en productos rectangulares.

El padding aporta una mejora reproducida en los dos GEMM grandes irregulares de
**OpenCL custom**. En las formas medidas de Vulkan y CUDA custom, las copias
adicionales no producen una mejora general. Se mantiene desactivado por defecto.
El [contrato](padding.md) permite elegirlo sin cambiar dimensiones lógicas,
precisión ni proveedor BLAS.

## Método y alcance

Windows, Python 3.12, RTX 5060 Ti, driver 616.64; un solo fabricante/dispositivo.
`benchmarks.padding` ejecutó seis formas float32, incluidas rectangular, impar,
alineada y broadcasting `(2,1,65,47) @ (1,3,47,67)`. Dos rondas con semillas 941
y 942, procesos separados, orden invertido de backends y 21 muestras por variante
y ámbito. Dentro de cada caso se alternan `padding=False` y `True` con los mismos
datos; la preparación y el calentamiento quedan fuera de los relojes repetidos.
La primera ejecución registrada puede incluir compilación de kernels; el plan
Python ya se ha preparado. No mide instalación ni importación inicial.

Los seis controles previos observaron inactividad GPU/memoria ≤5%, tres lecturas
consecutivas, al menos 10 s de pausa, temperatura ≤52 °C y dentro del margen
relativo configurado. Las esperas efectivas fueron **10,5–156,4 s**; no se relajó
el control para completar la sesión. Las temperaturas medias de las ventanas
finales estuvieron entre 41 y 49 °C. Esto limita interferencias entre procesos;
no garantiza frecuencia constante, aislamiento del sistema ni equilibrio térmico.
Hay variación visible entre rondas, especialmente en algunas llamadas pequeñas.

Se guardaron **6 048 muestras públicas**: ejecución compilada residente,
ejecutor preparado, `matmul` directo y recorrido completo con subida/descarga.
Todas incluyen las copias de padding y recorte. Otras **756 muestras** corresponden
al diagnóstico `prepacked_wall`: entradas ya rellenadas, salida física sin
recortar. Este diagnóstico excluye costes necesarios de la API y no es tiempo
puro de GPU: incluye despacho, asignación y sincronización.

## Ejecución compilada residente

Cada intervalo abarca las medianas de las dos rondas, **no** un intervalo de
confianza. La aceleración es `tiempo_sin / tiempo_con`, calculada por ronda;
valores menores de uno indican una regresión al activar padding.

| Backend | M,K,N / lote | Sin padding (ms) | Con padding (ms) | Aceleración por ronda |
| --- | --- | ---: | ---: | ---: |
| OpenCL | 33,17,65 | 0,055–0,057 | 0,132–0,139 | 0,41–0,42× |
| OpenCL | 257,255,259 | 0,093–0,096 | 0,133–0,158 | 0,61–0,70× |
| OpenCL | 1025,1023,1027 | 0,845–0,874 | 0,597–0,647 | **1,35–1,42×** |
| OpenCL | 2049,2047,2051 | 6,002–6,503 | 3,760–4,237 | **1,53–1,60×** |
| OpenCL | 1024,1024,1024 | 0,505–0,541 | 0,496–0,542 | 1,00–1,02× |
| OpenCL | lote 2×3: 65,47,67 | 0,059–0,082 | 0,142–0,217 | 0,38–0,42× |
| Vulkan | 33,17,65 | 0,141–0,172 | 0,231–0,270 | 0,61–0,64× |
| Vulkan | 257,255,259 | 0,178–0,179 | 0,272–0,283 | 0,63–0,66× |
| Vulkan | 1025,1023,1027 | 0,535–0,541 | 0,671–0,683 | 0,79–0,80× |
| Vulkan | 2049,2047,2051 | 2,826–2,992 | 3,317–3,319 | 0,85–0,90× |
| Vulkan | 1024,1024,1024 | 0,494–0,503 | 0,485–0,503 | 1,00–1,02× |
| Vulkan | lote 2×3: 65,47,67 | 0,168–0,171 | 0,266–0,269 | 0,63–0,63× |
| CUDA custom | 33,17,65 | 0,075–0,076 | 0,216–0,221 | 0,34–0,35× |
| CUDA custom | 257,255,259 | 0,098–0,151 | 0,215–0,462 | 0,33–0,46× |
| CUDA custom | 1025,1023,1027 | 0,529–0,546 | 0,616–0,688 | 0,79–0,86× |
| CUDA custom | 2049,2047,2051 | 3,700–3,720 | 3,689–3,713 | 1,00–1,01× |
| CUDA custom | 1024,1024,1024 | 0,470–0,470 | 0,461–0,466 | 1,01–1,02× |
| CUDA custom | lote 2×3: 65,47,67 | 0,083–0,107 | 0,215–0,306 | 0,35–0,38× |

El control alineado no asigna buffers privados ni hace copias adicionales. Sus
diferencias pequeñas no se consideran una ganancia robusta. Tampoco se interpreta
la diferencia marginal de CUDA en 2049 como una mejora demostrada.

## Coste completo y memoria

En OpenCL, el caso 2049×2047×2051 sigue ganando con transferencias:
**14,51–14,69 ms → 11,95–12,79 ms**, aceleración de **1,15–1,21×** por ronda.
Solicita **51 384 320 bytes privados (49,00 MiB)**, además de entradas y salida
lógica. El caso 1025×1023×1027 solicita 12,50 MiB y pasa de 3,14–3,31 ms a
2,93–3,03 ms completos, 1,07–1,09×. El beneficio residente no puede extrapolarse
sin incluir transferencias.

El diagnóstico OpenCL preempaquetado de 2049 toma 3,52–3,89 ms, frente a
3,76–4,24 ms del camino público residente con todas las copias. En Vulkan toma
2,80–2,84 ms y el camino público 3,32 ms; la ruta ordinaria ya tarda 2,83–2,99 ms.
Estos datos orientan hacia reducir copias y conservar layouts internos en cadenas
compatibles. No prueban qué instrucciones emitió el compilador ni que todas las
guardas del código fuente se conviertan en branches de máquina.

## Error numérico y límites

En las **36 comparaciones emparejadas**, el resultado con padding coincide
elemento a elemento exactamente con el obtenido sin padding en el mismo backend.
Ambos pasan la comparación con una referencia NumPy float64; el mayor error
relativo L2 observado fue **8,11×10⁻⁷** y el mayor error absoluto **4,02×10⁻⁵**.
No aparecieron valores no finitos en estos datos aleatorios. La comprobación por
elemento usa `rtol=5e-5, atol=5e-5`; no es una garantía para matrices arbitrarias.

Las pruebas específicas añaden NaN/Inf, ceros con signo, subnormales y extremos:
las copias conservan palabras uint32 bit a bit, los valores añadidos son cero
positivo exacto y la aritmética se contrasta con la ruta existente del backend.
También verifican recorte antes de `sum`/`mean`, independencia de resultados,
reenlace de entradas, lotes, límites de memoria/índices y conservación de BLAS.
No se cambian flags matemáticos ni se promete identidad binaria entre fabricantes.

## Reproducción y próximos márgenes

Comando y requisitos en [padding](padding.md). Las muestras, metadatos de planes,
hashes de fuentes, telemetría y errores de esta ejecución están en
`.artifacts/padding/measurements`; el resumen está en
`.artifacts/padding/summary.json`. Los artefactos privados no se incluyen en el
wheel. La [verificación](verification.md) registra las suites y el paquete final.

El siguiente margen útil sería amortizar copias conservando padding a través de
operaciones compatibles, con dimensiones lógicas explícitas y reglas para
reducciones/no finitos. También se puede evaluar un contrato explícito para
operandos preempaquetados reutilizables. Esta versión vuelve a copiar los inputs
actuales en cada llamada: no asume silenciosamente que son inmutables. El
autotuning por forma/hardware puede explorar después qué casos justifican el
coste. Los resultados actuales no justifican activarlo globalmente.
