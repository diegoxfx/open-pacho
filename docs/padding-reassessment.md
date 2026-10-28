# Reevaluación de padding con la base OpenCL corregida

Fecha: 2026-09-19. **Se conserva `padding=True` como opción explícita, desactivada
por defecto.** Encontramos una ventaja reproducible en productos con pocas filas
y K largo, incluso incluyendo relleno y recorte. La mayoría de las formas medidas
empeora; no se justifica activarlo automáticamente ni priorizar su expansión.

La [corrección anterior](kernel-inspection-results.md) eliminó conversiones
accidentales a float64 en cargas OpenCL. Invalidó la ventaja histórica del padding
en el producto grande irregular ensayado entonces. Esta ronda utiliza esa base
corregida en **ambos** lados de cada comparación. No modifica el runtime, sus
kernels, flags matemáticos ni tolerancias.

## Evidencia principal

RTX 5060 Ti, Windows, driver 616.64, float32, proveedores custom. Para
`(65,8191) @ (8191,2051)`, estas son las medianas y aceleraciones de tres rondas
con semillas distintas. Un factor mayor de uno favorece padding.
El control combina las dos medianas sin padding mediante su media geométrica;
la aceleración divide ese valor por la mediana con padding de la misma ronda.

| Backend | Residente sin padding, ms | Residente con padding, ms | Aceleración residente | Aceleración con upload y download |
| --- | ---: | ---: | ---: | ---: |
| OpenCL | 1,643–1,666 | 1,173–1,220 | **1,366–1,400×** | **1,040–1,041×** |
| CUDA custom | 1,759–1,769 | 1,451–1,458 | **1,212–1,214×** | 0,843–0,865× |
| Vulkan | 0,782–0,820 | 1,111–1,212 | 0,677–0,704× | 0,902–0,913× |

El tiempo residente incluye la llamada pública preparada, asignaciones, copias,
relleno, GEMM, recorte y sincronización. No usa entradas preempaquetadas ni elimina
las copias entre invocaciones. La ruta completa añade subir las dos entradas y
descargar el resultado de ese mismo plan preparado; no mide `matmul` eager.
CUDA gana con datos residentes y pierde al incluir transferencias en esta forma.
OpenCL conserva una ganancia pequeña en el recorrido completo.

Los dos controles independientes sin padding difieren como máximo un 0,47% en
este caso OpenCL, 0,33% en CUDA y 1,94% en Vulkan, entre ámbitos y rondas. La señal
residente favorable supera esa variabilidad observada. Esto no es una garantía
estadística para otros tamaños, dispositivos o aplicaciones.

Aquí se pasa físicamente a `(96,8192) @ (8192,2080)`. `explain` informa
**72.101.888 bytes adicionales** de buffers privados y tres dispatches de copia.
Ese valor excluye entradas originales, salida lógica, pools y memoria del driver.
El coste de memoria es real aunque el número de tiles y de iteraciones K del
GEMM ya fuese el mismo antes de rellenar.

## Distribución de resultados

Aceleración residente mínima–máxima entre las tres rondas. Todas las formas se
conservan en la tabla, incluidas las desfavorables. Los rangos son variación entre
rondas, no intervalos de confianza.

| M,K,N / lotes | OpenCL | CUDA custom | Vulkan |
| --- | ---: | ---: | ---: |
| 257,255,259 | 0,373–0,563× | 0,448–0,455× | 0,544–0,608× |
| 1024,1024,1024, alineado | 0,994–1,000× | 1,002–1,011× | 1,008–1,015× |
| 1025,1023,1027 | 0,910–0,917× | 0,833–0,868× | 0,766–0,832× |
| 2049,2047,2051 | 0,954–0,955× | 0,999–1,010× | 0,896–0,902× |
| 4095,4095,4095 | 1,006–1,008× | 0,919–0,928× | 0,857–0,866× |
| 4097,4093,4101 | 1,002–1,008× | 0,911–0,963× | 0,777–0,795× |
| 2048,2047,2048 | 0,987–0,988× | 0,965–0,966× | 0,925–0,938× |
| 2047,2048,2048 | 0,986–0,988× | 0,964–0,965× | 0,927–0,936× |
| 2048,2048,2049 | 0,993–0,996× | 0,970–0,980× | 0,937–0,943× |
| 1024,8191,1024 | 0,949–0,951× | 0,945–0,947× | 0,898–0,899× |
| 65,8191,2051 | **1,366–1,400×** | **1,212–1,214×** | 0,677–0,704× |
| 4097,255,4099 | 0,825–0,830× | 0,425–0,433× | 0,385–0,391× |
| 8191,513,257 | 0,907–0,927× | 0,768–0,848× | 0,726–0,784× |
| 127,2047,8191 | 0,796–0,803× | 0,699–0,714× | 0,706–0,713× |
| lote 4; 257,255,259 | 0,524–0,587× | 0,531–0,546× | 0,581–0,663× |
| lotes (2,1)/(1,3); 257,511,259 | 0,623–0,720× | 0,611–0,636× | 0,641–0,701× |

Solo una de las quince formas irregulares supera el 5% residente en las tres
rondas de OpenCL y CUDA. Ninguna lo hace en Vulkan. Las mejoras inferiores al 1%
de OpenCL en matrices grandes no cambian la decisión. El caso completamente
alineado no rellena ni copia: pequeñas diferencias allí corresponden a
especialización de guardas o variación, no prueban valor del padding físico.

## Comprobación de tamaños cercanos

Antes de cerrar la decisión se probaron siete formas cercanas en OpenCL y CUDA,
con dos rondas nuevas y la misma metodología: **28 grupos y 5.040 muestras**.
El beneficio sobrevive en varias formas, pero no se extiende uniformemente:

| M,K,N | OpenCL residente | CUDA residente | OpenCL completo | CUDA completo |
| --- | ---: | ---: | ---: | ---: |
| 65,8191,2051 | 1,394× | 1,213–1,218× | 1,041× | 0,863× |
| 64,8191,2051 | 1,469–1,472× | 0,918–0,920× | 1,045–1,047× | 0,833–0,835× |
| 127,8191,2051 | 1,153–1,159× | 0,608–0,610× | 1,016–1,018× | 0,775–0,794× |
| 256,8191,2051 | 0,871× | 0,775–0,778× | 0,975–0,977× | 0,804–0,821× |
| 65,4095,2051 | 1,290–1,343× | 1,167–1,178× | 1,032–1,037× | 1,025–1,026× |
| 65,8191,2048 | 1,431–1,437× | 0,847–0,873× | 1,043–1,045× | 0,830–0,839× |
| 65,8192,2051 | 1,376–1,403× | 1,215–1,223× | 1,039–1,041× | 0,864–0,865× |

El caso favorable principal vuelve a ganar con datos distintos. Reducir M de 65
a 64 o alinear N basta para invertir la ventaja CUDA; OpenCL se comporta de otra
manera. Una regla basada solo en «K largo» o «tamaño irregular» sería insuficiente.
Este seguimiento selecciona vecinos de una forma descubierta en la suite; no
debe interpretarse como una muestra imparcial de todos los workloads posibles.

## Precisión

Los **144 grupos forma/backend/semilla** producen resultados idénticos bit a bit
entre `off-a`, `on` y `off-b`; la repetición de cada ámbito también conserva sus
bits. Eso demuestra ausencia de cambios numéricos observados por padding, no
exactitud universal de GEMM float32.

Se mantiene `rtol=5e-5, atol=5e-5` contra NumPy float64 construido desde los inputs
ya cuantizados a float32. OpenCL y Vulkan pasan esa comparación en los 48 grupos
de cada backend. CUDA presenta un elemento fuera de tolerancia en cada uno de
estos tres grupos, **con y sin padding**:

- Semilla 972, `(1024,8191) @ (8191,1024)`.
- Semillas 972 y 973, `(65,8191) @ (8191,2051)`.

Para semilla 972 y la segunda forma, el elemento `[16,653]` es
`0.05214071273803711`, frente a `0.05208303933545566` en la referencia. Reproducir
en CPU los productos y la suma secuencial redondeados a float32 da exactamente
el valor CUDA. Es evidencia de acumulación/redondeo con K largo y cancelación,
no de corrupción introducida por el relleno. Mejorar esta acumulación queda como
trabajo independiente; no se cambió FMA ni se relajó el umbral.

En la suite principal, el máximo error absoluto es 1,518e-4 en OpenCL/Vulkan y
1,613e-4 en CUDA; el máximo error relativo L2 es aproximadamente 1,630e-6. Una
métrica global pequeña puede coexistir con un elemento fuera de tolerancia.
Ver también el [seguimiento numérico](numerical-accuracy.md).

Los 28 grupos del seguimiento también conservan los bits entre modalidades y
pasan el umbral con sus semillas. Sus máximos errores absolutos son 1,691e-4
en OpenCL y 1,634e-4 en CUDA; error relativo L2 máximo de aproximadamente 1,629e-6.
En total hay **172 grupos y 30.960 muestras**, con 22 formas distintas; el caso
principal aparece en ambas suites con inputs distintos. El seguimiento no
elimina los tres fallos de tolerancia registrados en la suite principal.

La primera tentativa de la segunda ronda CUDA se detuvo en el control sin
padding por esa tolerancia. Se conserva ese informe incompleto y se excluye de
las tablas. El worker se ajustó para registrar violaciones finitas y continuar
los demás modos, manteniendo datos, flags y tolerancias. Ahora `complete` indica
ejecución terminada y `all_reference_close` indica precisión; no son equivalentes.

## Método y reproducción

La suite principal tiene 16 formas, tres backends y tres semillas: nueve procesos
completos, 144 grupos, 432 modalidades y **25.920 muestras cronometradas**. Cada
grupo usa tres planes independientes (`off-a`, `on`, `off-b`), los mismos inputs,
30 muestras por modalidad y ámbito, y los seis órdenes posibles con igual
frecuencia. Preparación, calentamiento mínimo de 0,25 s, referencias, validación
y telemetría quedan fuera del reloj. Se guarda cada muestra y el hash del código.

Los procesos GPU son secuenciales. Antes de cada proceso se observa al menos
10 s de reposo, tres lecturas consecutivas con utilización GPU/memoria de hasta
5%, temperatura de hasta 52 °C y dentro de 2 °C de la referencia de reposo; se
espera hasta 300 s sin saltar el control. El orden de backends y casos varía.
Las trece esperas previas seleccionadas duraron entre 10,4 y 62,2 segundos.
Estos controles no fijan frecuencias ni garantizan equilibrio térmico durante
toda la ejecución. Las medidas son de pared, no tiempo puro del dispositivo.

```bash
python -m benchmarks.isolated --worker padding_decision --backends opencl,vulkan,cuda-custom --rounds 3 --repeats 30 --seed 971 --max-wait-seconds 300 --max-temperature 52 --output .artifacts/padding-decision/reproduction
```

Requiere los runtimes GPU y el compilador Vulkan configurados como las otras
suites. El worker también acepta `--case-indices` al ejecutarse directamente;
por ejemplo, el índice 10 selecciona la forma favorable. El nuevo benchmark
registra violaciones numéricas sin abortar las otras modalidades; su código de
salida por ejecución completada **no certifica** `all_reference_close`.

Los resultados locales principales están en `.artifacts/padding-decision/broad`
(solo ronda 1) y `continuation` (rondas 2 y 3). `broad/2-cuda-custom.json` es la
tentativa incompleta excluida. `final-metrics.json` conserva la selección y hashes
de informes. Los datos crudos y herramientas privadas no se incluyen en el wheel.

El seguimiento está en `.artifacts/padding-decision/neighbors`. Su adaptador
`neighbor_worker.py` llama al mismo `benchmarks.padding_decision.run` con las
siete formas de la tabla, en ese orden, semillas 971 y 972 y 30 muestras. Los
inputs se generan con `default_rng(semilla * 101 + índice_de_forma)` y uniformes
en `[-0.5,0.5)` convertidos a float32. Por eso el caso repetido, de índice 0 aquí
y 10 en la suite principal, tiene otros valores. Los hashes identifican tanto
el worker compartido como el adaptador de formas.

También se corrigieron literales `0.0` a `0.0f` en el generador **experimental**
de `benchmarks.kernel_variants`, donde permanecía la misma promoción accidental.
La prueba de los tres lenguajes protege esa base para futuros experimentos; el
benchmark de esta decisión usa el generador del runtime, que ya estaba corregido.

## Decisión y siguiente prioridad

Mantener el contrato actual y `False` por defecto. Recomendar `True` únicamente
después de medir el workload completo y su memoria; en este equipo, no hay razón
para recomendarlo en Vulkan. No introducir una heurística automática a partir de
una sola GPU, ni extrapolar a todos los productos con K largo.

La siguiente prioridad es mejorar el GEMM sin padding para formas rectangulares
y su precisión de acumulación. Alineación de accesos, tratamiento de bordes y
selección de tiles pueden capturar parte del beneficio sin materializar entradas
completas. Esta ronda cambia a la vez stride, guardas y envío, por lo que no
atribuye toda la mejora a eliminar branches. Tampoco constituye un límite físico
de ningún backend. La posible extensión a `mean` sigue [anotada](padding.md#extensión-futura-reducciones),
sin implementación ni prioridad adicional en esta ronda.
