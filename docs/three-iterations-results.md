# Tres iteraciones: GEMM rectangular, vectores CUDA y layouts compilados

Fecha: 2026-09-24. Hardware de validación: NVIDIA RTX 5060 Ti, Windows,
driver 616.64. Esta evidencia no certifica otros fabricantes ni representa un
límite físico del hardware.

## Cambios y selección

1. **GEMM custom con pocas filas.** `_gemm_rect.py` conserva la acumulación
   secuencial en K y usa 64 hilos por grupo. OpenCL calcula tiles 16×32 con K=16;
   CUDA calcula 16×16 con K=32. La ruta habitual calcula 32×32. Reducir el tile
   puede repartir mejor estas matrices, pero empeora otros tamaños: no se aplica
   universalmente ni se comparte una configuración a ciegas entre backends.
2. **Productos vectoriales cuBLAS.** DOT escribe el resultado directamente en
   dispositivo. GEMV admite matrices por lotes con un vector compartido, flags de
   transposición y offsets contiguos. La política final se decide con llamadas
   públicas sincronizadas, no por el nombre de la primitiva nativa.
3. **Transpuestas privadas compiladas.** El compilador consulta un hook opcional
   del backend y CLBlast recibe flags y leading dimensions. Una transpuesta 2D
   completa de único consumidor puede omitirse si no es una salida pública.
   Eager, transpuestas compartidas, otros proveedores y layouts no admitidos
   conservan sus copias. No cambia la sintaxis NumPy del programa del usuario.

La selección rectangular exige float32, capacidades de register-GEMM existentes,
dos matrices 2D, `32 <= M < 96`, `256 <= N <= 4096` y una cuadrícula anterior de
como máximo 224 tiles. OpenCL requiere K≥4081 y CUDA K≥2048. Los casos totalmente
alineados (M/N múltiplos de 32 y K de 16), lotes y regiones mayores conservan su
ruta. Padding explícito mantiene su plan separado. Son fronteras conservadoras
de esta medición, no un autotuner ni garantías universales de superioridad.

Los kernels solo usan instrucciones portables existentes. Las dos variantes
mantienen memoria compartida dentro de los 4 KiB ya requeridos, guardas para
bordes irregulares y la política matemática anterior. El usuario no necesita
ajustar tiles, crear buffers privados ni cambiar sus algoritmos.

## Protocolo

Procesos GPU secuenciales, dirigidos por el agente principal. Los subagentes no
ejecutaron pruebas GPU. Antes de cada proceso: al menos 10 s de reposo, tres
lecturas consecutivas con utilización GPU/memoria ≤5%, temperatura ≤52 °C y
como máximo 2 °C sobre la referencia de reposo. Un timeout detiene el ensayo;
no autoriza saltarse la condición. Compilación, calentamiento, referencias
NumPy y validación quedan fuera del reloj.

Cada caso tiene dos controles anteriores independientes y una variante. Se
intercalan las seis permutaciones de orden por igual, 30 muestras por modo y
scope, en dos rondas con semillas distintas y orden de backends invertido.
`prepared`/`resident` incluye API preparada, asignación de salida y sincronización;
`complete` incluye además subir todos los inputs y descargar el resultado.
La selección del control ocurre fuera del reloj. La aceleración divide la media
geométrica de las medianas de ambos controles por la mediana de la variante.

Los controles rectangulares desactivan solo la nueva configuración de tile.
Los de layouts desactivan el hook únicamente al preparar planes independientes.
Los de cuBLAS extraen únicamente `CUDABackend.matmul` del wheel histórico local
`0.1.0a1`, SHA256
`a577d80ffde9c8be17ab1c2ca9c51278aaf34412e6be84bd0ecd24450fa66e3b`;
el resto del runtime es actual. El literal OpenCL `0.0f` ya estaba corregido en
ese baseline: ninguna ganancia aquí compara contra aquel bug de promoción.

Los reportes contienen muestras, hashes de fuentes, telemetría, planes, rutas,
errores y fallos de tolerancia. Una ejecución completa no equivale a aprobación
numérica. Los artefactos locales están en `.artifacts/next-three/candidates`
y `.artifacts/next-three/confirmation`, conservando también pilotos y candidatos
descartados. El formateo final no modifica el generador rectangular; su AST se
comprobó idéntico y la unión equivalente de condiciones del selector se contrastó
en 4.374 combinaciones (`format-equivalence.json`).

En total: **38.880 muestras públicas**, incluyendo el candidato vectorial luego
acotado, más **2.352 muestras de confirmación Vulkan** con salida preasignada.
Doce procesos de confirmación/medición terminaron correctamente. Todas sus
puertas de reposo registraron `observed-idle`, con esperas de 10,5 a 229,2 s.
Los pilotos exploratorios se conservan aparte y no se suman a esas cifras.

## GEMM rectangular: dos rondas públicas

21 formas por backend y ronda, incluidas regiones que deben conservar el kernel
anterior. Ningún caso seleccionado empeoró el tiempo residente en estas dos
rondas: las ganancias fueron 1,14–1,79× en OpenCL y 1,09–3,09× en CUDA custom.
Estos intervalos describen exclusivamente las formas seleccionadas medidas.

| Producto float32 | Backend | Residente, control → nuevo (ms, ronda 1) | Aceleración residente, rondas 1 / 2 | Completo, rondas 1 / 2 |
| --- | --- | ---: | ---: | ---: |
| 65×8191 @ 8191×2051 | OpenCL custom | 1,650 → 1,024 | 1,61× / 1,60× | 1,055× / 1,054× |
| 64×8191 @ 8191×2051 | OpenCL custom | 1,652 → 0,923 | 1,79× / 1,79× | 1,066× / 1,063× |
| 65×8191 @ 8191×257 | OpenCL custom | 0,945 → 0,549 | 1,72× / 1,70× | 1,173× / 1,170× |
| 65×8191 @ 8191×2051 | CUDA custom | 1,766 → 1,020 | 1,73× / 1,71× | 1,066× / 1,068× |
| 65×8191 @ 8191×257 | CUDA custom | 1,143 → 0,370 | 3,09× / 3,04× | 1,367× / 1,351× |

En los grupos seleccionados, los dos controles anteriores difirieron como máximo
1,55% en sus medianas residentes y 1,02% en las completas. Los controles cuadrados,
lotes y regiones excluidas conservan el código anterior. El ahorro procede de
otra distribución del trabajo; no de eliminar transferencias incluidas en el
control ni de omitir comprobaciones del programa.

### Precisión que esta optimización no resuelve

Los 84 grupos forma/backend/semilla rectangulares conservaron los bits del control.
No se amplió ninguna tolerancia: rtol=atol=5e-5 contra referencia float64 desde los
mismos inputs float32. Persisten siete grupos con elementos fuera del umbral,
idénticos en ambos controles y la variante:

- K=16383: OpenCL tuvo 7/15 elementos y CUDA 6/14, según la ronda.
- `64×8191 @ 8191×4096`, ronda 2: un elemento en cada backend; esta forma ni
  siquiera activa el tile nuevo.
- `65×8191 @ 8191×257`, CUDA, ronda 2: un elemento.

Por ello una mejora de tiempo no convierte todos estos casos en aprobados
numéricamente. La acumulación float32 larga y la cancelación siguen pendientes
de una política de precisión explícita. El caso CUDA de 3× incluye esa alerta.

## CLBlast: omitir transpuestas privadas

28 grupos en dos rondas: doce productos con flags izquierdo/derecho/ambos y dos
covarianzas centradas. Todos pasan la referencia float64 con rtol=atol=5e-5.

| Operación float32 | Residente, control → nuevo (ms, ronda 1) | Aceleración residente, rondas 1 / 2 | Completo, rondas 1 / 2 |
| --- | ---: | ---: | ---: |
| A.T @ B.T, producto lógico 257×513×129 | 0,161 → 0,100 | 1,61× / 1,59× | 1,157× / 1,154× |
| A @ B.T, producto lógico 65×8191×2051 | 1,563 → 0,950 | 1,65× / 1,65× | 1,054× / 1,052× |
| A.T @ B.T, producto lógico 65×8191×2051 | 1,613 → 0,971 | 1,66× / 1,69× | 1,056× / 1,058× |
| Covarianza centrada de 4097×129 | 0,374 → 0,346 | 1,08× / 1,06× | 1,036× / 1,049× |
| Covarianza centrada de 8191×513 | 2,478 → 2,425 | 1,02× / 1,03× | 1,012× / 1,009× |

La covarianza pequeña tuvo 11,45% de diferencia entre controles residentes en
la primera ronda: su 1,08× no es una estimación precisa de ganancia. La segunda
tuvo controles mucho más próximos. En covarianza grande el beneficio es marginal;
no justifica seguir priorizando microajustes de ese patrón. Eliminar copias no
garantiza ganar si otro layout conduce a un kernel BLAS menos eficiente.

Los 28 grupos también coincidieron bit a bit con sus controles materializados,
pero eso es una observación de esta suite, no una garantía de la API. El error
absoluto máximo fue 1,36e-4 y el relativo L2 máximo 1,63e-6; la tolerancia combina
parte absoluta y relativa por elemento. No se asume que error absoluto máximo
deba ser menor que el término absoluto de una tolerancia mixta.

## cuBLAS: seleccionar después de medir

El primer candidato usaba GEMV para todos los productos con un vector. En dos
rondas perdió aproximadamente 4–15% de throughput residente en buena parte de
los casos sin batch. La versión integrada mantiene GEMM para esos productos y
usa DOT para dos vectores y GEMV para matrices con batch explícito. El cambio
de pointer mode queda bajo lock, con recuperación explícita si restaurar HOST
falla; los buffers ya enviados permanecen retenidos.

Las dos rondas finales añaden lotes singleton en ambas orientaciones. Se
mantienen los casos anteriores, los dos dtypes y un GEMM de control. La mejora
batched incluye eliminar cálculo general de índices en Python: el experimento
no separa ese ahorro del kernel GEMV nativo.

| Operación | Aceleración residente, rondas finales 1 / 2 | Completo, rondas 1 / 2 |
| --- | ---: | ---: |
| DOT 257, float32 | 1,12× / 1,11× | 1,126× / 1,025× |
| DOT 8191, float64 | 1,12× / 1,10× | 0,993× / 1,047× |
| DOT 1048573, float32 | 1,10× / 1,11× | 1,016× / 1,003× |
| (4,257,511) @ (511,), float32 | 1,13× / 1,08× | 1,012× / 1,006× |
| (511,) @ (4,511,257), float64 | 1,14× / 1,10× | 1,017× / 1,021× |

DOT largo float64 fue mixto (0,998× / 1,117× residente); no se declara una mejora
universal del 10%. Los singleton batches variaron aproximadamente entre 1,00× y
1,03× residente y el recorrido completo fue mixto. La ruta sin batch recupera
el comportamiento anterior, con diferencias residuales de hasta ~2,3% a la baja
en esta suite, del orden de 0,4–1,9 µs. La nueva selección/recuperación también
añade trabajo host: no se presenta ese coste como eliminado.

El control negativo GEMM tuvo una ganancia completa aparente de 6,8% en una ronda,
sin cambiar su kernel; otro control A/A difirió 9,8% en residente. Esto limita la
interpretación de porcentajes pequeños. Las ganancias completas CUDA son modestas
y ruidosas frente al ahorro residente. Las muestras y controles se conservan.

Los 56 grupos finales pasan las tolerancias vectoriales existentes
(rtol=5e-5/atol=5e-6 en float32; 1e-11/1e-12 en float64). La referencia float64
es más precisa solo para float32. Cambiar la primitiva puede cambiar el árbol de
suma; no se exige igualdad de bits con GEMM. La etiqueta `native` del reporte
significa la política final, que conserva GEMM donde corresponde.

## Vulkan: confirmación aislada de una dirección descartada

El piloto ya ejecutó OpenCL, CUDA y Vulkan secuencialmente, sin benchmarks de
subagentes. Se repitió Vulkan en dos procesos aislados adicionales: seis formas,
cinco tiles candidatos y dos controles del kernel actual, 28 muestras por modo.
Órdenes cíclicos y sus reversos equilibran la posición de los siete modos.
Reposo observado antes de cada proceso; ningún otro proceso de benchmark o test
GPU del proyecto corrió a la vez. Esto controla nuestro trabajo, no garantiza
la ausencia absoluta de actividad del escritorio/driver.

| Producto | Control A / control B, ronda 1 (ms) | Mejor candidato nuevo, ronda 1 / 2 (ms) | Control A / B, ronda 2 (ms) |
| --- | ---: | ---: | ---: |
| 65×8191 @ 8191×2051 | 0,758 / 0,759 | 0,969 / 0,938 | 0,735 / 0,733 |
| 64×8191 @ 8191×2051 | 0,696 / 0,696 | 0,866 / 0,868 | 0,691 / 0,693 |
| 65×4095 @ 4095×2051 | 0,432 / 0,437 | 0,535 / 0,531 | 0,425 / 0,426 |
| 1024×1024 @ 1024×1024 | 0,438 / 0,432 | 0,530 / 0,549 | 0,455 / 0,451 |

Aquí se mide lanzamiento sincronizado hacia una salida preasignada, no la llamada
pública completa. El caso pequeño 257×129×513 es mixto/cercano al ruido; tampoco
justifica promover la variante. Las pérdidas grandes sí se reproducen con
controles muy próximos. Se conserva el kernel Vulkan actual. No se concluye que
Vulkan carezca de margen: solo que reducir estos tiles no lo mejoró. El compilador
y el backend ya producen comportamientos distintos con geometrías similares;
no se atribuye la diferencia a una instrucción concreta sin ISA/perfil adicional.
Las dos confirmaciones Vulkan conservan los bits del control. La segunda detecta
un elemento fuera de tolerancia en `65×8191 @ 8191×2051`, idéntico en ambos
controles y las cinco variantes (−0,10885158 frente a −0,10890857 en float64).
También es una alerta de precisión del kernel existente, no una regresión
introducida por el candidato. La primera ronda pasa el umbral.

## Verificación de integración

- **1.652 pruebas pasan y una se omite**: runtime y herramientas de benchmark,
  con OpenCL, Vulkan y CUDA obligatorios y CLBlast nativo. La omitida requiere
  un segundo dispositivo OpenCL. Se conservan 277 avisos del compilador PyOpenCL.
- Wheel construido desde sdist: **525 pasan y 909 se omiten** en el entorno CPU
  sin bindings GPU; imports opcionales bloqueados durante el smoke inicial.
  El runtime empaquetado coincide byte a byte con las fuentes.
- Ruff check y formato globales pasan; `pip check` no detecta dependencias rotas.
- El workflow CPU incorpora ahora `tests` y `benchmarks` con su extra opcional;
  su ejecución remota y PoCL siguen pendientes.

Las pruebas funcionales aprobadas no anulan las alertas numéricas del benchmark:
son conjuntos de inputs y objetivos diferentes. Las garantías de acumulación
siguen siendo una tarea explícita previa a la primera estable.

## Próxima frontera

Las transferencias siguen reduciendo muchas ganancias observadas con datos
residentes. Las tres mejoras se aplican dentro del compilador/backend, sin
modificar algoritmos de ejemplo para favorecer OpenPacho. La primera release
necesita además endurecimiento de contratos, precisión y distribución; ver
[criterios y estimación](release-readiness.md).
