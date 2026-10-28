# Preparar extremos materializados — 2026-10-22

La reducción compilada de un array ya materializado puede reutilizar el lowerer
cooperativo, con un grupo elementwise vacío. La comparación justifica activar
esta ruta para **min/max float32 en OpenCL y CUDA**, conservando la ruta anterior
en Vulkan. No modifica kernels, precisión, proveedores BLAS ni reducciones eager.
Float64, enteros, bool y sum conservan su selección anterior cuando el productor
está materializado; la fusión de productores privados sigue funcionando como antes.

Esta iteración sigue a los [extremos residentes](extrema-results.md). La hipótesis
era concreta: evitar preparación repetida y aprovechar la reducción cooperativa
existente donde la primitiva eager recorre serialmente el eje. No se introduce
un algoritmo de normalización distinto para hacer favorable la comparación.

## Experimento

`benchmarks/extrema_planning.py` prepara tres planes independientes sobre los mismos
datos float32: dos controles eager y un candidato con
`prepare_map_reduce(node, (), node.inputs)`. Solo cambia el paso de reducción de
extremos materializados; los demás pasos, dependencias, salidas y usos se conservan.
El runner fuerza explícitamente ambos modos, incluso después de promover la ruta.
No modifica políticas globales ni atribuye fusión a un productor compartido.

Once casos incluyen reducciones por filas, columnas, ejes no contiguos y total,
un caso pequeño, tres softmax y un logsumexp. Cada proceso usa una semilla distinta
por ronda y conserva los hashes de entradas, runtime, runner y receta. Las entradas
son uniformes en [-4,4]; la prueba de rendimiento no sustituye la cobertura de
NaN, ceros con signo, vacíos o límites de dtype.

Dos rondas, cada una con un proceso por backend, invierten el orden entre
OpenCL/Vulkan/CUDA y CUDA/Vulkan/OpenCL. Antes de cada proceso se exige al menos
diez segundos de reposo, tres lecturas consecutivas de utilización GPU/memoria
≤5%, temperatura ≤52 °C y ≤2 °C sobre la base. Las seis puertas se cumplieron:
10,48–16,75 segundos y 43–44 °C al iniciar. No se relajó la política.

Cada modo se calienta durante al menos 0,1 s y tres llamadas. Se miden 30 llamadas
por modo y ámbito, usando las seis permutaciones de orden con igual frecuencia.
Son **66 grupos caso/proceso y 11.880 muestras**. El ámbito residente incluye
asignaciones, cálculo y sincronización; el completo añade subida y descarga.
Preparación, primera llamada, calentamiento y validación se registran por separado.
Son tiempos de pared del flujo, no tiempos de kernel obtenidos por timestamps GPU.

Antes y después de medir, cada salida de ambos ámbitos coincide bit a bit con
el control inicial. Los extremos son exactos frente a los mismos datos en float64;
softmax/logsumexp cumplen rtol=2e-5 y atol=2e-6 frente a la fórmula estable float64.
Los seis procesos terminaron y todas esas comprobaciones pasaron. El informe
distingue ejecución completa, aceptación numérica y equivalencia de bits.

## Resultados

Las tablas muestran el intervalo de las dos rondas para
`mediana(candidato) / media(mediana(control A), mediana(control B))`.
Menor que uno significa menos tiempo. Se conservan todos los casos.

### OpenCL custom

| Caso | Residente | Completo |
| --- | ---: | ---: |
| max filas 3×8193 | 0,109–0,148 | 0,211–0,323 |
| max filas 1024×1024 | 0,256–0,266 | 0,884–0,887 |
| min filas 8193×33 | 0,595–0,599 | 0,892–0,910 |
| max columnas 8193×33 | 0,074–0,110 | 0,233–0,248 |
| max total 1.048.577 | 0,585–0,636 | 0,886–0,917 |
| min ejes (0,2), 17×129×33 | 0,265–0,267 | 0,668–0,691 |
| max filas 3×17 | 0,561–0,604 | 0,810–0,814 |
| softmax 32×8193 | 0,217–0,317 | 0,492–0,498 |
| softmax 1024×1024 | 0,787–0,790 | 0,969–0,993 |
| softmax 8193×33 | 0,820–0,822 | 0,947–0,961 |
| logsumexp 32×8193 | 0,142–0,252 | 0,292–0,315 |

### CUDA custom

| Caso | Residente | Completo |
| --- | ---: | ---: |
| max filas 3×8193 | 0,759–0,830 | 0,883–0,917 |
| max filas 1024×1024 | 0,673–0,681 | 0,964–0,966 |
| min filas 8193×33 | 0,717–0,722 | 0,932–0,939 |
| max columnas 8193×33 | 0,849–0,876 | 0,944–0,951 |
| max total 1.048.577 | 0,851–0,855 | 0,972–0,978 |
| min ejes (0,2), 17×129×33 | 0,667–0,669 | 0,886–0,887 |
| max filas 3×17 | 0,621–0,683 | 0,845–0,846 |
| softmax 32×8193 | 0,937–0,943 | 0,990–1,002 |
| softmax 1024×1024 | 0,876–0,907 | 0,974–0,988 |
| softmax 8193×33 | 0,884–0,886 | 0,963–0,965 |
| logsumexp 32×8193 | 0,910–0,922 | 0,937–0,957 |

### Vulkan, candidato descartado como ruta predeterminada

| Caso | Residente | Completo |
| --- | ---: | ---: |
| max filas 3×8193 | 1,141–1,160 | 1,207–1,224 |
| max filas 1024×1024 | 0,880–0,885 | 1,003–1,004 |
| min filas 8193×33 | 0,916–0,931 | 1,019–1,058 |
| max columnas 8193×33 | 1,152–1,160 | 1,007–1,007 |
| max total 1.048.577 | 0,756–0,777 | 0,995–0,999 |
| min ejes (0,2), 17×129×33 | 0,883–0,893 | 0,911–0,932 |
| max filas 3×17 | 0,881–0,897 | 0,933–0,934 |
| softmax 32×8193 | 1,039–1,057 | 1,025–1,078 |
| softmax 1024×1024 | 0,950–0,957 | 0,989–1,021 |
| softmax 8193×33 | 0,964–0,966 | 0,965–1,040 |
| logsumexp 32×8193 | 1,088–1,103 | 1,069–1,072 |

La mejora residente de OpenCL y CUDA se repite en todos los casos. En el flujo
completo las transferencias atenúan muchas ganancias: softmax CUDA 32×8193 no
demuestra una mejora completa consistente. Vulkan presenta regresiones repetidas,
incluido logsumexp; no se generaliza la selección entre backends.

Los dos controles también muestran ruido: la mayor razón entre sus medianas fue
1,055 en OpenCL, 1,045 en Vulkan y 1,157 en CUDA. Esta última corresponde al
logsumexp completo de la primera ronda; el max pequeño residente CUDA alcanzó
1,143 en la segunda. Los JSON conservan muestras y dispersión. Diferencias de
unos pocos puntos porcentuales no prueban por sí solas una mejora fiable.

## Integración y límites

La selección pertenece al lowerer mediante una capacidad opcional, resuelta al
preparar el plan. El compilador no selecciona por nombre del backend. Un lowerer
externo sin esa capacidad conserva la ejecución ordinaria. El paso nuevo consume
la entrada materializada como hoja: los productores compartidos, devueltos o no
elementwise no se duplican ni se absorben indebidamente. `explain` muestra
`map_reduce` y su operación. La fusión privada existente tiene prioridad.

La medición usa Windows, RTX 5060 Ti/driver 617.14, Python 3.12.14 y NumPy 2.5.3.
No demuestra portabilidad del rendimiento a otros dispositivos ni un beneficio
para dtypes no medidos. Tampoco compara contra bibliotecas externas. La ruta
matemática mantiene NaN, infinitos, ceros con signo y dtype; no añade tolerancias.

Los informes locales y el supervisor están en
`.artifacts/iterations-2026-10-22/extrema-planning/` y su directorio padre.
`measured-runner.py` conserva el runner anterior a promover el cambio; sus hashes
coinciden con los informes. El runner actual fuerza los controles para que repetir
el experimento no compare accidentalmente dos planes optimizados.

```powershell
.venv/Scripts/python.exe -m benchmarks.extrema_planning --backend opencl --repeats 30 --seed 1101 --output .artifacts/benchmarks/extrema-opencl.json
```

La puerta de reposo corresponde al supervisor, no a ese comando individual.
Usar un archivo nuevo; el runner rechaza sobrescribir resultados.

## Verificación de la integración

Pasan **106 pruebas dirigidas**, con cuatro omisiones por enteros no soportados
en CUDA. Incluyen los contratos de extremos y normalización, la capacidad opcional
del lowerer, productores compartidos/devueltos/transpuestos, rebinding y liberación
de arrays, así como el protocolo y la inversión de controles del benchmark.
OpenCL, Vulkan y CUDA fueron obligatorios. No se ejecutó la suite completa.

Los XML locales son `extrema-planning-tests.xml` (102 pasan, cuatro omisiones)
y `extrema-materialized-native.xml` (cuatro pasan), bajo el directorio de la iteración.

Después de la promoción, una ejecución corta de integración repite los once casos
en cada backend con seis llamadas por modo/ámbito: **1.188 muestras**, todos los
controles de referencia y bits aprobados. Comprueba el runner v2 con controles
forzados y el runtime integrado; no sustituye las dos rondas de rendimiento de
las tablas. Se conserva en `extrema-planning-integration/`, con puertas de reposo
y procesos separados.

QA independiente: **APPROVE**. Reejecutó 45 pruebas CPU y contrastó todas las
razones, muestras, órdenes, puertas de reposo, hashes y resultados de integración.
La revisión no ejecutó GPU ni modificó la implementación.
