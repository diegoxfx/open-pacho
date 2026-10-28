# Ejecución y transferencias Vulkan: continuación de la prioridad principal

Snapshot histórico. La etapa posterior está en [map-reduce, preparación y BLAS batched](remaining-suite-results.md).

2026-09-06. Esta ronda avanza el **primer bloque** del
[orden de trabajo](roadmap.md): ejecución, staging y reutilización de recursos
Vulkan. Solvers/map-reduce, ejecutor preparado y BLAS batched conservan su orden.
La [especialización por hardware/SO](hardware-specialization.md) queda como diseño
de menor prioridad; no se añadió un selector de familia ni una variante NVIDIA.

## Cambios generales implementados

1. **Subida sin serialización intermedia.** Se copia el array directamente al
   mapeo de staging; se elimina `data.tobytes()`, su asignación y copia adicional.
   Se mantienen almacenamiento propio, dtypes, bool y entradas no contiguas.
2. **Readback con preferencia por caché CPU.** Entre los tipos de memoria
   compatibles con el buffer, se prefiere HOST_VISIBLE + HOST_COHERENT +
   HOST_CACHED. Si no hay uno, se conserva una opción coherente compatible.
   No se confunde «visible al host» con «rápido de leer desde CPU».
3. **Pool separado por uso.** Staging de subida y readback tienen bins separados;
   reciclar un buffer no elimina la preferencia de memoria. El límite compartido
   sigue siendo 128 MiB de asignación real y 128 buffers libres.
4. **Descriptor sets reutilizables.** Se reciclan solamente tras completar GPU.
   Se actualizan todos sus bindings antes de volver a usarlos. Los layouts viven
   hasta cerrar el backend y sobreviven a la expulsión de pipelines; los sets de
   comandos en vuelo permanecen distintos.
5. **Envío agrupado por llamada compilada.** Los command buffers de un plan se
   envían juntos. La ejecución eager conserva su envío inmediato. Sincronización,
   expulsión de pipelines y el límite de 64 comandos pueden dividir el grupo.
   El lock cubre la grabación del grupo; los recursos grabados aún no enviados
   también conservan referencias fuertes. Se vacía el grupo al salir, incluso
   ante una excepción. Esto no introduce lazy global ni cambia los chequeos de
   convergencia de los solvers.

Se siguen grabando command buffers y construyendo estructuras por operación.
No se implementaron command graphs persistentes, un ejecutor nativo o map-reduce
Vulkan. El agrupamiento reduce llamadas de envío, pero no elimina el resto del
overhead del backend.

La preferencia de memoria usa propiedades del estándar, no nombres de GPU.
HOST_COHERENT evita requerir flush/invalidate manual del host; se mantienen las
barreras y la espera antes de leer. [Propiedades de memoria de Vulkan](https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryPropertyFlagBits.html).
Los layouts y sets se mantienen con la vida útil exigida por
[el contrato de descriptores](https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html).

## Transferencias: antes y después

RTX 5060 Ti, Windows 11, driver 616.64. Once muestras calientes, mediana de pared
sincronizada. Se mide el API público, incluyendo asignación/pool, map/unmap,
copias CPU y transferencia; **no es una medición aislada de PCIe**. Cada resultado
se valida exactamente, fuera del reloj. Los buffers de 256 MiB exceden el límite
del pool y vuelven a asignarse. Los procesos antes/después son independientes y
no bloquean frecuencias; las muestras y cuartiles están en los JSON.

| Tamaño | Subida antes → después | Descarga antes → después |
| --- | ---: | ---: |
| 256 bytes | 0.199 → 0.200 ms | 0.231 → 0.258 ms |
| 4 MiB | 1.873 → 1.267 ms | 17.871 → 1.528 ms |
| 64 MiB | 26.746 → 15.874 ms | 276.144 → 26.747 ms |
| 256 MiB | 122.536 → 105.615 ms | 1154.668 → 110.950 ms |

La descarga grande mejora aproximadamente diez veces. El caso de unos pocos
bytes no mejora: allí domina la latencia. Esta mejora corresponde a la selección
de memoria y copias del runtime; no demuestra que el enlace PCIe se haya acelerado
ni que toda GPU o SO vaya a obtener el mismo factor.

## Control residente

El perfil diagnóstico de `benchmarks.gaps`, sobre el mismo equipo, compara la
rueda congelada de la etapa anterior con el runtime actual. Siete muestras,
datos residentes y sincronización antes/después; cProfile se ejecuta aparte.

| Cálculo float32 | Antes | Después |
| --- | ---: | ---: |
| GEMV2048 | 0.366 ms | 0.315 ms |
| DOT2048 | 0.324 ms | 0.292 ms |
| Energía compilada 2048×32 | 0.611 ms | 0.541 ms |
| CG diagnóstico, 2 iteraciones | 13.522 ms | 11.107 ms |

Son mejoras de pared más modestas que las descargas grandes. No se atribuye cada
microsegundo a un cambio individual: esta comparación agrupa staging,
descriptores y envíos. El CG diagnóstico no es el mismo input que los solvers
completos de `benchmarks.execution`.

## Verificación y reproducción

Se repitieron además **319 configuraciones** de operaciones/pasos, solvers y
controles de tamaño/lotes: siete rondas por modo, sin errores de validación.
Las tablas completas (`.artifacts/benchmarks/vulkan-execution/report/all-results.md`)
y el CSV (`.artifacts/benchmarks/vulkan-execution/report/all-results.csv`),
artefactos locales no distribuidos, conservan
todas las implementaciones, incluyendo NumPy, OpenCL, CuPy, cuBLAS y nvmath.
Comparación Vulkan contra el snapshot anterior de gaps, milisegundos float32:

| Caso | Residente antes → después | Carga+cálculo+descarga antes → después |
| --- | ---: | ---: |
| Matmul2048² | 11.905 → 12.069 | 92.240 → 24.017 |
| Jacobi completo2048 | 82.502 → 66.667 | 88.606 → 69.264 |
| Descenso más pronunciado2048 | 100.486 → 84.218 | 106.417 → 88.690 |
| CG completo2048 | 49.800 → 42.927 | 54.800 → 46.353 |
| 100 actualizaciones de 1 048 576 elementos | 72.459 → 79.128 | 94.944 → 78.696 |

La última fila sigue siendo mixta: cien llamadas compiladas independientes no
forman un solo envío ni un bucle GPU. El modo residente no mejoró en esa medición.
En la ronda actual Jacobi2048 da 10.120 ms en NumPy y 12.740 ms en OpenCL frente a
66.667 ms en Vulkan: la brecha de ejecución sigue siendo grande. No se cambia el
orden de prioridades por esta mejora de transferencias.

- Suite completa: **530 pasan, 1 omitida** por requerir dos dispositivos OpenCL.
- **118 pruebas Vulkan** incluidas en esa suite: transferencia, tipos, fusión,
  solvers, límites, recursos en vuelo, supervivencia de sets al expulsar pipelines,
  envío agrupado, sincronización y excepciones. La selección de memoria también
  tiene una prueba CPU de requisitos, preferencias y fallback.
- Se mantiene la falta de validación física AMD/Intel y de otros sistemas.
- Wheel instalado sin dependencias GPU: **194 pasan, 320 omitidas**. Ruff y
  formato pasan; wheel/sdist generados en `.artifacts/dist-vulkan-execution`.

Artefactos privados en `.artifacts/benchmarks/vulkan-execution`: los perfiles
`before-profile.json`/`after-profile.json` y transferencias
`before-transfers.json`/`after-transfers.json` conservan las muestras.
La fuente y su hash, IDs de dispositivo/vendor y versión raw del driver Vulkan
se incluyen en la medición de transferencias.

```powershell
$env:PACHO_VULKAN_COMPILER='<ruta a glslang o glslc>'
python -m benchmarks.vulkan_transfers --output .artifacts/benchmarks/vulkan-execution/after-transfers.json
python -m benchmarks.gaps --backends vulkan --output .artifacts/benchmarks/vulkan-execution/after-profile.json
```

Para repetir la batería amplia, configurar también CLBlast, PyOpenCL y las
referencias CUDA como en el informe de gaps:

```powershell
python -m benchmarks.execution --output .artifacts/benchmarks/vulkan-execution/operations.json
python -m benchmarks.execution --solvers-only --output .artifacts/benchmarks/vulkan-execution/solvers.json
python -m benchmarks.scale --output .artifacts/benchmarks/vulkan-execution/scale.json
python -m benchmarks.gaps_tables --root .artifacts/benchmarks/vulkan-execution --operations operations.json --solvers solvers.json --scale scale.json --output .artifacts/benchmarks/vulkan-execution/report
```

Para medir el estado anterior, usar la rueda congelada mediante `PYTHONPATH` en
otro proceso. No mezclar otras cargas GPU durante la medición. El benchmark se
llama Vulkan transfers porque estudia una ruta general; no habilita perfiles de
familia de hardware en el runtime.
