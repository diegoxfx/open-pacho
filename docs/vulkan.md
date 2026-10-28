# Backend Vulkan

Backend compute nativo e independiente de OpenCL, implementado en 0.1.0a1.
Su objetivo inicial es ejecutar correctamente el mismo subconjunto de arrays;
todavía no tiene paridad de rendimiento con OpenCL o las referencias CUDA.

## Instalación y selección

Desde este repositorio:

```sh
python -m pip install ".[vulkan]"
pacho devices --backend vulkan
```

Se necesita un driver Vulkan del fabricante y un compilador GLSL a SPIR-V:
`glslangValidator`, `glslang` o `glslc`. Se busca primero el ejecutable indicado
en `PACHO_VULKAN_COMPILER`, después PATH y después `VULKAN_SDK/Bin`. Por ejemplo:

```powershell
$env:PACHO_VULKAN_COMPILER = "C:\VulkanSDK\<version>\Bin\glslangValidator.exe"
```

El paquete Python no descarga compiladores ni instala drivers automáticamente.
El [SDK Vulkan](https://vulkan.lunarg.com/) y
[glslang de Khronos](https://github.com/KhronosGroup/glslang) se distribuyen por separado.

```python
import openpacho as pacho

print(pacho.devices("vulkan"))
with pacho.use_backend("vulkan", device="0"):
    a = pacho.asarray([[1, 2], [3, 4]], dtype=pacho.float32)
    resultado = a @ a + 2
    print(pacho.asnumpy(resultado))
```

Los IDs Vulkan son índices propios, no los IDs `plataforma:dispositivo` de OpenCL.
Un backend ausente o una operación no soportada produce un error explícito.
No se cambia a CPU, CUDA u OpenCL para resolverla.

## Implementación y capacidades

Los arrays permanecen en buffers device-local. Las transferencias usan staging,
con barreras de visibilidad y sincronización. La cola conserva referencias a los
recursos en vuelo; descarga/synchronize y el límite de 64 operaciones pendientes
esperan su finalización. Una fence reutilizable marca el trabajo anterior de la
cola propia al observar resultados; solo después de esperar y resetearla se
reciclan los recursos. No añade una fence por dispatch ni elimina barreras.
Con la cola vacía, `synchronize()` retorna sin llamada nativa de espera.
Un fallo de finalización se propaga y bloquea nuevos envíos en esa instancia;
no se reciclan recursos cuya finalización sea incierta. Ver el
[control frente a espera de cola](routing-completion-results.md).
Los pipelines SPIR-V se cachean por fuente/configuración.
Los escalares aritméticos se pasan mediante push constants, sin subir un buffer
escalar por operación. Los bool se almacenan como uint32 en GPU y se convierten
a bool NumPy al transferirlos.

El staging host coherente de lectura conserva sus mapeos mientras existe la asignación,
también al reutilizarla desde el pool. Solo se desmapea al expulsarla o liberarla.
Esto elimina llamadas repetidas de map/unmap, sin eliminar copias, barreras ni
esperas. Las subidas mantienen map/unmap por copia; la variante general dio tiempos
mixtos. Los límites compartidos de memoria del pool siguen vigentes. La selección
de capacidad y la liberación de caché toleran que un finalizador Python devuelva
un buffer mientras se recorre el pool: el RLock por sí solo no evita esa reentrada.

| Área | Estado |
| --- | --- |
| float32, int32, uint32, bool | Operaciones soportadas de la API común |
| float64 | Requiere shaderFloat64; sin exp/log/sin/cos/tanh/potencia float64 |
| int64, uint64 | Requiere shaderInt64 |
| Broadcasting y elementwise | Kernels propios y escalares por valor |
| Aleatorios | [Philox residente](random.md): random/uniform float32/64; normal/exponential float32. Semilla/contador en dos palabras uint32, sin exigir shaderInt64 |
| sum/mean y normas | Reducciones propias; normas escaladas y parciales compactos para reducciones grandes |
| Productos vector/matriz | GEMV/DOT cooperativos; GEMM float32 elegible con bloques de registros 32×32 y alternativa 8×8; broadcasting por lotes; sin proveedor BLAS Vulkan |
| Reshape, slicing, transposición, diagonales | Contrato común; slicing/transposición copian |
| compile | Plan reutilizable, preparación de bindings, fusión elementwise/map-reduce y del patrón privado de residuo GEMV compatible |
| Memoria | Pool de buffers por tamaño/tipo de memoria; límites de storage buffer e índices comprobados |

`compile(..., padding=True)` o `matmul(..., padding=True)` permiten
[padding interno de GEMM float32](padding.md), con relleno y recorte en GPU.
Las copias mantienen los bits originales y las formas visibles no cambian.
Está desactivado por defecto; mide el pipeline completo antes de elegirlo.

La API pública conserva sus restricciones aunque una primitiva interna admita
más casos. No se promete compatibilidad NumPy completa ni igualdad de dtypes
disponibles entre dispositivos. Un shader rechazado no se traduce en un resultado
CPU escondido.

## Verificación

Pruebas específicas incluidas selección pública, buffers residentes,
dependencias entre 80 operaciones, tipos, slicing, reducción, matmul, compilación
cacheada y los tres solvers de `examples/solvers.py` en float32/64, como
aplicaciones externas al paquete. Ejecutadas en RTX 5060 Ti con driver
616.64, Windows 11 y glslang 16.5.0. **AMD e Intel todavía no se han probado.**

```powershell
$env:PACHO_REQUIRE_VULKAN = "1"
python -m pytest tests/test_vulkan.py -q
pacho benchmark --backend vulkan --size 128 --repeat 7
```

El pool conserva como máximo 128 buffers libres y 128 MiB de memoria Vulkan
realmente asignada, no solo bytes lógicos. Los recursos en vuelo no vuelven al
pool hasta terminar. Al recibir un buffer nuevo, expulsa buffers de las clases
de tamaño menos recientemente usadas si hace falta espacio. Así, tamaños viejos
no impiden indefinidamente reutilizar los nuevos. Prefiere capacidad exacta y,
si no existe, la menor compatible hasta `2*max(solicitado,256)`; mantiene tamaño
lógico para descriptores/copias y capacidad original para reciclar. No confunde
el tamaño del VkBuffer con los requisitos nativos de asignación. Alternar un
conjunto mayor que su capacidad todavía puede causar reservas y liberaciones
repetidas. Se reutilizan hasta 128 command buffers. La ruta de descriptor sets
reutiliza hasta 128 pools y sets después de terminar su uso. Sus layouts
pertenecen al backend y sobreviven a la expulsión de pipelines. Cada llamada
compilada graba sus dispatches en un command buffer y lo envía; sincronización,
presión de caché o 64 operaciones pendientes pueden dividir el grupo. Las
barreras de memoria se mantienen entre operaciones. Eager conserva el envío
inmediato. No hay subasignación de grandes bloques ni command graphs persistentes.

El readback prefiere memoria HOST_VISIBLE + HOST_COHERENT + HOST_CACHED cuando
los bits de compatibilidad del buffer la permiten. Si no existe, conserva memoria
coherente compatible; no usa memoria no coherente sin invalidación. El pool
separa staging de subida y readback para no perder esa preferencia al reciclar.
La subida copia directamente al mapeo, sin crear un objeto bytes intermedio.
Son mejoras de la ruta portable, no especializaciones NVIDIA.

Las plantillas de dispatch conservan fuente, ABI y geometría en una LRU de 128
entradas. En la ruta de fallback, los descriptor sets conservan sus estructuras
CFFI y se reenlazan después de completar su uso. Los registros host de push se
reutilizan tras grabar su snapshot en el command buffer; los buffers GPU siguen
retenidos hasta terminar. Las estructuras de envío/barreras se reutilizan bajo el lock.
Esto reduce construcción de objetos, sin eliminar la grabación de comandos.

La [medición de memoria y kernels](numerics-vulkan-results.md) cuantifica esa etapa;
el [análisis anterior](gaps-and-optimization.md) conserva la referencia y los
gaps de Python/CFFI, grabación y transferencias que siguen pendientes.

## Frontera de comandos tipada

La ruta actual resuelve comandos frecuentes por VkDevice y usa argumentos CFFI
tipados. Evita conversiones genéricas del binding, conserva sus comprobaciones
VkResult y mantiene contextos/colas/recursos separados. Si no puede resolver la
ruta tipada al inicializar, conserva los wrappers del mismo backend; los errores
de ejecución se propagan. No requiere un compilador C/C++ ni un extra nuevo.

La tabla privada también permite instrumentación por instancia y un control
`wrapped`/`direct` en benchmarks. No es configuración pública que deba conocer
el usuario. Las barreras y la presión de 64 operaciones no cambian. Ver
[medición y diagnóstico](vulkan-dispatch-results.md). La preparación y el
control de planes siguen en Python; no es todavía un grabador nativo de grupos.

La ampliación del 2026-09-13 detecta y habilita `VK_KHR_push_descriptor` y su
dependencia properties2. Dentro de maxPushDescriptors, una llamada escribe todos
los descriptores en el command buffer; el driver conserva su snapshot. Se omiten
la actualización y el binding de un descriptor set por dispatch. Las estructuras
host se reutilizan bajo el lock; los buffers GPU siguen retenidos hasta completar
el trabajo. Dispositivos sin esa capacidad y grupos que exceden el límite
conservan descriptor sets. No se necesitan macros ni ajustes del usuario.
Los controles privados `sets`/`push` y el límite quedan registrados en la
[comparación entre backends](backend-sharing-results.md).

Power float32 ahora maneja bases negativas con exponentes enteros y los casos
de ceros con signo, NaN/Inf y exponentes fraccionarios fuera del dominio real.
Se evita llamar GLSL pow con argumentos para los que su resultado es indefinido.
Power float64 sigue fuera del contrato de este backend.
