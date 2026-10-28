# Especialización por capacidades y track de hardware

Este documento fija reglas para especialización por dispositivo y sistema
operativo. La línea principal sigue siendo portable. Las variantes implementadas
se distinguen de los futuros perfiles por familia y autotuning.

## Variantes implementadas — 2026-09-13

- Vulkan detecta extensiones en instancia/dispositivo, habilita properties2 y
  push_descriptor cuando están disponibles y comprueba maxPushDescriptors.
  Dentro del límite usa push descriptors; fuera, descriptor sets. La selección
  no usa el nombre NVIDIA ni cambia shaders, barreras o precisión. La alternativa
  portátil se conserva y se prueba en el mismo equipo.
- CUDA comprueba el límite de hilos y memoria compartida antes de usar el GEMM
  float32 tiled compartido con OpenCL/Vulkan. NVRTC compila para la capacidad de
  cómputo del dispositivo. No activa TF32, fast-math ni contracción FMA nueva.
- OpenCL usa índices int32 para elementwise/gather cuando una cota conservadora
  demuestra que caben los accesos, con ruta int64. Es una especialización por
  forma/layout, no por fabricante ni por precisión de los datos.

Las macros pueden expresar una variante dentro de un kernel, pero no sustituyen
la consulta/habilitación de capacidades ni la selección de llamadas host. Las
cachés de programas son locales al backend/dispositivo; las cachés compartidas
guardan solo texto/metadatos. Futuras instrucciones de subgrupo, FMA o productos
matriciales especiales deberán declarar sus requisitos y semántica numérica antes
de entrar al selector. Que una instrucción sea más rápida no demuestra equivalencia
de redondeo, acumulación o tratamiento de subnormales.

Ver [implementación y mediciones](backend-sharing-results.md). Los perfiles por
familia/SO y autotuning siguen como track secundario.

## Orden de trabajo

El track principal conserva este orden:

1. Ejecución Vulkan: bajar el coste de ejecución y staging; reutilizar
   descriptores y agrupar envíos (batching).
2. GEMV y map-reduce, con mediciones residentes y aplicaciones externas
   (solvers de `examples/`, con criterios de convergencia explícitos).
3. Un ejecutor preparado que reutilice planes, memoria y metadatos sin ocultar
   validaciones ni sincronizaciones.
4. BLAS batched y kernels custom donde las mediciones justifiquen cruzar una
   frontera de proveedor.

Hardware y sistema operativo son un track de menor prioridad y no deben frenar
la [línea principal vigente](roadmap.md).

## Reglas de selección

- Comprobar capacidades reales primero: límites de workgroup, memoria local,
  tipos y extensiones. No inferirlas por el nombre de la familia del dispositivo.
- Si falta una capacidad y existe una ruta portable válida, usarla dentro del
  mismo backend y contrato numérico. Si no existe, dar un error explícito.
  Nunca cambiar de backend implícitamente.
- La elección de proveedor, como BLAS frente a kernel propio, debe ser explícita,
  observable y reversible.
- Separar tuning opcional (tiles, subgrupos o estrategias específicas del SO)
  de workarounds necesarios para corrección. Batching, pooling y fusión portables
  pertenecen a la línea principal. Acotar un workaround a un hardware requiere
  evidencia de su alcance; con causa desconocida se conserva la protección
  general. No cambiar dtype, tolerancias o frecuencia de comprobación.

## Límite de la evidencia de hardware

RTX50 es una familia candidata para repetir pruebas, no una conclusión de
portabilidad. La evidencia local disponible se limita a RTX 5060 Ti, Windows 11
y driver 616.64. PCIe Gen3 ×8 describe la configuración del host y del enlace
observado; no es una propiedad de la familia RTX50. No se atribuye un bug a un
hardware concreto sin una reproducción comparada que lo aísle.

## Casos numéricos que deben permanecer separados

- El subdesbordamiento de la norma con `A=eye(2)`,
  `b=[1e-30,-1e-30]` float32 y `rtol=atol=0` es un caso general de rango
  numérico. Se verificó en tres backends locales: la suma de cuadrados float32
  dio cero, mientras la comprobación float64 dio aproximadamente `1.414e-30`.
  No se presenta como un fallo específico de GPU.
- CLBlast 1.7.0 produjo NaN en GEMV con `beta=0` cuando el destino reciclado
  contenía NaN/Inf; se reprodujo en NVIDIA sobre Windows. El alcance causal
  sigue desconocido: no se extiende la observación a toda GPU, driver, BLAS o
  versión. El adaptador ya aplica un relleno preventivo general, con pruebas:
  no se restringirá a NVIDIA mientras falte evidencia para hacerlo con seguridad.

## Criterio para avanzar

La [ronda de kernels y estabilidad](kernel-stability-results.md) incorpora reposo
observado, calentamiento uniforme, orden invertido entre procesos y controles de
variantes. Promueve solo la eliminación de guardas OpenCL para tiles completos;
padding, layout local y bloques alternativos permanecen experimentales cuando
no muestran ganancias consistentes. Incluye el criterio de amortización para
packing/transpuestas globales y el coste de memoria previo a la selección.

Una especialización entra en el track principal solo con capacidad detectada,
fallback del mismo backend, proveedor explícito y comparación reproducible. Los
resultados deben conservar forma, dtype, tolerancias, residuo e iteraciones;
medir una mejora de un caso no autoriza una promesa para toda la familia.

Referencias: [gaps y optimización](gaps-and-optimization.md), [backend Vulkan](vulkan.md)
y [diagnóstico local](../benchmarks/gaps.py).
