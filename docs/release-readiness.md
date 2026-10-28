# Camino a la primera versión estable

Revisión: 2026-09-24. La versión actual continúa siendo `0.1.0a1` y no se publica
desde esta tarea. «Estable» debe describir un contrato y una matriz comprobados;
no exige ganar todos los benchmarks ni alcanzar compatibilidad total con NumPy.

## Estimación de trabajo

Después de las tres iteraciones actuales, la estimación es **2–4 iteraciones de
endurecimiento para una candidata a release de alcance acotado**, si no aparecen
defectos estructurales. Es una estimación de trabajo, no una fecha comprometida.
Una beta con usuarios externos debe preceder la declaración de estabilidad;
su duración depende de obtener pruebas y reportes representativos.

Una primera `0.1.0` puede estabilizar la API documentada y certificar la
configuración local, manteniendo explícitamente experimental el resto del
hardware. Para anunciar estabilidad entre fabricantes todavía falta evidencia
AMD/Intel física; PoCL ayuda a comprobar portabilidad OpenCL, pero no sustituye
esas GPUs. La disponibilidad de esos equipos es una dependencia externa, no
algo que otra optimización NVIDIA pueda resolver.

## Criterios de salida

| Frente | Evidencia disponible | Pendiente antes de la candidata |
| --- | --- | --- |
| API | Subconjunto NumPy, backend explícito, compilación y límites documentados | Revisar nombres, errores, contratos de arrays/aliasing, RNG y política de compatibilidad/deprecación; congelar el alcance de la release |
| Correctitud | Suites CPU/GPU locales, inputs irregulares/extremos, vida útil y concurrencia | Repetir sobre el artefacto candidato; clasificar los fallos numéricos conocidos y corregir los que violen el contrato declarado |
| Precisión | Referencias float64/Decimal80 y alertas visibles | Definir garantías por operación y límites de acumulación, cancelación, subnormales y FMA; no convertir «resultado finito» en aprobación numérica |
| Instalación | Extras independientes, wheel CPU mínimo y builds locales | Smoke tests desde artefactos en entornos limpios para cada extra, dependencias ausentes y versión mínima de Python; comprobar mensajes de driver/compilador faltante |
| CI | Workflow CPU en Windows/Linux/macOS, infraestructura de benchmarks y OpenCL PoCL definido | Ejecutarlo realmente en remoto; verificar artefactos instalados en la matriz, no solo su build; no confundir análisis sintáctico con ejecución Python 3.10 |
| Rendimiento | Resultados con copias, sincronización, errores y controles | Fijar una suite corta de regresión y registrar baseline del candidato; no usar mejoras microbenchmark como criterio único de estabilidad |
| Trazabilidad | Fuentes y hashes de experimentos locales | Crear historial versionado y release reproducible: este checkout todavía no tiene commits y los archivos del proyecto están sin seguimiento |
| Distribución y soporte | MIT y documentación técnica | Confirmar nombre/cuenta de publicación, changelog, instrucciones de contribución, plantilla de incidencias y matriz de soporte; publicación como acción separada |

Los backends pueden tener alcances distintos si están declarados. Los tipos aún
no implementados no deben anunciarse como soportados. No son automáticamente
bloqueantes para una release acotada
si fallan explícitamente; una operación anunciada que devuelve resultados
incorrectos sí lo es. Las alertas de cancelación requieren juicio numérico y
documentación, no simplemente subir tolerancias hasta que la suite pase.

## Secuencia propuesta

1. **Congelar contrato y cerrar correctitud:** revisar API/capacidades, clasificar
   alertas y resolver inconsistencias. Mantener dispersas, autodiff, solvers como
   API y compatibilidad NumPy total fuera del alcance.
2. **Endurecer distribución y CI:** instalación por extras, artefactos limpios,
   matriz real CPU/PoCL, recursos y errores de inicialización. Versionar el código
   y fijar un candidato reproducible.
3. **Beta externa y candidata:** publicar matriz exacta comprobada, recoger casos
   reales y fallos, ejecutar la suite de regresión y cerrar problemas bloqueantes.
   Una ronda extra queda reservada para lo que revele esa validación.

El rendimiento seguirá mejorando después de una release estable. Las siguientes
extensiones no deben retrasarla por sí solas: más kernels especializados,
autotuning general, nuevos fabricantes pendientes de medición y paridad completa
entre backends. Sí debe retrasarse si falla la semántica que se anuncia o no se
puede reproducir la instalación y las pruebas del candidato.
