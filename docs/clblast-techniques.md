# Qué aprovechar de CLBlast

Investigación del 2026-09-07. La biblioteca nativa local es CLBlast 1.7.0;
las fuentes enlazadas explican su diseño y no certifican qué variante exacta
eligió ese binario para cada llamada. No se atribuye toda diferencia temporal
a la capa Python: el kernel propio también tenía mucho trabajo por optimizar.

## La diferencia principal

OpenPacho ya usaba tiles en memoria local/compartida, pero cada hilo acumulaba
una sola salida. CLBlast combina esa reutilización con bloques privados de
acumuladores: un hilo calcula varias salidas y reutiliza cada dato cargado en
varias multiplicaciones. Su kernel directo declara bloques privados de A, B y C,
carga desde memoria local a registros y aplica multiply-add sobre esos bloques.
Esto está visible en el [kernel directo](https://github.com/CNugteren/CLBlast/blob/master/src/kernels/level3/xgemm_direct_part3.opencl).

La técnica aplicada aquí es una implementación propia, compartida entre OpenCL
y Vulkan en `_gemm.py`: cada grupo de 64 hilos calcula 32×32 salidas; cada hilo
mantiene 4×4 acumuladores, y recorre K en bloques de 16. Se requieren 4 KiB de
memoria compartida. La ruta cubre float32, M/N≥32 y K≥16, mantiene broadcasting
y comprueba bordes. Otros tipos/tamaños conservan las rutas anteriores.
CLBlast sigue siendo un proveedor explícito y no se sustituye bajo esa selección.

## Qué reproducir y qué posponer

| Técnica | Decisión | Motivo |
| --- | --- | --- |
| Bloques privados y reutilización local→registros | Implementado en OpenCL/Vulkan | Mejora el cálculo sin cambiar API ni almacenamiento público. |
| Cargas globales contiguas y acumuladores con índices constantes | Implementado | Permite que el compilador optimice registros; no exige instrucciones de un fabricante. |
| Vectorización explícita y unroll afinado | Siguiente experimento acotado | La alineación, presión de registros y forma determinan si gana. |
| Packing, padding y transposición previos | Padding opt-in añadido el 2026-09-18; lo demás pospuesto | [Contrato de padding](padding.md): añade memoria, kernels y preparación; debe amortizarse con suficiente trabajo. |
| Base de parámetros por familia y autotuning exhaustivo | Track secundario | Necesita más hardware y persistencia/versionado de resultados. |
| FP16/TF32 o fast-math para mejorar una comparación | Fuera del contrato actual | Cambiaría precisión; cualquier futura modalidad debe ser explícita y medirse con error. |

CLBlast mantiene una ruta directa para formas generales y otra indirecta con
preparación del almacenamiento. En matrices grandes, mover O(n²) elementos puede
amortizarse frente a O(n³) operaciones; en pequeñas puede dominar. Sus parámetros
incluyen geometría, bloques de registros, anchuras vectoriales y uso de memoria
local. Ver [detalles de GEMM](https://github.com/CNugteren/CLBlast/blob/master/doc/details_gemm.md).

Los valores concretos no son universales. CLBlast dispone de resultados de tuning
por dispositivo/familia y de búsqueda en dos etapas para GEMM; también afina la
decisión entre ruta directa e indirecta. Reproducir su base completa no es necesario
para empezar a ganar rendimiento. Ver [tuning oficial](https://github.com/CNugteren/CLBlast/blob/master/doc/tuning.md)
y el [paper de CLBlast](https://arxiv.org/abs/1705.05249).

## Evidencia local de la técnica implementada

Control residente intercalado, once muestras por variante, float32; mismo proceso,
inputs y backend. La columna anterior desactiva solo la nueva ruta de registros.
Son medianas de tiempo completo sincronizado, no tiempos de kernel aislados.

| Backend / GEMM | Local tiling anterior | Local + registros | Error relativo L2 frente a float64, ambas variantes |
| --- | ---: | ---: | ---: |
| OpenCL 512² | 0.324 ms | 0.247 ms | 4.045e-7 |
| OpenCL 1024² | 1.681 ms | 0.934 ms | 5.741e-7 |
| OpenCL 2048² | 12.092 ms | 6.126 ms | 8.099e-7 |
| Vulkan 512² | 0.477 ms | 0.331 ms | 4.045e-7 |
| Vulkan 1024² | 1.679 ms | 0.694 ms | 5.741e-7 |
| Vulkan 2048² | 11.262 ms | 2.930 ms | 8.099e-7 |

En 2048² el error absoluto máximo fue 1.517e-4 en ambas rutas. Estas métricas
idénticas en los casos probados no prometen igualdad bit a bit para cualquier
matriz. No se activó fast-math ni una precisión reducida. Los casos adversos de
cancelación se publican aparte y pueden tener error relativo elevado.

El código y datos del control están en `.artifacts/numerics-vulkan/register_gemm.py`
y `.artifacts/benchmarks/numerics-vulkan/register-gemm.json`. Las comparaciones
completas contra CLBlast, CuPy y nvmath, incluida transferencia, están en el
[informe de esta iteración](numerics-vulkan-results.md).
