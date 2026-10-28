# Generación aleatoria residente

`openpacho.random.default_rng` crea un generador ligado a un backend. Las muestras
GPU se calculan directamente en su memoria; no se genera una matriz NumPy para
subirla. La selección GPU es explícita y sus extras son independientes.

```python
import openpacho as pacho

rng = pacho.random.default_rng(47, backend="opencl")
A = rng.uniform(-1, 1, size=(1025, 513), dtype=pacho.float32)
ruido = rng.normal(0, 0.05, size=A.shape, dtype=pacho.float32)
tiempos = rng.exponential(scale=2, size=(257, 129), dtype=pacho.float32)
resultado = A + ruido  # sigue en el dispositivo
```

`backend=` también acepta una instancia existente. El generador la conserva
aunque cambie después el backend predeterminado. El valor inicial predeterminado
de la librería sigue siendo NumPy.

## API y capacidades

| Método | Distribución y parámetros |
| --- | --- |
| `random(size=None, dtype=float64)` | Uniforme en [0, 1). |
| `uniform(low=0, high=1, size=None, *, dtype=float64)` | Uniforme en [low, high); límites iguales producen una constante. |
| `normal(loc=0, scale=1, size=None, *, dtype=float64)` | Normal con media loc y desviación scale ≥ 0. No está acotada a un intervalo. |
| `exponential(scale=1, size=None, *, dtype=float64)` | Exponencial de media scale ≥ 0; cero produce una constante cero. |

| Backend | random/uniform | normal/exponential |
| --- | --- | --- |
| NumPy, OpenCL, CUDA | float32 y float64 | float32 y float64 |
| Vulkan | float32 y float64 | float32 |

float64 requiere fp64. Vulkan no implementa los trascendentes GLSL necesarios en
doble precisión: pide float32 o selecciona otro backend explícitamente. No se
reduce la precisión ni se cambia a CPU. `random_distributions(dtype)` declara
soporte mediante un contrato interno; la fachada no decide por nombre de backend.

Los parámetros son escalares reales finitos interpretados en el dtype solicitado.
Se rechazan escalas negativas, límites invertidos y un ancho uniforme que desborde
ese dtype. La conversión puede redondear los límites; la pertenencia al intervalo
se refiere a esos límites representados. Se corrige el redondeo hacia el extremo
superior para conservar `high` excluido. Parámetros extremos y las colas normal/
exponencial todavía pueden producir valores fuera del rango del dtype; no se
recortan silenciosamente. Los subnormales dependen también del dispositivo.

`size` admite enteros y tuplas, dimensiones vacías, rectangulares e impares.
`None` produce un ndarray 0-D residente; `.item()` descarga el escalar. El límite
inicial es 2³¹−1 elementos por llamada, además de los límites de memoria del
backend. No hay `out=`, parámetros array, enteros aleatorios, elección/permutaciones,
distribuciones truncadas ni `spawn`.

La API se parece a NumPy, pero no promete igualdad de secuencias. `dtype=` en
uniform/normal/exponential es una extensión; NumPy sí ofrece dtype en `random`.

## Paralelismo y reproducibilidad

La fachada delega a `backend.random_data(...)` después de validar capacidad y
reservar contadores. Cada backend posee su adaptador de generación/lanzamiento;
el algoritmo compartido no pregunta por su nombre. Las cachés guardan fuente
inmutable, nunca buffers, semillas, posición del stream ni parámetros de distribución.
Separar adaptadores no cambia las secuencias ni añade dependencias GPU obligatorias.

Se usa **Philox4x32-10**, de la familia basada en contadores descrita por
[Random123](https://github.com/DEShawResearch/random123). Cada elemento evalúa
diez rondas enteras sobre su contador. No hay estado mutable por hilo, atómicos
ni una cadena secuencial de estados GPU. La multiplicación alta de uint32 usa
`mul_hi` OpenCL, `umulExtended` GLSL y `__umulhi` CUDA. El núcleo no requiere
enteros GPU de 64 bits.

1. `seed` es un entero en [0, 2⁶⁴); sus mitades forman la clave. `None` obtiene
   una nueva semilla de entropía del sistema sin modificar un RNG global.
2. El contador ocupa dos palabras uint32; las otras dos son cero. Cada salida
   consume un bloque de cuatro palabras, en orden C.
3. Las llamadas consumen tantos contadores como elementos; las vacías ninguno.
   Cambiar la forma o dividir llamadas con iguales distribución, dtype y parámetros
   conserva los valores. Hay un error antes de envolver 2⁶⁴.
4. `random` convierte 24 bits para float32 y 53 para float64 a [0, 1). El núcleo
   entero y esa conversión tienen controles exactos entre CPU y GPUs locales.
5. Se usa Box-Muller para normal e inversa de CDF para exponencial. Las funciones
   matemáticas pueden diferir entre APIs/compiladores: no se garantiza igualdad
   bit a bit entre dispositivos o drivers para las distribuciones transformadas.
   La resolución finita también limita las colas.

Un lock reserva rangos disjuntos cuando varios threads comparten un generador.
La planificación determina qué llamada obtiene cada rango; su orden no es
reproducible entre threads. Validaciones fallidas no consumen contadores. Tras
reservar, un fallo de asignación/compilación/dispatch sí consume el rango para
evitar reutilizarlo ante un resultado asíncrono incierto. No hay checkpoint/restore
público en esta versión.

Normal/exponencial usan `log1p` donde existe. Vulkan usa una serie de ocho términos
para `-log(1-u)` cuando u < 1/8, evitando pérdida de precisión cerca de cero; fuera
de ese intervalo conserva `log`. No es un modo de matemáticas rápidas ni una
optimización específica de NVIDIA.

No es un generador criptográfico. Los [vectores conocidos de Random123](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors)
y las pruebas estadísticas locales no equivalen a TestU01/PractRand ni certifican
independencia estadística entre todas las semillas.

## Compilación y medición

La generación con estado no se captura dentro de `compile` o `fuse`, tampoco si
el generador proviene de una closure. Falla antes de avanzar el contador. Genera
fuera y pasa los arrays como argumentos. Aleatoriedad dentro del grafo requerirá
un contrato explícito de estado/efectos; no se congela una muestra al compilar.

`benchmarks.random_generation` separa generación con resultado residente,
generación y descarga al host, y generación NumPy seguida de subida a la GPU.
Incluye asignación, conserva outliers y mide primera llamada/calienta por separado.
Los algoritmos de los comparadores pueden ser distintos: se comparan distribuciones
y destinos, no secuencias. Ver [resultados de esta ronda](staging-random-results.md).
