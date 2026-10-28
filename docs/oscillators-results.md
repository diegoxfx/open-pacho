# Coupled oscillators: iterative flow without per-step observations

This application evaluates a case distinct from solvers run to convergence:
fixed work, reuse of the same matrix, and error accumulation along a
trajectory. It is implemented in `examples/oscillators.py`, outside the wheel, through
public operations. It adds neither physics, autodiff, nor a GPU loop to the runtime.

## Model and algorithm

For unit masses, it solves `q'' = -K @ q`, with symmetric positive-definite K.
State includes position q, velocity v, and acceleration a. One velocity-Verlet step
computes:

```
vhalf = v + (dt/2)*a
q1 = q + dt*vhalf
a1 = -(K @ q1)
v1 = vhalf + (dt/2)*a1
```

`integrate` computes `a0 = -(K @ q0)` once and carries acceleration from one step
to the next. An N-step trajectory uses N+1 GEMVs. Every repetition starts from the
same original state. Functions neither modify inputs nor query host values.
`energy` computes `(v·v + q·Kq)/2` as a separate diagnostic.

Compilation prepares only `verlet_step`; the outer loop remains in
Python and reuses the plan. Unrolling the entire trajectory would be another strategy
and is not used in this comparison. Synchronization occurs when measuring the complete trajectory,
without introducing downloads or energy reductions between steps.

Stability requires `abs(dt)*sqrt(lambda_max(K)) < 2`. The example validates
metadata and leaves K's properties, finiteness, and stability to the caller. The
dense sizes in this experiment do not necessarily represent the appropriate way
to store a local or sparse physical system.

## References and checks

Backend comparisons must use exactly the inputs already rounded to the
requested dtype. The NumPy float64 recurrence separates execution differences
from time-discretization error: it has higher precision for float32 inputs and equal precision
for float64, rather than being an exact oracle.

For a small system, if `K = Q diag(omega²) Q.T`, `c = Q.T @ q0`, and
`d = Q.T @ v0`, the continuous solution is:

```
q(t) = Q @ (c*cos(omega*t) + d*sin(omega*t)/omega)
v(t) = Q @ (-omega*c*sin(omega*t) + d*cos(omega*t))
```

The discrete recurrence is compared with this trajectory separately from the error
between backends. Initial energy is evaluated at t=0 and velocity always at
integer time points. Verlet energy may oscillate; neither monotonic decrease
nor zero final drift is required. Maximum excursion is obtained from a diagnostic
repetition outside the timings.

CPU tests check a scalar step, second-order convergence at fixed final time,
bounded energy excursion, reversibility, product counts,
immutable inputs, and plan rebinding. Eight native tests check eager,
compiled, and prepared execution in OpenCL, Vulkan, and CUDA custom/cuBLAS, float32 and float64.
All pass; Vulkan float64 supports this flow, which needs no transcendental functions.

## Measurement protocol

N=31, 257, and 2049, float32/float64, dt=1/32, and 64 steps are used: final time 2.
Seed 47 generates `K = I + R.T @ R / N` in float64; it is symmetrized before
conversion to the dtype, and the initial state is scaled by `1/sqrt(N)`. Generation and
references always use one BLAS thread, outside integration timings.
Hashes of the effective inputs are retained to compare processes and providers.

The supplied matrix is checked for symmetry, numerical Cholesky is checked
in float64, and `dt*sqrt(norm_inf(K)) < 2` is required. Cholesky is a numerical
positivity check, not an exact mathematical certificate. N≤257 adds the
modal reference and a separate energy repetition with per-step observation;
N=2049 retains the discrete reference and final energy, without diagonalization.

Comparators are NumPy with one and eight threads, OpenCL custom/CLBlast, Vulkan, CUDA
custom/cuBLAS, and CuPy. OpenPacho backends run eager and a prepared
single-step plan. Both modes interleave across 12 balanced rounds; each
backend has an independent process and is repeated in reverse order.

Preparation, initial upload, sample acceleration, first call, and
warm-up are recorded separately. Each resident sample recomputes
initial acceleration and performs all 64 steps on resident inputs; the
complete sample includes uploading K/q/v and downloading q/v/a. Validation remains outside
the timer. Samples, errors, and failures are retained; a timing is not removed because
its result is incorrect. Initial/final energy and modal error are also
evaluated outside the timings.

Resident and complete are distinct samples. Variation may make a
complete median lower than its resident median; subtracting them does not represent
transfer cost and is not used to break down time.

The benchmark records input/output bytes and does not measure peak memory.
These sizes establish neither performance beyond cache capacity nor a bandwidth
limit. Local conditions remain RTX 5060 Ti, Windows, and
driver 617.14; results are not extrapolated to other GPUs.

```powershell
python -m benchmarks.oscillators --backend numpy --profile quick --repeats 12 --output .artifacts/oscillators-new.json
python -m benchmarks.oscillators --backend cuda-custom --profile all --repeats 12 --output .artifacts/oscillators-cuda-new.json
```

GPU runs require exclusive access and observed idle state before starting
each process. Local artifacts are retained in
`.artifacts/iterations-2026-10-22/oscillators/`, with the session and individual
reports; `oscillators-native.xml` retains the eight native tests.

## Local results

For this workload, NumPy with eight threads is faster than every measured GPU path,
both resident and complete, in both dtypes and all three sizes. Compiling one
step consistently improves OpenPacho versus eager, but that gain is
insufficient to beat the selected CPU reference. For N=2049 float64, compiled CUDA custom
takes 17.806–18.011 ms resident versus 58.222–63.942 ms eager;
NumPy with eight threads takes 11.752–12.572 ms. The complete flow retains that conclusion.

GPU wall time changes little between N=31 and N=2049, while NumPy cost grows
with size. This justifies studying loop dispatch/reuse before
adding another GEMV kernel; without GPU events specific to this trajectory, the
difference is not quantitatively attributed to Python or the driver. Matrices are not artificially
expanded to turn this measurement into a GPU win either.

The application provides a reproducible flow and a bounded backend recommendation,
not a runtime modification. In systems with a modal reference, Verlet's
time-discretization error exceeds observed differences between backends: changing
from float32 to float64 does not remove the error from using this time step.

The 14 runs complete 156 accepted rows and 3,744
resident/complete samples. Input, runtime, example, and runner hashes match
across processes. Tables show median ranges from both rounds,
in milliseconds per 64-step trajectory; they are not confidence intervals.

### float32

| Implementation | N31 resident | N31 complete | N257 resident | N257 complete | N2049 resident | N2049 complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numpy-1t | 0.717–0.731 | 0.713–0.773 | 1.061–1.062 | 1.062–1.064 | 14.667–14.729 | 14.400–15.191 |
| numpy-8t | 0.713–0.723 | 0.706–0.720 | 1.058–1.058 | 1.058–1.116 | 8.570–9.253 | 8.158–9.168 |
| opencl-eager | 44.913–53.373 | 40.999–48.919 | 42.758–46.266 | 47.239–50.018 | 43.234–47.489 | 43.716–44.575 |
| opencl-compiled | 18.943–26.317 | 19.161–27.352 | 22.377–23.182 | 22.193–24.180 | 19.000–19.837 | 21.921–22.388 |
| opencl-clblast-eager | 42.690–47.087 | 44.814–49.177 | 41.899–49.103 | 41.451–50.105 | 44.875–45.283 | 42.722–45.421 |
| opencl-clblast-compiled | 20.837–23.468 | 20.478–20.989 | 20.513–27.558 | 20.623–27.244 | 20.056–21.831 | 22.622–23.441 |
| vulkan-eager | 62.037–63.766 | 63.346–65.112 | 62.576–73.770 | 64.606–67.502 | 61.263–81.046 | 62.653–78.739 |
| vulkan-compiled | 18.112–19.516 | 18.879–18.955 | 19.106–19.848 | 19.758–20.760 | 17.699–20.978 | 19.224–21.744 |
| cuda-custom-eager | 60.117–62.197 | 59.347–59.387 | 62.184–66.478 | 59.725–68.805 | 58.959–61.645 | 63.191–68.546 |
| cuda-custom-compiled | 17.579–17.621 | 17.947–18.076 | 18.328–18.567 | 18.108–18.978 | 18.106–18.296 | 20.834–23.223 |
| cuda-cublas-eager | 58.558–58.663 | 62.553–63.052 | 64.029–70.443 | 59.620–69.676 | 60.521–63.052 | 61.504–64.686 |
| cuda-cublas-compiled | 18.422–18.929 | 18.660–19.021 | 19.505–21.653 | 19.401–19.606 | 18.971–19.211 | 21.909–22.245 |
| cupy-eager | 14.991–15.194 | 15.284–15.334 | 15.155–16.364 | 15.691–16.003 | 15.402–16.148 | 16.247–17.760 |

### float64

| Implementation | N31 resident | N31 complete | N257 resident | N257 complete | N2049 resident | N2049 complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numpy-1t | 0.681–0.695 | 0.674–0.688 | 1.328–1.332 | 1.326–1.355 | 27.644–29.461 | 27.412–29.292 |
| numpy-8t | 0.670–0.679 | 0.680–0.682 | 1.288–1.410 | 1.287–1.333 | 11.752–12.572 | 10.995–11.933 |
| opencl-eager | 43.799–46.097 | 45.547–46.023 | 47.316–48.534 | 45.522–47.421 | 43.465–47.175 | 45.213–51.313 |
| opencl-compiled | 20.840–21.582 | 20.494–20.710 | 23.135–26.860 | 19.909–21.993 | 18.885–20.421 | 24.123–27.328 |
| opencl-clblast-eager | 43.055–53.785 | 40.954–52.316 | 41.576–46.281 | 41.020–49.449 | 40.112–44.307 | 45.514–47.912 |
| opencl-clblast-compiled | 20.591–26.980 | 21.364–28.178 | 21.926–25.881 | 20.916–21.185 | 19.894–25.658 | 24.819–26.016 |
| vulkan-eager | 66.628–72.313 | 68.901–71.502 | 64.030–67.724 | 67.771–71.295 | 62.423–70.050 | 64.925–67.081 |
| vulkan-compiled | 19.665–19.811 | 20.448–21.081 | 19.002–19.214 | 18.905–19.239 | 18.873–20.224 | 23.070–24.708 |
| cuda-custom-eager | 59.523–61.556 | 58.802–62.116 | 58.927–62.371 | 64.388–65.933 | 58.222–63.942 | 63.327–67.844 |
| cuda-custom-compiled | 18.549–21.822 | 17.950–18.540 | 17.806–19.734 | 18.130–18.944 | 17.806–18.011 | 22.881–23.427 |
| cuda-cublas-eager | 58.766–59.594 | 59.261–61.730 | 61.607–65.043 | 61.732–74.093 | 59.879–64.922 | 65.140–71.768 |
| cuda-cublas-compiled | 18.889–18.968 | 18.912–19.528 | 18.617–18.890 | 18.906–19.355 | 19.498–20.004 | 24.661–25.143 |
| cupy-eager | 16.182–16.562 | 16.014–16.942 | 15.670–15.939 | 15.782–16.461 | 15.803–17.774 | 18.313–21.487 |

### Numerical differences

`rtol=2e-4, atol=2e-6` for float32 and `rtol=2e-10, atol=2e-12`
for float64 were required against the NumPy float64 recurrence on the same inputs.
Maximum absolute errors observed in saved diagnostics are:

| Dtype | Position q | Velocity v | Acceleration a |
| --- | ---: | ---: | ---: |
| float32 | 1.539e-07 | 1.370e-07 | 2.709e-07 |
| float64 | 1.665e-16 | 2.515e-16 | 4.718e-16 |

The modal reference measures another error: time discretization with Verlet.
Energy excursion is the maximum absolute value of `(E(t)-E(0))/E(0)`
at 65 time points, across providers and both rounds.

| N / dtype | Maximum time-discretization error q | Maximum time-discretization error v | Observed energy excursion |
| --- | ---: | ---: | ---: |
| 31 / float32 | 8.607e-05 | 1.372e-04 | 3.938e-04–3.939e-04 |
| 31 / float64 | 8.607e-05 | 1.372e-04 | 3.939e-04–3.939e-04 |
| 257 / float32 | 6.862e-05 | 8.378e-05 | 3.693e-04–3.693e-04 |
| 257 / float64 | 6.862e-05 | 8.378e-05 | 3.693e-04–3.693e-04 |

N=2049 has neither a modal reference nor per-step excursion in this measurement;
it does retain the discrete reference and final energy change.

5,916 validation calls and 12,480 individual saved diagnostics
are recorded. Warm-up validates all its
calls and retains only the final set of metrics to bound
the report; any failure is retained. No row failed numerically.

Targeted verification comprises 21 CPU example tests, eight native tests, and
11 benchmark-protocol tests: 40 passed. The global suite was not run.

Independent QA: **APPROVE**. The 32 CPU tests were rerun, and the
156 timing cells, numerical tables, hashes, physical references, and 12
idle gates were checked. All six inputs were recreated under single-threaded BLAS. The 33
runtime files are identical to the previous state; native review is based
on the saved XML of eight cases, without new GPU runs during QA.
