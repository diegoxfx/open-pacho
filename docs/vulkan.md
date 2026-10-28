# Vulkan backend

A native compute backend independent of OpenCL, implemented in 0.1.0a1.
Its initial goal is to execute the same array subset correctly;
it does not yet have performance parity with OpenCL or the CUDA references.

## Installation and selection

From this repository:

```sh
python -m pip install ".[vulkan]"
pacho devices --backend vulkan
```

A vendor Vulkan driver and a GLSL-to-SPIR-V compiler are required:
`glslangValidator`, `glslang` or `glslc`. Search first checks the executable specified
in `PACHO_VULKAN_COMPILER`, then PATH and then `VULKAN_SDK/Bin`. For example:

```powershell
$env:PACHO_VULKAN_COMPILER = "C:\VulkanSDK\<version>\Bin\glslangValidator.exe"
```

The Python package does not automatically download compilers or install drivers.
The [Vulkan SDK](https://vulkan.lunarg.com/) and
[Khronos glslang](https://github.com/KhronosGroup/glslang) are distributed separately.

```python
import openpacho as pacho

print(pacho.devices("vulkan"))
with pacho.use_backend("vulkan", device="0"):
    a = pacho.asarray([[1, 2], [3, 4]], dtype=pacho.float32)
    result = a @ a + 2
    print(pacho.asnumpy(result))
```

Vulkan IDs are its own indices, not OpenCL `platform:device` IDs.
An unavailable backend or unsupported operation produces an explicit error.
It does not switch to CPU, CUDA or OpenCL to resolve it.

## Implementation and capabilities

Arrays remain in device-local buffers. Transfers use staging,
with visibility barriers and synchronization. The queue retains references to
in-flight resources; download/synchronize and the limit of 64 pending operations
wait for completion. A reusable fence marks prior work on the
owned queue when observing results; resources are recycled only after waiting and
resetting it. It does not add a fence per dispatch or remove barriers.
With an empty queue, `synchronize()` returns without a native wait call.
A completion failure propagates and blocks new submissions on that instance;
resources with uncertain completion are not recycled. See the
[control against queue waiting](routing-completion-results.md).
SPIR-V pipelines are cached by source/configuration.
Arithmetic scalars are passed through push constants without uploading a scalar
buffer per operation. Bools are stored as uint32 on the GPU and converted
to NumPy bool during transfer.

Coherent host read staging retains its mappings for the lifetime of the allocation,
including reuse from the pool. It is unmapped only on eviction or release.
This removes repeated map/unmap calls without removing copies, barriers or
waits. Uploads retain map/unmap per copy; the general variant gave mixed
times. The pool's shared memory limits remain in force. Capability selection
and cache release tolerate a Python finalizer returning a buffer while the pool
is being traversed: the RLock alone does not prevent that reentrancy.

| Area | Status |
| --- | --- |
| float32, int32, uint32, bool | Supported operations of the common API |
| float64 | Requires shaderFloat64; no float64 exp/log/sin/cos/tanh/power |
| int64, uint64 | Requires shaderInt64 |
| Broadcasting and elementwise | Custom kernels and scalars by value |
| Random | [Resident Philox](random.md): float32/64 random/uniform; float32 normal/exponential. Seed/counter in two uint32 words, without requiring shaderInt64 |
| sum/mean and norms | Custom reductions; scaled norms and compact partials for large reductions |
| Vector/matrix products | Cooperative GEMV/DOT; eligible float32 GEMM with 32×32 register blocks and an 8×8 alternative; batch broadcasting; no Vulkan BLAS provider |
| Reshape, slicing, transpose, diagonals | Common contract; slicing/transpose copy |
| compile | Reusable plan, binding preparation, elementwise/map-reduce fusion and fusion of the compatible private GEMV residual pattern |
| Memory | Buffer pool by size/memory type; storage-buffer and index limits checked |

`compile(..., padding=True)` or `matmul(..., padding=True)` allow
[internal float32 GEMM padding](padding.md), with padding and cropping on the GPU.
Copies retain original bits and visible shapes do not change.
It is disabled by default; measure the complete pipeline before selecting it.

The public API retains its restrictions even if an internal primitive supports
more cases. Full NumPy compatibility or identical available dtypes across devices
is not promised. A rejected shader does not turn into a hidden CPU result.

## Verification

Specific tests include public selection, resident buffers,
dependencies across 80 operations, types, slicing, reduction, matmul, cached
compilation and the three solvers in `examples/solvers.py` in float32/64 as
applications external to the package. Run on RTX 5060 Ti with driver
616.64, Windows 11 and glslang 16.5.0. **AMD and Intel have not yet been tested.**

```powershell
$env:PACHO_REQUIRE_VULKAN = "1"
python -m pytest tests/test_vulkan.py -q
pacho benchmark --backend vulkan --size 128 --repeat 7
```

The pool retains at most 128 free buffers and 128 MiB of actually allocated
Vulkan memory, not just logical bytes. In-flight resources do not return to the
pool until completion. On receiving a new buffer, it evicts buffers from the least
recently used size classes if space is needed. Old sizes therefore do not prevent
new ones from being reused indefinitely. It prefers exact capacity and,
if none exists, the smallest compatible capacity up to `2*max(requested,256)`;
it keeps logical size for descriptors/copies and original capacity for recycling.
It does not confuse VkBuffer size with native allocation requirements. Alternating
a set larger than its capacity can still cause repeated allocations and releases.
Up to 128 command buffers are reused. The descriptor-set path
reuses up to 128 pools and sets after their use completes. Their layouts
belong to the backend and survive pipeline eviction. Each compiled call
records its dispatches in one command buffer and submits it; synchronization,
cache pressure or 64 pending operations can split the group. Memory barriers
remain between operations. Eager retains immediate submission.
There is no suballocation of large blocks or persistent command graphs.

Readback prefers HOST_VISIBLE + HOST_COHERENT + HOST_CACHED memory when
the buffer's compatibility bits allow it. If none exists, it retains compatible
coherent memory; it does not use noncoherent memory without invalidation. The pool
separates upload staging and readback to retain that preference when recycling.
Upload copies directly to the mapping without creating an intermediate bytes object.
These improve the portable path, rather than specializing for NVIDIA.

Dispatch templates retain source, ABI and geometry in a 128-entry LRU.
On the fallback path, descriptor sets retain their CFFI structures and are
rebound after use completes. Host push records are reused after recording their
snapshot in the command buffer; GPU buffers remain retained until completion.
Submission/barrier structures are reused under the lock.
This reduces object construction without eliminating command recording.

The [memory and kernel measurement](numerics-vulkan-results.md) quantifies that stage;
the [earlier analysis](gaps-and-optimization.md) preserves the reference and
remaining Python/CFFI, recording and transfer gaps.

## Typed command boundary

The current path resolves frequent commands per VkDevice and uses typed CFFI
arguments. It avoids generic binding conversions, preserves its VkResult checks
and keeps contexts/queues/resources separate. If it cannot resolve the typed
path at initialization, it retains wrappers from the same backend; execution
errors propagate. It requires no C/C++ compiler or new extra.

The private table also allows per-instance instrumentation and a
`wrapped`/`direct` control in benchmarks. This is not public configuration users
need to know. Barriers and the pressure of 64 operations do not change. See
[measurement and diagnostics](vulkan-dispatch-results.md). Plan preparation and
control remain in Python; this is not yet a native group recorder.

The 2026-09-13 extension detects and enables `VK_KHR_push_descriptor` and its
properties2 dependency. Within maxPushDescriptors, one call writes all
descriptors into the command buffer; the driver retains their snapshot. A descriptor
set update and binding per dispatch are skipped. Host structures
are reused under the lock; GPU buffers remain retained until work completes.
Devices without that capability and groups exceeding the limit
retain descriptor sets. No macros or user adjustments are needed.
Private `sets`/`push` controls and the limit are recorded in the
[cross-backend comparison](backend-sharing-results.md).

Float32 power now handles negative bases with integer exponents and the cases
of signed zeros, NaN/Inf and fractional exponents outside the real domain.
It avoids calling GLSL pow with arguments for which its result is undefined.
Float64 power remains outside this backend's contract.
