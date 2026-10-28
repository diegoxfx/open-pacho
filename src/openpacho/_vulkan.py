"""Optional native Vulkan compute backend, with resident device-local buffers.

Shaders are compiled by glslang or glslc. Set PACHO_VULKAN_COMPILER to its
executable, or install it on PATH. No compiler is downloaded at runtime.
Boolean storage uses uint32 internally; public transfers preserve NumPy bool.
"""

import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from math import prod
from pathlib import Path
from threading import RLock

import numpy as np

from .backends import DeviceInfo, check_dtype
from .exceptions import BackendUnavailableError, UnsupportedDtypeError

_TYPES = {
    "bool": "uint",
    "int32": "int",
    "uint32": "uint",
    "float32": "float",
    "int64": "int64_t",
    "uint64": "uint64_t",
    "float64": "double",
}
_EXPRESSIONS = {
    "add": "a + b",
    "subtract": "a - b",
    "multiply": "a * b",
    "divide": "a / b",
    "true_divide": "a / b",
    "power": "pacho_power(a, b)",
    "positive": "a",
    "negative": "-a",
    "absolute": "abs(a)",
    "sqrt": "sqrt(a)",
    "exp": "exp(a)",
    "log": "log(a)",
    "sin": "sin(a)",
    "cos": "cos(a)",
    "tanh": "tanh(a)",
    "equal": "a == b",
    "not_equal": "a != b",
    "less": "a < b",
    "less_equal": "a <= b",
    "greater": "a > b",
    "greater_equal": "a >= b",
    "isfinite": "!(isnan(a) || isinf(a))",
    "cast": "a",
    "where": "a != 0u ? b : c",
}
_COMPARISONS = {"equal", "not_equal", "less", "less_equal", "greater", "greater_equal", "isfinite"}

# GLSL pow is undefined for negative bases and zero bases with nonpositive
# exponents. Supply the real-valued float32 cases promised by the array API.
_POWER = """
float pacho_power(float a, float b) {
    float infinity = uintBitsToFloat(0x7f800000u);
    if (b == 0.0 || a == 1.0) return 1.0;
    if (isnan(a) || isnan(b)) return uintBitsToFloat(0x7fc00000u);
    float magnitude = abs(a);
    if (isinf(b)) {
        if (magnitude == 1.0) return 1.0;
        return ((magnitude > 1.0) == (b > 0.0)) ? infinity : 0.0;
    }
    bool integral = b == trunc(b);
    // All representable float32 integers of magnitude >= 2^24 are even.
    bool odd = integral && abs(b) < 16777216.0 && mod(abs(b), 2.0) == 1.0;
    uint sign = odd ? (floatBitsToUint(a) & 0x80000000u) : 0u;
    if (a == 0.0) return uintBitsToFloat(sign | (b > 0.0 ? 0u : 0x7f800000u));
    if (isinf(a)) return uintBitsToFloat(sign | (b > 0.0 ? 0x7f800000u : 0u));
    if (a < 0.0 && !integral) return uintBitsToFloat(0x7fc00000u);
    float result = pow(magnitude, b);
    return (a < 0.0 && odd) ? -result : result;
}
"""


def _load():
    try:
        import vulkan as vk
    except (ImportError, OSError) as exc:
        raise BackendUnavailableError(
            'Vulkan requires pip install "open-pacho[vulkan]" and a vendor Vulkan driver'
        ) from exc
    return vk


def _instance(vk, extensions=()):
    try:
        return vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(
                ppEnabledExtensionNames=list(extensions),
                pApplicationInfo=vk.VkApplicationInfo(
                    pApplicationName="OpenPacho", apiVersion=vk.VK_MAKE_VERSION(1, 0, 0)
                ),
            ),
            None,
        )
    except Exception as exc:
        raise BackendUnavailableError(f"Cannot create Vulkan instance: {exc}") from exc


def vulkan_devices():
    """Enumerate Vulkan devices without requiring a shader compiler."""
    vk = _load()
    instance = _instance(vk)
    try:
        result = []
        for index, physical in enumerate(vk.vkEnumeratePhysicalDevices(instance)):
            properties = vk.vkGetPhysicalDeviceProperties(physical)
            features = vk.vkGetPhysicalDeviceFeatures(physical)
            memory = vk.vkGetPhysicalDeviceMemoryProperties(physical)
            total = sum(
                memory.memoryHeaps[i].size
                for i in range(memory.memoryHeapCount)
                if memory.memoryHeaps[i].flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
            )
            result.append(
                DeviceInfo(
                    str(index),
                    properties.deviceName,
                    f"0x{properties.vendorID:04x}",
                    "GPU"
                    if properties.deviceType
                    in (
                        vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                        vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                    )
                    else "CPU/other",
                    bool(features.shaderFloat64),
                    int(total),
                    str(properties.driverVersion),
                )
            )
        return result
    finally:
        vk.vkDestroyInstance(instance, None)


def _compiler():
    configured = os.environ.get("PACHO_VULKAN_COMPILER")
    if configured:
        path = shutil.which(configured)
        if path:
            return path
        raise BackendUnavailableError(
            f"PACHO_VULKAN_COMPILER does not name an executable: {configured}"
        )
    for name in ("glslangValidator", "glslang", "glslc"):
        found = shutil.which(name)
        if found:
            return found
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for name in ("glslangValidator.exe", "glslang.exe", "glslc.exe"):
            path = Path(sdk) / "Bin" / name
            if path.is_file():
                return str(path)
    raise BackendUnavailableError(
        "Vulkan needs glslang or glslc on PATH, or PACHO_VULKAN_COMPILER set to its executable"
    )


def _strides(shape):
    return tuple(prod(shape[i + 1 :]) for i in range(len(shape)))


def _offset(shape, source_shape):
    padding = len(shape) - len(source_shape)
    terms = [
        f"((i / {max(1, prod(shape[axis + padding + 1 :]))}) % {dim}) * {stride}"
        for axis, (dim, stride) in enumerate(zip(source_shape, _strides(source_shape)))
        if dim != 1
    ]
    return " + ".join(terms) or "0"


def _type(dtype):
    return _TYPES[np.dtype(dtype).name]


def _storage(dtype):
    return np.dtype("uint32") if np.dtype(dtype).kind == "b" else np.dtype(dtype)


def _memory_type_index(properties, supported, required, preferred=0):
    candidates = [
        i
        for i, flags in enumerate(properties)
        if supported & (1 << i) and flags & required == required
    ]
    if not candidates:
        raise BackendUnavailableError("No compatible Vulkan memory type")
    return next((i for i in candidates if properties[i] & preferred == preferred), candidates[0])


class _Buffer:
    def __init__(self, backend, size, host=False, readback=False):
        self.backend = backend
        self.host = host
        self.readback = readback
        self.size = max(4, size)
        # Keep descriptor/copy extent separate from the reused VkBuffer's size.
        self.capacity = self.size
        vk, device = backend.vk, backend._device
        cached = backend._take_buffer(self.size, host, readback)
        if cached is not None:
            self.handle, self.memory, self.allocation_size, self.capacity = cached
            return
        self.handle = vk.vkCreateBuffer(
            device,
            vk.VkBufferCreateInfo(
                size=self.size,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(device, self.handle)
        flags = (
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            if host
            else vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )
        self.memory = None
        self.allocation_size = int(requirements.size)
        memory_type_bits = int(requirements.memoryTypeBits)
        try:
            index = _memory_type_index(
                backend._memory_type_flags,
                memory_type_bits,
                flags,
                vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT if readback else 0,
            )
            self.memory = vk.vkAllocateMemory(
                device,
                vk.VkMemoryAllocateInfo(allocationSize=self.allocation_size, memoryTypeIndex=index),
                None,
            )
            vk.vkBindBufferMemory(device, self.handle, self.memory, 0)
        except Exception:
            self.close()
            raise

    def close(self):
        vk, device = self.backend.vk, self.backend._device
        handle, memory = self.handle, self.memory
        self.handle = self.memory = None
        if handle is not None and memory is not None:
            self.backend._recycle_buffer(
                self.capacity, self.host, handle, memory, self.allocation_size, self.readback
            )
        else:
            if handle is not None:
                vk.vkDestroyBuffer(device, handle, None)
            if memory is not None:
                vk.vkFreeMemory(device, memory, None)

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - interpreter shutdown is best effort
            pass


class VulkanBackend:
    """Native Vulkan execution. First-version kernels prioritize correctness.

    Queue submissions are asynchronous. Host transfers and synchronize wait;
    every 64 pending commands are also drained to bound descriptor/resource use.
    """

    name = "vulkan"

    def __init__(
        self,
        device=None,
        *,
        _command_mode="auto",
        _descriptor_mode="auto",
        _staging_mode="readback",
        _completion_mode="fence",
    ):
        if _descriptor_mode not in ("auto", "sets", "push"):
            raise ValueError("descriptor mode must be auto, sets or push")
        if _staging_mode not in ("persistent", "transient", "readback"):
            raise ValueError("staging mode must be persistent, readback or transient")
        if _completion_mode not in ("queue", "fence"):
            raise ValueError("completion mode must be queue or fence")
        self.vk = vk = _load()
        self.compiler = _compiler()
        self._lock = RLock()
        self._pending = []
        self._completion_mode = _completion_mode
        self._completion_fence = None
        self._completion_failed = False
        self._recorded_commands = []
        self._batch_depth = 0
        self._recording_command = None
        self._recording_references = []
        self._recording_pools = []
        self._queued_operations = 0
        self._buffer_cache = OrderedDict()
        self._staging_mode = _staging_mode
        # Mappings follow native allocations, including completed pool entries.
        # They do not own _Buffer leases or permit reuse of in-flight storage.
        self._mapped_memory = {}
        self._cached_bytes = 0
        self._pool_limit = 128 * 1024**2
        self._pool_items = 0
        self._free_commands = []
        self._free_descriptor_pools = {}
        self._descriptor_layouts = {}
        self._push_records = {}
        self._push_descriptor_limit = 0
        self._closing = False
        self._fusion_bodies = OrderedDict()
        self._pipelines = OrderedDict()
        self._dispatch_templates = OrderedDict()
        properties_extension = "VK_KHR_get_physical_device_properties2"
        instance_extensions = {
            e.extensionName for e in vk.vkEnumerateInstanceExtensionProperties(None)
        }
        query_push = _descriptor_mode != "sets" and properties_extension in instance_extensions
        self._instance = _instance(vk, [properties_extension] if query_push else [])
        self._device = None
        self._command_pool = None
        try:
            physicals = list(vk.vkEnumeratePhysicalDevices(self._instance))
            if device is None:
                choices = [
                    p
                    for p in physicals
                    if vk.vkGetPhysicalDeviceProperties(p).deviceType
                    in (
                        vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                        vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                    )
                ]
            else:
                choices = [p for i, p in enumerate(physicals) if str(i) == str(device)]
            if not choices:
                raise BackendUnavailableError(f"No matching Vulkan GPU/device {device!r}")
            physical = choices[0]
            self._properties = vk.vkGetPhysicalDeviceProperties(physical)
            self.device = self._properties.deviceName
            # The binding wraps native structs dynamically. Cache immutable
            # capabilities once instead of traversing CFFI properties per call.
            limits = self._properties.limits
            self._max_storage_range = int(limits.maxStorageBufferRange)
            self._register_gemm = int(limits.maxComputeSharedMemorySize) >= 4096
            self._max_storage_bindings = int(limits.maxPerStageDescriptorStorageBuffers)
            self._max_push_constants_size = int(limits.maxPushConstantsSize)
            self._max_workgroup_count = tuple(int(v) for v in limits.maxComputeWorkGroupCount)
            features = vk.vkGetPhysicalDeviceFeatures(physical)
            self.fp64, self.int64 = bool(features.shaderFloat64), bool(features.shaderInt64)
            device_extensions = {
                e.extensionName for e in vk.vkEnumerateDeviceExtensionProperties(physical, None)
            }
            if query_push and "VK_KHR_push_descriptor" in device_extensions:
                push_properties = vk.VkPhysicalDevicePushDescriptorPropertiesKHR()
                properties2 = vk.VkPhysicalDeviceProperties2(pNext=push_properties)
                query = vk.vkGetInstanceProcAddr(
                    self._instance, "vkGetPhysicalDeviceProperties2KHR"
                )
                query(physical, properties2)
                self._push_descriptor_limit = int(push_properties.maxPushDescriptors)
            if _descriptor_mode == "push" and not self._push_descriptor_limit:
                raise BackendUnavailableError("Vulkan push descriptors are unavailable")
            queues = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical)
            family = next(i for i, q in enumerate(queues) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)
            self._timestamp_valid_bits = int(queues[family].timestampValidBits)
            self._timestamp_period = float(limits.timestampPeriod)
            self._memory = vk.vkGetPhysicalDeviceMemoryProperties(physical)
            self._memory_type_flags = tuple(
                int(self._memory.memoryTypes[i].propertyFlags)
                for i in range(int(self._memory.memoryTypeCount))
            )
            self._device = vk.vkCreateDevice(
                physical,
                vk.VkDeviceCreateInfo(
                    ppEnabledExtensionNames=(
                        ["VK_KHR_push_descriptor"] if self._push_descriptor_limit else []
                    ),
                    pQueueCreateInfos=[
                        vk.VkDeviceQueueCreateInfo(
                            queueFamilyIndex=family, queueCount=1, pQueuePriorities=[1.0]
                        )
                    ],
                    pEnabledFeatures=vk.VkPhysicalDeviceFeatures(
                        shaderFloat64=self.fp64, shaderInt64=self.int64
                    ),
                ),
                None,
            )
            self._queue = vk.vkGetDeviceQueue(self._device, family, 0)
            self._command_pool = vk.vkCreateCommandPool(
                self._device,
                vk.VkCommandPoolCreateInfo(
                    queueFamilyIndex=family,
                    flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                ),
                None,
            )
            # Immutable native records are reused while recording under _lock.
            self._begin_info = vk.ffi.new("VkCommandBufferBeginInfo[]", 1)
            self._begin_info[0].sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            self._begin_info[0].flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            # vkQueueSubmit consumes these host records during the call. Queue
            # ownership is instead represented by _pending's strong references.
            self._submit_buffers = vk.ffi.new("VkCommandBuffer[]", 64)
            self._submit_records = vk.ffi.new("VkSubmitInfo[]", 1)
            self._submit_records[0].sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO
            self._submit_records[0].pCommandBuffers = self._submit_buffers
            self._copy_region = vk.ffi.new("VkBufferCopy[]", 1)
            self._memory_barrier = vk.ffi.new(
                "VkMemoryBarrier[]",
                [
                    vk.VkMemoryBarrier(
                        srcAccessMask=vk.VK_ACCESS_MEMORY_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT | vk.VK_ACCESS_MEMORY_WRITE_BIT,
                    )
                ],
            )
            self._host_barrier = vk.ffi.new(
                "VkMemoryBarrier[]",
                [
                    vk.VkMemoryBarrier(
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
                    )
                ],
            )
            from ._vulkan_commands import DeviceCommands

            self.vk = DeviceCommands(vk, self._device, _command_mode)
            if self._push_descriptor_limit:
                pointer = vk.lib.vkGetDeviceProcAddr(self._device, b"vkCmdPushDescriptorSetKHR")
                if pointer == vk.ffi.NULL:
                    raise BackendUnavailableError("Vulkan push-descriptor entry point missing")
                self.vk.vkCmdPushDescriptorSetKHR = vk.ffi.cast(
                    "PFN_vkCmdPushDescriptorSetKHR", pointer
                )
        except Exception as exc:
            self.close()
            if isinstance(exc, BackendUnavailableError):
                raise
            raise BackendUnavailableError(f"Vulkan initialization failed: {exc}") from exc

    def _take_buffer(self, size, host, readback=False):
        with self._lock:
            key = (size, host, readback)
            entries = self._buffer_cache.get(key)
            while not entries:
                # Nearby shapes can share a completed VkBuffer. Native memory
                # requirements may exceed its actual capacity, so only the
                # creation size in the cache key can establish a safe fit.
                upper = 2 * max(size, 256)
                key = min(
                    (
                        candidate
                        # A finalizer can recycle/evict a buffer on this same
                        # thread while Python allocates during the scan. RLock
                        # permits that reentry, so scan keys detached from the
                        # cache and revalidate the selected class afterwards.
                        for candidate in tuple(self._buffer_cache)
                        if size <= candidate[0] <= upper and candidate[1:] == (host, readback)
                    ),
                    key=lambda candidate: candidate[0],
                    default=None,
                )
                if key is None:
                    return None
                entries = self._buffer_cache.get(key)
            handles = entries.pop()
            if not entries:
                del self._buffer_cache[key]
            else:
                self._buffer_cache.move_to_end(key)
            self._cached_bytes -= handles[2]
            self._pool_items -= 1
            return (*handles, key[0])

    def _recycle_buffer(self, capacity, host, handle, memory, allocation_size, readback=False):
        # In-flight submissions own strong references to every lease. This is
        # called only once the final array and submission references disappear.
        with self._lock:
            if not self._closing and allocation_size <= self._pool_limit:
                # Completed buffers of old shapes must not monopolize the pool.
                # Evict least-recently-used size classes until the incoming lease
                # fits. Live arrays and in-flight leases are never in this cache.
                while self._buffer_cache and (
                    self._pool_items >= 128
                    or self._cached_bytes + allocation_size > self._pool_limit
                ):
                    old_key = next(iter(self._buffer_cache))
                    old_entries = self._buffer_cache[old_key]
                    old_handle, old_memory, old_size = old_entries.pop()
                    if not old_entries:
                        del self._buffer_cache[old_key]
                    self._cached_bytes -= old_size
                    self._pool_items -= 1
                    self._free_buffer(old_handle, old_memory)
                key = (capacity, host, readback)
                self._buffer_cache.setdefault(key, []).append((handle, memory, allocation_size))
                self._buffer_cache.move_to_end(key)
                self._cached_bytes += allocation_size
                self._pool_items += 1
            elif self._device is not None:
                self._free_buffer(handle, memory)

    def _free_buffer(self, handle, memory):
        mapped = self._mapped_memory.pop(memory, None)
        if mapped is not None:
            self.vk.vkUnmapMemory(self._device, memory)
        self.vk.vkDestroyBuffer(self._device, handle, None)
        self.vk.vkFreeMemory(self._device, memory, None)

    @contextmanager
    def _map_staging(self, staging):
        # Callers hold _lock and own the lease. Upload writes precede submit;
        # readback waits for the transfer/host barrier and queue completion.
        assert staging.host
        if self._staging_mode == "transient" or (
            self._staging_mode == "readback" and not staging.readback
        ):
            mapped = self.vk.vkMapMemory(self._device, staging.memory, 0, staging.capacity, 0)
            try:
                yield mapped
            finally:
                self.vk.vkUnmapMemory(self._device, staging.memory)
        else:
            mapped = self._mapped_memory.get(staging.memory)
            if mapped is None:
                mapped = self.vk.vkMapMemory(self._device, staging.memory, 0, staging.capacity, 0)
                self._mapped_memory[staging.memory] = mapped
            yield mapped

    def validate_dtype(self, dtype):
        dtype = check_dtype(dtype)
        if (
            dtype == np.dtype("float64")
            and not self.fp64
            or dtype.kind in "iu"
            and dtype.itemsize == 8
            and not self.int64
        ):
            raise UnsupportedDtypeError(f"{self.device} does not support {dtype} in Vulkan shaders")
        return dtype

    def empty(self, shape, dtype):
        dtype = self.validate_dtype(dtype)
        size = prod(shape) * _storage(dtype).itemsize
        if prod(shape) > 2**31 - 1 or size > self._max_storage_range:
            raise ValueError("Array exceeds the Vulkan backend's indexing/storage-buffer limits")
        with self._lock:
            return _Buffer(self, size)

    def _begin(self):
        if self._recording_command is not None:
            return self._recording_command
        vk = self.vk
        if self._free_commands:
            command = self._free_commands.pop()
            vk.vkResetCommandBuffer(command, 0)
        else:
            command = vk.vkAllocateCommandBuffers(
                self._device,
                vk.VkCommandBufferAllocateInfo(
                    commandPool=self._command_pool,
                    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1,
                ),
            )[0]
        vk.vkBeginCommandBuffer(command, self._begin_info)
        self._recording_command = command
        return command

    def _barrier(self, command):
        vk = self.vk
        vk.vkCmdPipelineBarrier(
            command,
            vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            1,
            self._memory_barrier,
            0,
            vk.ffi.NULL,
            0,
            vk.ffi.NULL,
        )

    def _submit(self, command, references, descriptor_pool=None):
        assert command == self._recording_command
        self._recording_references.extend(references)
        if descriptor_pool is not None:
            self._recording_pools.append(descriptor_pool)
        self._queued_operations += 1
        if not self._batch_depth:
            self._flush_commands()
        if self._queued_operations >= 64:
            self.synchronize()

    def _flush_commands(self):
        if self._completion_failed:
            raise RuntimeError("Vulkan completion failed; this backend cannot submit more work")
        if self._recording_command is not None:
            self.vk.vkEndCommandBuffer(self._recording_command)
            self._recorded_commands.append(
                (self._recording_command, self._recording_pools, self._recording_references)
            )
            self._recording_command = None
            self._recording_pools = []
            self._recording_references = []
        if not self._recorded_commands:
            return
        for index, record in enumerate(self._recorded_commands):
            self._submit_buffers[index] = record[0]
        self._submit_records[0].commandBufferCount = len(self._recorded_commands)
        self.vk.vkQueueSubmit(self._queue, 1, self._submit_records, self.vk.ffi.NULL)
        self._pending.extend(self._recorded_commands)
        self._recorded_commands = []

    @contextmanager
    def _batch_commands(self):
        # Bound to a compiled invocation, not global lazy execution. Hold the
        # queue lock so another thread cannot interleave or observe recorded work.
        with self._lock:
            self._batch_depth += 1
            try:
                yield
            finally:
                self._batch_depth -= 1
                if not self._batch_depth:
                    self._flush_commands()

    def execution_scope(self):
        """Plan boundary: group commands, without changing eager semantics."""
        return self._batch_commands()

    def create_lowering(self):
        from ._lowering import VulkanLowering

        return VulkanLowering(self)

    def prepare_padded_matmul(self, ashape, bshape, shape, dtype):
        if not self._register_gemm:
            return None
        from ._padding import prepare_register_padding
        from ._padding_copy import vulkan_copy

        return prepare_register_padding(
            self,
            ashape,
            bshape,
            shape,
            dtype,
            vulkan_copy,
            max_buffer_bytes=self._max_storage_range,
        )

    def _copy(self, source, destination, size):
        if not size:
            return
        with self._lock:
            command = self._begin()
            self._barrier(command)
            self._copy_region[0].size = size
            self.vk.vkCmdCopyBuffer(
                command, source.handle, destination.handle, 1, self._copy_region
            )
            if destination.host:
                vk = self.vk
                vk.vkCmdPipelineBarrier(
                    command,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_PIPELINE_STAGE_HOST_BIT,
                    0,
                    1,
                    self._host_barrier,
                    0,
                    vk.ffi.NULL,
                    0,
                    vk.ffi.NULL,
                )
            self._submit(command, [source, destination])

    def upload(self, host):
        host = np.asarray(host)
        self.validate_dtype(host.dtype)
        data = np.ascontiguousarray(host, dtype=_storage(host.dtype))
        out = self.empty(host.shape, host.dtype)
        if data.size:
            with self._lock:
                staging = _Buffer(self, data.nbytes, host=True)
                if self._staging_mode == "persistent":
                    with self._map_staging(staging) as mapped:
                        np.copyto(np.frombuffer(mapped, data.dtype, data.size), data.reshape(-1))
                else:
                    # Keep the original upload path: persisting write mappings
                    # did not consistently improve end-to-end workloads.
                    mapped = self.vk.vkMapMemory(self._device, staging.memory, 0, data.nbytes, 0)
                    try:
                        np.copyto(np.frombuffer(mapped, data.dtype, data.size), data.reshape(-1))
                    finally:
                        self.vk.vkUnmapMemory(self._device, staging.memory)
                self._copy(staging, out, data.nbytes)
        return out

    def download(self, a):
        if not a.size:
            return np.empty(a.shape, a.dtype)
        with self._lock:
            size = a.size * _storage(a.dtype).itemsize
            staging = _Buffer(self, size, host=True, readback=True)
            self._copy(a._data, staging, size)
            self.synchronize()
            with self._map_staging(staging) as mapped:
                return (
                    np.frombuffer(mapped, _storage(a.dtype), a.size)
                    .astype(a.dtype, copy=True)
                    .reshape(a.shape)
                )

    def random_distributions(self, dtype):
        return (
            ("random", "uniform")
            if dtype == np.dtype("float64")
            else ("random", "uniform", "normal", "exponential")
        )

    def random_data(self, *args):
        from ._random_vulkan import random_data

        return random_data(self, *args)

    def full(self, shape, value, dtype):
        from ._fusion import HostScalar

        return self.elementwise("cast", [HostScalar(value, dtype)], shape, dtype)

    def _pipeline(self, source, bindings, push_size):
        if source in self._pipelines:
            self._pipelines.move_to_end(source)
            return self._pipelines[source]
        vk, device = self.vk, self._device
        with tempfile.TemporaryDirectory(prefix="openpacho-vulkan-") as folder:
            shader = Path(folder) / "compute.comp"
            binary = Path(folder) / "compute.spv"
            shader.write_text(source, encoding="utf-8")
            args = (
                [self.compiler, "-fshader-stage=compute", str(shader), "-o", str(binary)]
                if Path(self.compiler).stem.lower() == "glslc"
                else [
                    self.compiler,
                    "-V",
                    "--target-env",
                    "vulkan1.0",
                    str(shader),
                    "-o",
                    str(binary),
                ]
            )
            result = subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)
            if result.returncode:
                raise RuntimeError(
                    f"Vulkan shader compilation failed:\n{result.stdout}\n{result.stderr}"
                )
            code = binary.read_bytes()
        module = vk.vkCreateShaderModule(
            device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None
        )
        descriptor_layout = layout = pipeline = None
        try:
            push = 0 < bindings <= self._push_descriptor_limit
            descriptor_key = bindings, push
            descriptor_layout = self._descriptor_layouts.get(descriptor_key)
            if descriptor_layout is None:
                descriptor_layout = vk.vkCreateDescriptorSetLayout(
                    device,
                    vk.VkDescriptorSetLayoutCreateInfo(
                        flags=vk.VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR
                        if push
                        else 0,
                        pBindings=[
                            vk.VkDescriptorSetLayoutBinding(
                                binding=i,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                descriptorCount=1,
                                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                            )
                            for i in range(bindings)
                        ],
                    ),
                    None,
                )
                # Layouts outlive cached sets, including pipeline LRU eviction.
                self._descriptor_layouts[descriptor_key] = descriptor_layout
            push = (
                [
                    vk.VkPushConstantRange(
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=push_size
                    )
                ]
                if push_size
                else []
            )
            layout = vk.vkCreatePipelineLayout(
                device,
                vk.VkPipelineLayoutCreateInfo(
                    pSetLayouts=[descriptor_layout], pPushConstantRanges=push
                ),
                None,
            )
            # The binding copies embedded structs by value: retain stage so its
            # pName string remains alive through the native pipeline call.
            stage = vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName="main"
            )
            pipeline = vk.vkCreateComputePipelines(
                device,
                vk.VK_NULL_HANDLE,
                1,
                [vk.VkComputePipelineCreateInfo(layout=layout, stage=stage)],
                None,
            )[0]
        except Exception:
            if layout is not None:
                vk.vkDestroyPipelineLayout(device, layout, None)
            raise
        finally:
            vk.vkDestroyShaderModule(device, module, None)
        record = pipeline, layout, descriptor_layout
        self._pipelines[source] = record
        if len(self._pipelines) > 128:
            self.synchronize()
            _, previous = self._pipelines.popitem(last=False)
            self._destroy_pipeline(previous)
        return record

    def _dispatch_template(self, body, count, dtypes, scalar_dtypes, cooperative, shared):
        # Cache metadata only: scalar values and buffer leases are always rebound.
        key = body, count, dtypes, scalar_dtypes, cooperative, shared
        cached = self._dispatch_templates.get(key)
        if cached is not None:
            self._dispatch_templates.move_to_end(key)
            return cached
        declarations = [
            f"layout(set=0, binding={i}, std430) buffer B{i} {{ {_type(dt)} x{i}[]; }};"
            for i, dt in enumerate(dtypes)
        ]
        push_size = 0
        scalar_layout = []
        fields = []
        for index, scalar_dtype in enumerate(scalar_dtypes):
            dtype = _storage(scalar_dtype)
            push_size += (-push_size) % dtype.itemsize
            scalar_layout.append((push_size, dtype))
            fields.append(f"layout(offset={push_size}) {_type(dtype)} s{index};")
            push_size += dtype.itemsize
        if push_size > self._max_push_constants_size:
            raise NotImplementedError("Vulkan push-constant limit exceeded")
        if fields:
            declarations.append(
                "layout(push_constant) uniform Scalars { " + " ".join(fields) + " } pc;"
            )
        groups = count if cooperative else (count + 63) // 64
        width = min(groups, self._max_workgroup_count[0])
        height = (groups + width - 1) // width
        if height > self._max_workgroup_count[1]:
            raise ValueError("Vulkan dispatch exceeds device workgroup limits")
        extensions = "#extension GL_ARB_gpu_shader_fp64 : enable\n" if self.fp64 else ""
        if self.int64:
            extensions += "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable\n"
        index_expression = (
            f"gl_WorkGroupID.x + gl_WorkGroupID.y * {width}u"
            if cooperative
            else f"gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * {width * 64}u"
        )
        source = (
            "#version 450\n"
            + extensions
            + "\n".join(declarations)
            + (_POWER if "pacho_power(" in body else "")
            + f"""
layout(local_size_x=64) in;
{shared}
void main() {{
    uint index = {index_expression};
    if (index >= {count}u) return;
    int i = int(index);
    {body}
}}
"""
        )
        record = source, tuple(scalar_layout), push_size, width, height
        self._dispatch_templates[key] = record
        if len(self._dispatch_templates) > 128:
            self._dispatch_templates.popitem(last=False)
        return record

    def _descriptor_lease(self, bindings, descriptor_layout):
        vk = self.vk
        free_pools = self._free_descriptor_pools.get(bindings, [])
        if free_pools:
            return free_pools.pop()
        pool = vk.vkCreateDescriptorPool(
            self._device,
            vk.VkDescriptorPoolCreateInfo(
                maxSets=1,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=bindings
                    )
                ],
            ),
            None,
        )
        try:
            descriptors = vk.vkAllocateDescriptorSets(
                self._device,
                vk.VkDescriptorSetAllocateInfo(
                    descriptorPool=pool, pSetLayouts=[descriptor_layout]
                ),
            )
            # Native arrays own their storage for the complete descriptor lease.
            # A lease is never reused while a submitted command references it.
            infos = vk.ffi.new("VkDescriptorBufferInfo[]", bindings)
            writes = vk.ffi.new("VkWriteDescriptorSet[]", bindings)
            for i in range(bindings):
                writes[i].sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
                writes[i].dstSet = descriptors[0]
                writes[i].dstBinding = i
                writes[i].descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                writes[i].descriptorCount = 1
                writes[i].pBufferInfo = vk.ffi.addressof(infos, i)
            return pool, bindings, descriptors, infos, writes
        except Exception:
            vk.vkDestroyDescriptorPool(self._device, pool, None)
            raise

    def _run(self, body, count, buffers, scalars=(), *, cooperative=False, shared=""):
        if not count:
            return
        vk = self.vk
        if len(buffers) > self._max_storage_bindings:
            raise NotImplementedError("Vulkan device storage-buffer descriptor limit exceeded")
        with self._lock:
            source, scalar_layout, push_size, width, height = self._dispatch_template(
                body,
                count,
                tuple(np.dtype(dt) for _, dt in buffers),
                tuple(s.dtype for s in scalars),
                cooperative,
                shared,
            )
            push_data = bytearray(push_size)
            for scalar, (offset, dtype) in zip(scalars, scalar_layout):
                push_data[offset : offset + dtype.itemsize] = np.asarray(
                    scalar._data, dtype=dtype
                ).tobytes()
            pipeline, layout, descriptor_layout = self._pipeline(source, len(buffers), push_size)
            push = len(buffers) <= self._push_descriptor_limit
            if push:
                # vkCmdPushDescriptorSetKHR snapshots these host records while
                # recording. Reuse under _lock; GPU buffer leases stay in _pending.
                records = self._push_records.get(len(buffers))
                if records is None:
                    infos = vk.ffi.new("VkDescriptorBufferInfo[]", len(buffers))
                    writes = vk.ffi.new("VkWriteDescriptorSet[]", len(buffers))
                    for i in range(len(buffers)):
                        writes[i].sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
                        writes[i].dstBinding = i
                        writes[i].descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                        writes[i].descriptorCount = 1
                        writes[i].pBufferInfo = vk.ffi.addressof(infos, i)
                    records = self._push_records[len(buffers)] = infos, writes
                infos, writes = records
                lease = None
            else:
                lease = self._descriptor_lease(len(buffers), descriptor_layout)
                _pool, _bindings, descriptors, infos, writes = lease
            for i, (buffer, _) in enumerate(buffers):
                infos[i].buffer = buffer.handle
                infos[i].range = buffer.size
            if not push:
                vk.vkUpdateDescriptorSets(self._device, len(buffers), writes, 0, vk.ffi.NULL)
            command = self._begin()
            self._barrier(command)
            vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
            if push:
                vk.vkCmdPushDescriptorSetKHR(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, len(buffers), writes
                )
            else:
                vk.vkCmdBindDescriptorSets(
                    command,
                    vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    layout,
                    0,
                    1,
                    descriptors,
                    0,
                    vk.ffi.NULL,
                )
            if push_data:
                vk.vkCmdPushConstants(
                    command,
                    layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    push_size,
                    vk.ffi.from_buffer(push_data),
                )
            vk.vkCmdDispatch(command, width, height, 1)
            self._submit(command, [b for b, _ in buffers], lease)

    def residual_matvec(self, a, x, b):
        from ._gemv import residual_matvec_data

        return residual_matvec_data(self, a, x, b)

    def norm(self, a, axes, keepdims, dtype):
        from ._norm import norm_data

        return norm_data(self, a, axes, keepdims, dtype)

    def elementwise(self, name, args, shape, dtype):
        if name not in _EXPRESSIONS:
            raise NotImplementedError(f"Vulkan does not implement {name}")
        if name in ("exp", "log", "sin", "cos", "tanh", "power") and any(
            a.dtype == np.dtype("float64") for a in args
        ):
            raise NotImplementedError(
                f"Vulkan float64 {name} is not supported by this GLSL implementation"
            )
        out = self.empty(shape, dtype)
        buffers, scalars, reads = [], [], []
        for a, label in zip(args, ("a", "b", "c")):
            self.validate_dtype(a.dtype)
            if getattr(a, "_host_scalar", False):
                expression = f"pc.s{len(scalars)}"
                scalars.append(a)
            else:
                expression = f"x{len(buffers)}[{_offset(shape, a.shape)}]"
                buffers.append((a._data, a.dtype))
            reads.append(f"{_type(a.dtype)} {label} = {expression};")
        expression = _EXPRESSIONS[name]
        if name == "absolute" and args[0].dtype.kind in "bu":
            expression = "a"
        if name == "isfinite" and args[0].dtype.kind in "biu":
            expression = "true"
        if np.dtype(dtype).kind == "b":
            if name not in _COMPARISONS:
                expression = f"({expression}) != {_type(args[0].dtype)}(0)"
            expression = f"(({expression}) ? 1u : 0u)"
        else:
            expression = f"{_type(dtype)}({expression})"
        reads.append(f"x{len(buffers)}[i] = {expression};")
        buffers.append((out, dtype))
        self._run("\n".join(reads), prod(shape), buffers, scalars)
        return out

    def _fused_expression(self, root, group, leaves):
        """Return value-only metadata and GLSL; never retain operand storage."""
        indices = {id(node): i for i, node in enumerate((*leaves, *group))}
        key = (
            "expression",
            root.shape,
            tuple((leaf.shape, leaf.dtype.str, leaf.op == "scalar") for leaf in leaves),
            tuple(
                (
                    node.op,
                    node.dtype.str,
                    tuple(dt.str for dt in node.resolved),
                    tuple(indices[id(child)] for child in node.inputs),
                )
                for node in group
            ),
            indices[id(root)],
        )
        with self._lock:
            cached = self._fusion_bodies.get(key)
            if cached is not None:
                self._fusion_bodies.move_to_end(key)
                return key, cached
            reads, buffers, scalars = [], 0, 0
            for i, leaf in enumerate(leaves):
                self.validate_dtype(leaf.dtype)
                if leaf.op == "scalar":
                    read = f"pc.s{scalars}"
                    scalars += 1
                else:
                    read = f"x{buffers}[{_offset(root.shape, leaf.shape)}]"
                    buffers += 1
                reads.append(f"{_type(leaf.dtype)} v{i} = {read};")
            for i, node in enumerate(group, len(leaves)):
                self.validate_dtype(node.dtype)
                if node.op not in _EXPRESSIONS:
                    raise NotImplementedError(f"Vulkan does not implement {node.op}")
                if node.op in ("exp", "log", "sin", "cos", "tanh", "power") and any(
                    dt == np.dtype("float64") for dt in node.resolved[:-1]
                ):
                    raise NotImplementedError(
                        f"Vulkan float64 {node.op} is not supported by this GLSL implementation"
                    )
                qualifier = "precise " if node.dtype.kind == "f" else ""
                reads.append(f"{qualifier}{_type(node.dtype)} v{i}; {{")
                for label, child, dtype in zip(("a", "b", "c"), node.inputs, node.resolved):
                    value = f"v{indices[id(child)]}"
                    if dtype.kind == "b":
                        value = f"(({value} != {_type(child.dtype)}(0)) ? 1u : 0u)"
                    reads.append(f"{_type(dtype)} {label} = {_type(dtype)}({value});")
                expression = _EXPRESSIONS[node.op]
                if node.op == "absolute" and node.resolved[0].kind in "bu":
                    expression = "a"
                if node.op == "isfinite" and node.resolved[0].kind in "biu":
                    expression = "true"
                if node.dtype.kind == "b":
                    if node.op not in _COMPARISONS:
                        expression = f"({expression}) != {_type(node.resolved[0])}(0)"
                    expression = f"(({expression}) ? 1u : 0u)"
                reads.append(f"v{i} = {_type(node.dtype)}({expression}); }}")
            cached = "\n".join(reads), f"v{indices[id(root)]}", buffers
            self._fusion_bodies[key] = cached
            if len(self._fusion_bodies) > 128:
                self._fusion_bodies.popitem(last=False)
            return key, cached

    @staticmethod
    def _fused_operands(operands):
        buffers, scalars = [], []
        for operand in operands:
            if getattr(operand, "_host_scalar", False):
                scalars.append(operand)
            else:
                buffers.append((operand._data, operand.dtype))
        return buffers, scalars

    def fused_elementwise(self, root, group, leaves, operands):
        """Lower a typed expression group to one shader, binding values per call."""
        _, (expression, value, output) = self._fused_expression(root, group, leaves)
        buffers, scalars = self._fused_operands(operands)
        out = self.empty(root.shape, root.dtype)
        buffers.append((out, root.dtype))
        self._run(f"{expression}\nx{output}[i] = {value};", prod(root.shape), buffers, scalars)
        return out

    def fused_reduce(self, node, group, leaves, operands):
        """Map into reduction accumulators; large axes use compact GPU partials.

        The producer is evaluated in its resolved dtype before converting into
        the accumulator dtype. No mapped array or host scalar result is created.
        """
        from ._reductions import combine, identity

        mapped = node.inputs[0]
        if mapped.size > 2**31 - 1:
            raise ValueError("Mapped expression exceeds the Vulkan backend's indexing limits")
        expression_key, (expression, value, output) = self._fused_expression(mapped, group, leaves)
        self.validate_dtype(node.dtype)
        axes = node.attrs["axes"]
        operation = node.attrs.get("operation", "sum")
        key = "reduce", expression_key, tuple(axes), node.dtype.str, operation
        with self._lock:
            cached = self._fusion_bodies.get(key)
            if cached is None:
                out_shape = tuple(d for i, d in enumerate(mapped.shape) if i not in axes)
                reduced_shape = tuple(mapped.shape[i] for i in axes)
                reduction_size, count = prod(reduced_shape), prod(out_shape)
                dt = _type(node.dtype)
                neutral = (
                    f"{dt}(0)"
                    if operation == "sum"
                    else identity(operation, node.dtype, dt, "vulkan")
                )
                qualifier = "precise " if node.dtype.kind == "f" else ""
                base_terms, reduction_terms = [], []
                oi = ri = 0
                for axis, (dim, stride) in enumerate(zip(mapped.shape, _strides(mapped.shape))):
                    if axis in axes:
                        reduction_terms.append(
                            f"((r / {max(1, prod(reduced_shape[ri + 1 :]))}) % {max(1, dim)}) * {stride}"
                        )
                        ri += 1
                    else:
                        base_terms.append(
                            f"((o / {max(1, prod(out_shape[oi + 1 :]))}) % {max(1, dim)}) * {stride}"
                        )
                        oi += 1
                index = (
                    f"({' + '.join(base_terms) or '0'}) + ({' + '.join(reduction_terms) or '0'})"
                )
                if operation == "sum":
                    accumulate = (
                        f"acc |= uint({value} != {_type(mapped.dtype)}(0));"
                        if node.dtype.kind == "b"
                        else f"acc += {dt}({value});"
                    )
                else:
                    accumulate = (
                        f"acc = {combine(operation, node.dtype, dt, 'vulkan', 'acc', value)};"
                    )
                mapping = f"int i = {index};\n{expression}\n{accumulate}"
                contiguous = axes == tuple(range(mapped.ndim - len(axes), mapped.ndim))
                cooperative = reduction_size > 256 or (contiguous and reduction_size >= 32)
                chunks = (reduction_size + 4095) // 4096 if cooperative else 1
                if not reduction_size:
                    # Avoid emitting invalid broadcast indices for empty input
                    # dimensions: no producer element is evaluated in this case.
                    body = f"x{output}[i] = {neutral};"
                elif not cooperative:
                    body = (
                        f"int o=i; {qualifier}{dt} acc={neutral}; "
                        f"for(int r=0;r<{reduction_size};++r) {{ {mapping} }} "
                        f"x{output}[o]=acc;"
                    )
                else:
                    merge = (
                        f"scratch[lane] {'|=' if node.dtype.kind == 'b' else '+='} scratch[lane+step];"
                        if operation == "sum"
                        else f"scratch[lane] = {combine(operation, node.dtype, dt, 'vulkan', 'scratch[lane]', 'scratch[lane+step]')};"
                    )
                    body = (
                        f"int o=i/{chunks}, chunk=i%{chunks}, lane=int(gl_LocalInvocationID.x); "
                        f"{qualifier}{dt} acc={neutral}; "
                        # Increment a chunk-local index so the final loop step
                        # cannot cross INT_MAX when the mapped size approaches it.
                        f"for(int j=lane;j<min(4096,{reduction_size}-chunk*4096);j+=64) {{ "
                        f"int r=chunk*4096+j; {mapping} }} "
                        "scratch[lane]=acc; barrier(); "
                        f"for(int step=32;step>0;step/=2) {{ if(lane<step) {merge} barrier(); }} "
                        f"if(lane==0) x{output}[i]=scratch[0];"
                    )
                shared = f"shared {dt} scratch[64];" if cooperative else ""
                cached = body, count, chunks, cooperative, shared
                self._fusion_bodies[key] = cached
                if len(self._fusion_bodies) > 128:
                    self._fusion_bodies.popitem(last=False)
            else:
                self._fusion_bodies.move_to_end(key)
        body, count, chunks, cooperative, shared = cached
        buffers, scalars = self._fused_operands(operands)
        out = self.empty((count, chunks), node.dtype)
        buffers.append((out, node.dtype))
        self._run(body, count * chunks, buffers, scalars, cooperative=cooperative, shared=shared)
        if chunks > 1:
            from ._array import ndarray

            options = {} if operation == "sum" else {"operation": operation}
            out = self.reduce(
                ndarray._wrap(out, (count, chunks), node.dtype, self),
                (1,),
                False,
                node.dtype,
                **options,
            )
        return out

    def gather(self, a, shape, offset, source_strides):
        out = self.empty(shape, a.dtype)
        terms = [
            f"((i / {max(1, s)}) % {max(1, d)}) * ({t})"
            for s, d, t in zip(_strides(shape), shape, source_strides)
        ]
        index = " + ".join([str(offset), *terms])
        self._run(f"x1[i] = x0[{index}];", prod(shape), [(a._data, a.dtype), (out, a.dtype)])
        return out

    def reduce(self, a, axes, keepdims, dtype, *, operation="sum"):
        from ._reductions import combine, identity

        dt = _type(dtype)
        neutral = f"{dt}(0)" if operation == "sum" else identity(operation, dtype, dt, "vulkan")
        merge = (
            f"scratch[lane] {'|=' if np.dtype(dtype).kind == 'b' else '+='} scratch[lane+step];"
            if operation == "sum"
            else f"scratch[lane] = {combine(operation, dtype, dt, 'vulkan', 'scratch[lane]', 'scratch[lane+step]')};"
        )
        out_shape = tuple(d for i, d in enumerate(a.shape) if i not in axes)
        reduced_shape = tuple(a.shape[i] for i in axes)
        out = self.empty(out_shape, dtype)
        base_terms, reduction_terms = [], []
        oi = ri = 0
        for axis, (dim, stride) in enumerate(zip(a.shape, _strides(a.shape))):
            if axis in axes:
                reduction_terms.append(
                    f"((r / {max(1, prod(reduced_shape[ri + 1 :]))}) % {max(1, dim)}) * {stride}"
                )
                ri += 1
            else:
                base_terms.append(
                    f"((i / {max(1, prod(out_shape[oi + 1 :]))}) % {max(1, dim)}) * {stride}"
                )
                oi += 1
        base, index = " + ".join(base_terms) or "0", " + ".join(reduction_terms) or "0"

        def accumulate(value):
            if operation != "sum":
                return f"acc = {combine(operation, dtype, dt, 'vulkan', 'acc', value)};"
            return (
                f"acc |= uint({value} != {_type(a.dtype)}(0));"
                if np.dtype(dtype).kind == "b"
                else f"acc += {_type(dtype)}({value});"
            )

        if len(axes) == a.ndim and a.size > 256:
            count = (a.size + 255) // 256
            partial = self.empty((count,), dtype)
            self._run(
                f"int lane=int(gl_LocalInvocationID.x); {_type(dtype)} acc={neutral}; "
                f"for(int j=i*256+lane;j<min((i+1)*256,{a.size});j+=64) {{ {accumulate('x0[j]')} }} "
                f"scratch[lane]=acc; barrier(); "
                f"for(int step=32;step>0;step/=2) {{ if(lane<step) {merge} barrier(); }} "
                "if(lane==0) x1[i]=scratch[0];",
                count,
                [(a._data, a.dtype), (partial, dtype)],
                cooperative=True,
                shared=f"shared {_type(dtype)} scratch[64];",
            )
            from ._array import ndarray

            options = {} if operation == "sum" else {"operation": operation}
            return self.reduce(
                ndarray._wrap(partial, (count,), dtype, self),
                (0,),
                False,
                dtype,
                **options,
            )
        if prod(reduced_shape) >= 64:
            self._run(
                f"int lane=int(gl_LocalInvocationID.x); {_type(dtype)} acc={neutral}; "
                f"for(int r=lane;r<{prod(reduced_shape)};r+=64) {{ {accumulate(f'x0[({base})+({index})]')} }} "
                f"scratch[lane]=acc; barrier(); "
                f"for(int step=32;step>0;step/=2) {{ if(lane<step) {merge} barrier(); }} "
                "if(lane==0) x1[i]=scratch[0];",
                prod(out_shape),
                [(a._data, a.dtype), (out, dtype)],
                cooperative=True,
                shared=f"shared {_type(dtype)} scratch[64];",
            )
        else:
            self._run(
                f"{_type(dtype)} acc = {neutral}; for (int r=0;r<{prod(reduced_shape)};++r) {{ {accumulate(f'x0[({base})+({index})]')} }} x1[i]=acc;",
                prod(out_shape),
                [(a._data, a.dtype), (out, dtype)],
            )
        return out

    def matmul(self, a, b, shape, dtype):
        if not prod(shape) or not a.shape[-1]:
            return self.full(shape, 0, dtype)
        if (
            self._register_gemm
            and np.dtype(dtype) == np.dtype("float32")
            and a.ndim > 1
            and b.ndim > 1
            and min(a.shape[-2], b.shape[-1]) >= 32
            and a.shape[-1] >= 16
        ):
            from ._gemm import register_gemm_data

            return register_gemm_data(self, a, b, shape)
        m = a.shape[-2] if a.ndim > 1 else 1
        k = a.shape[-1]
        n = b.shape[-1] if b.ndim > 1 else 1
        a_batch = a.shape[:-2] if a.ndim > 1 else ()
        b_batch = b.shape[:-2] if b.ndim > 1 else ()
        batch_shape = np.broadcast_shapes(a_batch, b_batch)
        a_offset = _offset(batch_shape, a_batch).replace("i /", "batch /")
        b_offset = _offset(batch_shape, b_batch).replace("i /", "batch /")
        bases = f"int abase=({a_offset})*{m * k}, bbase=({b_offset})*{k * n};"
        dt = _type(dtype)
        boolean = np.dtype(dtype).kind == "b"
        if m == n == 1 and k > 4096:
            # Long DOT needs many workgroups, not one workgroup whose lanes each
            # walk the entire input. Partial sums remain entirely on device.
            chunks = (k + 1023) // 1024
            batch_count = prod(batch_shape)
            partial = self.empty((batch_count, chunks), dtype)
            update = (
                "acc |= uint(x0[abase+j] != 0u && x1[bbase+j] != 0u);"
                if boolean
                else "acc += x0[abase+j]*x1[bbase+j];"
            )
            combine = "|=" if boolean else "+="
            self._run(
                f"int lane=int(gl_LocalInvocationID.x), batch=i/{chunks}, chunk=i%{chunks}; {bases} "
                f"{dt} acc={dt}(0); for(int j=chunk*1024+lane;j<min((chunk+1)*1024,{k});j+=64) {{ {update} }} "
                "scratch[lane]=acc; barrier(); "
                f"for(int step=32;step>0;step/=2) {{ if(lane<step) scratch[lane] {combine} scratch[lane+step]; barrier(); }} "
                "if(lane==0) x2[i]=scratch[0];",
                batch_count * chunks,
                [(a._data, dtype), (b._data, dtype), (partial, dtype)],
                cooperative=True,
                shared=f"shared {dt} scratch[64];",
            )
            from ._array import ndarray

            return self.reduce(
                ndarray._wrap(partial, (batch_count, chunks), dtype, self),
                (1,),
                False,
                dtype,
            )
        out = self.empty(shape, dtype)
        buffers = [(a._data, dtype), (b._data, dtype), (out, dtype)]
        if n == 1:
            # A workgroup cooperates on a row/dot; adjacent lanes read adjacent
            # columns, unlike the original serial-per-output GEMV.
            update = (
                f"acc |= uint(x0[abase+row*{k}+j] != 0u && x1[bbase+j] != 0u);"
                if boolean
                else f"acc += x0[abase+row*{k}+j]*x1[bbase+j];"
            )
            combine = "|=" if boolean else "+="
            self._run(
                f"int lane=int(gl_LocalInvocationID.x), batch=i/{m}, row=i%{m}; {bases} "
                f"{dt} acc={dt}(0); for(int j=lane;j<{k};j+=64) {{ {update} }} "
                "scratch[lane]=acc; barrier(); "
                f"for(int step=32;step>0;step/=2) {{ if(lane<step) scratch[lane] {combine} scratch[lane+step]; barrier(); }} "
                "if(lane==0) x2[i]=scratch[0];",
                prod(batch_shape) * m,
                buffers,
                cooperative=True,
                shared=f"shared {dt} scratch[64];",
            )
        else:
            tile_rows, tile_cols = (m + 7) // 8, (n + 7) // 8
            update = (
                "acc |= uint(left_tile[ly*8+j] != 0u && right_tile[j*8+lx] != 0u);"
                if boolean
                else "acc += left_tile[ly*8+j]*right_tile[j*8+lx];"
            )
            self._run(
                f"int lane=int(gl_LocalInvocationID.x), ly=lane/8, lx=lane%8; "
                f"int batch=i/{tile_rows * tile_cols}, tile=i%{tile_rows * tile_cols}; "
                f"int row=(tile/{tile_cols})*8+ly, col=(tile%{tile_cols})*8+lx; {bases} "
                f"{dt} acc={dt}(0); for(int start=0;start<{k};start+=8) {{ "
                f"left_tile[lane]=(row<{m} && start+lx<{k}) ? x0[abase+row*{k}+start+lx] : {dt}(0); "
                f"right_tile[lane]=(start+ly<{k} && col<{n}) ? x1[bbase+(start+ly)*{n}+col] : {dt}(0); "
                f"barrier(); for(int j=0;j<8;++j) {{ {update} }} barrier(); }} "
                f"if(row<{m} && col<{n}) x2[(batch*{m}+row)*{n}+col]=acc;",
                prod(batch_shape) * tile_rows * tile_cols,
                buffers,
                cooperative=True,
                shared=f"shared {dt} left_tile[64]; shared {dt} right_tile[64];",
            )
        return out

    def synchronize(self):
        with self._lock:
            self._flush_commands()
            if not self._pending:
                return
            if self._completion_mode == "queue":
                self.vk.vkQueueWaitIdle(self._queue)
            else:
                # One reusable marker, submitted only at an observation boundary.
                # Its scope includes all earlier commands on this owned queue.
                # No per-dispatch fence allocation or changed dispatch batching.
                vk = self.vk
                if self._completion_fence is None:
                    # Allocate the host record before owning a native handle.
                    fences = vk.ffi.new("VkFence[]", 1)
                    fence = vk.vkCreateFence(self._device, vk.VkFenceCreateInfo(), None)
                    fences[0] = fence
                    self._completion_fences, self._completion_fence = fences, fence
                try:
                    vk.vkQueueSubmit(self._queue, 0, vk.ffi.NULL, self._completion_fence)
                    vk.vkWaitForFences(
                        self._device, 1, self._completion_fences, vk.VK_TRUE, 2**64 - 1
                    )
                    vk.vkResetFences(self._device, 1, self._completion_fences)
                except Exception:
                    # Keep every in-flight reference on an uncertain result;
                    # never reuse a fence or resources after a failed completion.
                    self._completion_failed = True
                    raise
            pending, self._pending = self._pending, []
            self._queued_operations = 0
            for command, pools, references in pending:
                for pool in pools:
                    handle, bindings, _descriptors, _infos, _writes = pool
                    if sum(map(len, self._free_descriptor_pools.values())) < 128:
                        self._free_descriptor_pools.setdefault(bindings, []).append(pool)
                    else:
                        self.vk.vkDestroyDescriptorPool(self._device, handle, None)
                if len(self._free_commands) < 128:
                    self._free_commands.append(command)
                else:
                    self.vk.vkFreeCommandBuffers(self._device, self._command_pool, 1, [command])

    def _destroy_pipeline(self, record):
        pipeline, layout, _descriptor_layout = record
        self.vk.vkDestroyPipeline(self._device, pipeline, None)
        self.vk.vkDestroyPipelineLayout(self._device, layout, None)

    def release_cached_memory(self):
        """Wait for in-flight work and release cached allocations/resources.

        Live arrays remain valid. Device and staging memory together are cached
        up to 128 MiB and 128 allocations, preferring exact capacity.
        """
        with self._lock:
            self.synchronize()
            # Finalizers can return completed leases while native frees allocate
            # Python objects. Detach the current pool before iterating it.
            buffers, self._buffer_cache = self._buffer_cache, OrderedDict()
            self._cached_bytes = self._pool_items = 0
            for entries in buffers.values():
                for handle, memory, _ in entries:
                    self._free_buffer(handle, memory)
            for pools in self._free_descriptor_pools.values():
                for lease in pools:
                    self.vk.vkDestroyDescriptorPool(self._device, lease[0], None)
            self._free_descriptor_pools.clear()
            if self._free_commands:
                self.vk.vkFreeCommandBuffers(
                    self._device,
                    self._command_pool,
                    len(self._free_commands),
                    self._free_commands,
                )
                self._free_commands.clear()

    def close(self):
        if self._device is not None:
            self._closing = True
            self.release_cached_memory()
            for record in self._pipelines.values():
                self._destroy_pipeline(record)
            self._pipelines.clear()
            self._dispatch_templates.clear()
            for layout in self._descriptor_layouts.values():
                self.vk.vkDestroyDescriptorSetLayout(self._device, layout, None)
            self._descriptor_layouts.clear()
            self._push_records.clear()
            if self._completion_fence is not None:
                self.vk.vkDestroyFence(self._device, self._completion_fence, None)
                self._completion_fence = None
            if self._command_pool is not None:
                self.vk.vkDestroyCommandPool(self._device, self._command_pool, None)
            self.vk.vkDestroyDevice(self._device, None)
            self._device = None
        if self._instance is not None:
            self.vk.vkDestroyInstance(self._instance, None)
            self._instance = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - interpreter shutdown is best effort
            pass
