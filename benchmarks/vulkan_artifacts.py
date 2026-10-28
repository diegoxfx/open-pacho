"""Capture Vulkan compiler evidence in a dedicated, single-threaded diagnostic process.

This module never changes the library's runtime defaults. The temporary hooks are
process-global, so callers must not construct or compile other Vulkan backends
concurrently. Pipelines created with capture flags are diagnostic artifacts;
their timing must not be substituted for ordinary production pipeline timing.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from threading import RLock

_EXTENSION = "VK_KHR_pipeline_executable_properties"
_PROPERTIES_EXTENSION = "VK_KHR_get_physical_device_properties2"
_LOCK = RLock()
_ATTEMPTS = 4
_MAX_RECORDS = 65536
_MAX_REPRESENTATION_BYTES = 128 * 1024**2


class CaptureUnavailableError(RuntimeError):
    """The driver or binding cannot expose the requested diagnostic evidence."""


def _text(ffi, value):
    return ffi.string(value).decode("utf-8", errors="replace")


def _procedure(vk, handle, name, *, instance=False):
    getter = vk.lib.vkGetInstanceProcAddr if instance else vk.lib.vkGetDeviceProcAddr
    address = getter(handle, name.encode("ascii"))
    if address == vk.ffi.NULL:
        raise CaptureUnavailableError(f"Vulkan diagnostic entry point is unavailable: {name}")
    return vk.ffi.cast("PFN_" + name, address)


def create_backend(device=None):
    """Create a backend with executable introspection explicitly enabled.

    The caller owns ``close()``. No queue work is submitted here. Unsupported
    extensions/features fail explicitly instead of silently omitting evidence.
    """
    from openpacho import _vulkan as runtime

    vk = runtime._load()
    metadata = {}
    original_instance = runtime._instance
    original_create_device = vk.vkCreateDevice

    def create_instance(binding, extensions=()):
        available = {
            item.extensionName for item in binding.vkEnumerateInstanceExtensionProperties(None)
        }
        if _PROPERTIES_EXTENSION not in available:
            raise CaptureUnavailableError(f"Vulkan instance lacks {_PROPERTIES_EXTENSION}")
        instance = original_instance(
            binding, list(dict.fromkeys([*extensions, _PROPERTIES_EXTENSION]))
        )
        metadata["instance"] = instance
        return instance

    def create_device(physical, info, allocator):
        available = {
            item.extensionName for item in vk.vkEnumerateDeviceExtensionProperties(physical, None)
        }
        if _EXTENSION not in available:
            raise CaptureUnavailableError(f"Vulkan device lacks {_EXTENSION}")
        ffi = vk.ffi
        feature = ffi.new("VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR *")
        feature.sType = (
            vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR
        )
        query = ffi.new("VkPhysicalDeviceFeatures2 *")
        query.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2
        query.pNext = feature
        procedure = _procedure(
            vk, metadata["instance"], "vkGetPhysicalDeviceFeatures2KHR", instance=True
        )
        procedure(physical, query)
        if not feature.pipelineExecutableInfo:
            raise CaptureUnavailableError("Vulkan device has pipelineExecutableInfo=false")

        names = [
            _text(ffi, info.ppEnabledExtensionNames[i]) for i in range(info.enabledExtensionCount)
        ]
        names = list(dict.fromkeys([*names, _EXTENSION]))
        # Keep the strings, pointer array and feature alive through vkCreateDevice.
        strings = [ffi.new("char[]", name.encode("ascii")) for name in names]
        pointers = ffi.new("char *[]", strings)
        previous = info.pNext, info.ppEnabledExtensionNames, info.enabledExtensionCount
        feature.pNext = ffi.cast("void *", info.pNext)
        feature.pipelineExecutableInfo = vk.VK_TRUE
        try:
            info.pNext = feature
            info.ppEnabledExtensionNames = pointers
            info.enabledExtensionCount = len(names)
            result = original_create_device(physical, info, allocator)
        finally:
            info.pNext, info.ppEnabledExtensionNames, info.enabledExtensionCount = previous
        metadata["device_extensions"] = names
        return result

    with _LOCK:
        runtime._instance = create_instance
        vk.vkCreateDevice = create_device
        try:
            backend = runtime.VulkanBackend(device=device)
        finally:
            runtime._instance = original_instance
            vk.vkCreateDevice = original_create_device
    metadata.pop("instance", None)
    metadata["pipelineExecutableInfo"] = True
    backend._executable_capture_metadata = metadata
    return backend


def _check_result(vk, status, label):
    if status not in (vk.VK_SUCCESS, vk.VK_INCOMPLETE):
        raise RuntimeError(f"{label} returned VkResult {int(status)}")


def _records(vk, function, device, info, ctype, stype, *, data=False):
    """Query count, records and optional payload, restarting on VK_INCOMPLETE.

    Unlike generated binding convenience wrappers, every output record has its
    required sType before each native query, and payload storage stays alive.
    """
    ffi = vk.ffi
    for _ in range(_ATTEMPTS):
        count = ffi.new("uint32_t *")
        status = function(device, info, count, ffi.NULL)
        _check_result(vk, status, ctype)
        if status == vk.VK_INCOMPLETE:
            continue
        capacity = int(count[0])
        if capacity > _MAX_RECORDS:
            raise RuntimeError(f"{ctype} returned an unreasonable record count: {capacity}")
        if not capacity:
            return [], []
        values = ffi.new(ctype + "[]", capacity)
        for item in values:
            item.sType = stype
        status = function(device, info, count, values)
        _check_result(vk, status, ctype)
        if status == vk.VK_INCOMPLETE or count[0] > capacity:
            continue
        length = int(count[0])
        if not data:
            return [values[i] for i in range(length)], [values]
        sizes = [int(values[i].dataSize) for i in range(length)]
        if sum(sizes) > _MAX_REPRESENTATION_BYTES:
            raise RuntimeError(f"{ctype} exceeds the diagnostic payload size limit")
        buffers = [ffi.new("char[]", max(1, size)) for size in sizes]
        for index, buffer in enumerate(buffers):
            values[index].pData = buffer
        status = function(device, info, count, values)
        _check_result(vk, status, ctype)
        if status == vk.VK_INCOMPLETE or count[0] > length:
            continue
        if any(int(values[i].dataSize) > sizes[i] for i in range(int(count[0]))):
            continue
        return [values[i] for i in range(int(count[0]))], [values, *buffers]
    raise RuntimeError(
        f"{ctype} remained incomplete after {_ATTEMPTS} attempts; no truncated export"
    )


def _statistic(vk, item):
    fields = {
        vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: ("BOOL32", "b32", bool),
        vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: ("INT64", "i64", int),
        vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: ("UINT64", "u64", int),
        vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: ("FLOAT64", "f64", float),
    }
    if item.format not in fields:
        raise RuntimeError(f"Unknown pipeline statistic format: {int(item.format)}")
    name, field, cast = fields[item.format]
    value = cast(getattr(item.value, field))
    if isinstance(value, float) and not math.isfinite(value):
        value = str(value)
    return {
        "name": _text(vk.ffi, item.name),
        "description": _text(vk.ffi, item.description),
        "format": name,
        "format_code": int(item.format),
        "value": value,
        "raw_value_bytes_hex": bytes(
            vk.ffi.buffer(vk.ffi.addressof(item.value), vk.ffi.sizeof(item.value))
        ).hex(),
    }


def _statistic_abi(vk):
    """Record the layout actually used by CFFI, without querying a GPU."""
    ctype = "VkPipelineExecutableStatisticKHR"
    return {
        "byte_order": sys.byteorder,
        "pointer_bytes": vk.ffi.sizeof("void *"),
        "statistic_struct_bytes": vk.ffi.sizeof(ctype),
        "statistic_struct_alignment": vk.ffi.alignof(ctype),
        "statistic_offsets": {
            field: vk.ffi.offsetof(ctype, field)
            for field in ("sType", "pNext", "name", "description", "format", "value")
        },
    }


def _evidence_limits(statistics, representations):
    limits = [
        (
            "Statistics are implementation-defined driver reports, not independently verified "
            "hardware measurements."
        )
    ]
    if not representations:
        limits.append(
            "The driver returned no internal representations; this capture cannot establish "
            "native instructions, predication or local-memory spill behavior."
        )
    suspect = []
    for statistic in statistics:
        value = statistic["value"]
        if (
            statistic["name"] == "Local Memory Size"
            and "per-thread" in statistic["description"].lower()
            and statistic["format"] in ("INT64", "UINT64")
            and value >= 2**32
        ):
            suspect.append(
                {
                    "name": statistic["name"],
                    "reported_value": value,
                    "diagnostic_threshold_bytes": 2**32,
                    "reason": (
                        "Reported per-thread local memory is at least 4 GiB. This is a "
                        "diagnostic plausibility threshold, not a Vulkan or hardware limit. "
                        "The raw value is retained and must not be interpreted as an actual "
                        "allocation or evidence of spilling without independent validation."
                    ),
                }
            )
    return limits, suspect


def _file(folder, name, data):
    (folder / name).write_bytes(data)
    return {"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def export_pipeline(backend, source, folder):
    """Export exact compiler input/output and all reported pipeline evidence.

    ``source`` must be a fresh GEMM GLSL source with three storage bindings and
    no push constants. Returns JSON-serializable metadata, also written to disk.
    Driver-provided representation names/descriptions are retained verbatim;
    their order is never interpreted as identifying machine code or an ISA.
    """
    from openpacho import _vulkan as runtime

    if not getattr(backend, "_executable_capture_metadata", None):
        raise CaptureUnavailableError("Use create_backend() to enable executable capture first")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    vk, ffi = backend.vk, backend.vk.ffi
    capture_flags = (
        vk.VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR
        | vk.VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR
    )
    metadata = {
        "schema_version": 2,
        "device": backend.device,
        "device_id": int(backend._properties.deviceID),
        "vendor_id": int(backend._properties.vendorID),
        "driver_version": int(backend._properties.driverVersion),
        "api_version": int(backend._properties.apiVersion),
        "capture": dict(backend._executable_capture_metadata),
        "pipeline_create_flags_added": int(capture_flags),
        "diagnostic_pipeline_not_for_timing": True,
        "source_string_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "statistic_abi": _statistic_abi(vk),
    }
    original_run = runtime.subprocess.run
    original_create = vk.vkCreateComputePipelines

    def compile_and_capture(args, **kwargs):
        result = original_run(args, **kwargs)
        if result.returncode == 0:
            binary_index = args.index("-o") + 1
            binary = Path(args[binary_index]).read_bytes()
            if len(binary) < 20 or len(binary) % 4 or binary[:4] != b"\x03\x02\x23\x07":
                raise RuntimeError("Vulkan compiler did not produce a valid SPIR-V header")
            # The existing compiler writes a .comp file before calling subprocess.
            inputs = [Path(arg) for arg in args[1:] if str(arg).endswith(".comp")]
            if len(inputs) != 1:
                raise RuntimeError("Cannot identify the exact Vulkan compiler source argument")
            metadata["source"] = _file(folder, "source.comp", inputs[0].read_bytes())
            metadata["spirv"] = _file(folder, "shader.spv", binary)
            metadata["compiler"] = {
                "executable": str(args[0]),
                "arguments": [str(arg) for arg in args],
                "stdout": result.stdout,
                "stderr": result.stderr,
                "sha256": hashlib.sha256(Path(args[0]).read_bytes()).hexdigest(),
            }
        return result

    def create_pipeline(device, cache, count, infos, allocator, *args, **kwargs):
        previous = [int(infos[i].flags) for i in range(count)]
        try:
            for index in range(count):
                infos[index].flags |= capture_flags
            return original_create(device, cache, count, infos, allocator, *args, **kwargs)
        finally:
            for index, flags in enumerate(previous):
                infos[index].flags = flags

    with _LOCK, backend._lock:
        if source in backend._pipelines:
            raise ValueError("Pipeline source is already cached; capture requires a fresh source")
        runtime.subprocess.run = compile_and_capture
        vk.vkCreateComputePipelines = create_pipeline
        try:
            pipeline = backend._pipeline(source, 3, 0)[0]
        finally:
            runtime.subprocess.run = original_run
            vk.vkCreateComputePipelines = original_create
        if "spirv" not in metadata:
            raise RuntimeError("Pipeline creation did not capture its exact SPIR-V")
        pipeline_info = ffi.new("VkPipelineInfoKHR *")
        pipeline_info.sType = vk.VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR
        pipeline_info.pipeline = pipeline
        properties, property_storage = _records(
            vk,
            _procedure(vk, backend._device, "vkGetPipelineExecutablePropertiesKHR"),
            backend._device,
            pipeline_info,
            "VkPipelineExecutablePropertiesKHR",
            vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR,
        )
        metadata["executables"] = []
        for index, prop in enumerate(properties):
            info = ffi.new("VkPipelineExecutableInfoKHR *")
            info.sType = vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR
            info.pipeline = pipeline
            info.executableIndex = index
            statistics, statistic_storage = _records(
                vk,
                _procedure(vk, backend._device, "vkGetPipelineExecutableStatisticsKHR"),
                backend._device,
                info,
                "VkPipelineExecutableStatisticKHR",
                vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR,
            )
            representations, representation_storage = _records(
                vk,
                _procedure(
                    vk, backend._device, "vkGetPipelineExecutableInternalRepresentationsKHR"
                ),
                backend._device,
                info,
                "VkPipelineExecutableInternalRepresentationKHR",
                vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR,
                data=True,
            )
            exported = []
            for number, representation in enumerate(representations):
                is_text = bool(representation.isText)
                # Preserve every byte, including a trailing text NUL if supplied.
                payload = bytes(ffi.buffer(representation.pData, int(representation.dataSize)))
                name = f"executable-{index}-representation-{number}.{'txt' if is_text else 'bin'}"
                exported.append(
                    {
                        "name": _text(ffi, representation.name),
                        "description": _text(ffi, representation.description),
                        "isText": is_text,
                        **_file(folder, name, payload),
                    }
                )
            exported_statistics = [_statistic(vk, item) for item in statistics]
            limitations, suspect = _evidence_limits(exported_statistics, exported)
            metadata["executables"].append(
                {
                    "index": index,
                    "name": _text(ffi, prop.name),
                    "description": _text(ffi, prop.description),
                    "stages": int(prop.stages),
                    "subgroup_size": int(prop.subgroupSize),
                    "statistics": exported_statistics,
                    "representations": exported,
                    "limitations": limitations,
                    "suspect_statistics": suspect,
                }
            )
            # These owners must outlive all accesses to their CFFI record views.
            del statistic_storage, representation_storage
        del property_storage
    (folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata
