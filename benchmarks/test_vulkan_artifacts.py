"""ABI-query and hook-lifetime tests; no Vulkan device or driver calls."""

import ctypes
import json
import sys
from threading import RLock
from types import SimpleNamespace

import pytest

from benchmarks import vulkan_artifacts as artifacts


@pytest.fixture
def vk():
    ffi = pytest.importorskip("cffi").FFI()
    ffi.cdef(
        """
        typedef unsigned int uint32_t;
        typedef unsigned long long uint64_t;
        typedef long long int64_t;
        typedef struct {
          int sType; void *pNext; unsigned int stages;
          char name[256]; char description[256]; uint32_t subgroupSize;
        } VkPipelineExecutablePropertiesKHR;
        typedef union {uint32_t b32; int64_t i64; uint64_t u64; double f64;} Value;
        typedef struct {
          int sType; void *pNext; char name[256]; char description[256]; int format; Value value;
        } VkPipelineExecutableStatisticKHR;
        typedef struct {
          int sType; void *pNext; char name[256]; char description[256];
          uint32_t isText; size_t dataSize; void *pData;
        } VkPipelineExecutableInternalRepresentationKHR;
        typedef struct {
          int sType; void *pNext; uint32_t pipelineExecutableInfo;
        } VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR;
        typedef struct {int sType; void *pNext;} VkPhysicalDeviceFeatures2;
        typedef struct {
          const void *pNext; uint32_t enabledExtensionCount; const char *const *ppEnabledExtensionNames;
        } VkDeviceCreateInfo;
        typedef struct {int sType; const void *pNext; uint64_t pipeline;} VkPipelineInfoKHR;
        typedef struct {
          int sType; const void *pNext; uint64_t pipeline; uint32_t executableIndex;
        } VkPipelineExecutableInfoKHR;
        """
    )
    return SimpleNamespace(
        ffi=ffi,
        VK_SUCCESS=0,
        VK_INCOMPLETE=5,
        VK_TRUE=1,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR=20,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2=21,
        VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR=30,
        VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR=31,
        VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR=32,
        VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR=33,
        VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR=34,
        VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR=0,
        VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR=1,
        VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR=2,
        VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR=3,
        VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR=64,
        VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR=128,
    )


def test_records_initialize_stype_and_restart_incomplete(vk):
    ffi = vk.ffi
    queries = []

    def query(device, info, count, values):
        assert device == "device" and info == "info"
        queries.append(values == ffi.NULL)
        if values == ffi.NULL:
            count[0] = 1
        else:
            assert values[0].sType == 37
            if len(queries) == 2:
                return vk.VK_INCOMPLETE
            values[0].name = b"complete executable"
            values[0].subgroupSize = 32
        return vk.VK_SUCCESS

    records, storage = artifacts._records(
        vk, query, "device", "info", "VkPipelineExecutablePropertiesKHR", 37
    )
    assert queries == [True, False, True, False]
    assert ffi.string(records[0].name) == b"complete executable"
    assert records[0].subgroupSize == 32
    assert storage


def test_representation_uses_three_calls_and_preserves_binary_and_text(vk):
    ffi = vk.ffi
    payloads = [b"SASS?\n\x00", b"\x00\xff\x0a"]
    calls = []

    def query(_device, _info, count, values):
        if values == ffi.NULL:
            calls.append("count")
            count[0] = 2
            return vk.VK_SUCCESS
        assert count[0] == 2
        calls.append("sizes" if values[0].pData == ffi.NULL else "data")
        for i, payload in enumerate(payloads):
            assert values[i].sType == 41
            values[i].name = f"IR {i}".encode()
            values[i].description = b"No assumed native ISA"
            values[i].isText = i == 0
            if values[i].pData != ffi.NULL:
                assert values[i].dataSize == len(payload)
                ffi.buffer(values[i].pData, len(payload))[:] = payload
            values[i].dataSize = len(payload)
        return vk.VK_SUCCESS

    records, storage = artifacts._records(
        vk, query, None, None, "VkPipelineExecutableInternalRepresentationKHR", 41, data=True
    )
    assert calls == ["count", "sizes", "data"]
    assert [bytes(ffi.buffer(item.pData, item.dataSize)) for item in records] == payloads
    assert len(storage) == 3


@pytest.mark.parametrize("incomplete_phase", ["sizes", "data"])
def test_representation_retries_without_exporting_truncated_buffers(vk, incomplete_phase):
    ffi = vk.ffi
    attempts = 0

    def query(_device, _info, count, values):
        nonlocal attempts
        if values == ffi.NULL:
            attempts += 1
            count[0] = 1
            return vk.VK_SUCCESS
        phase = "sizes" if values[0].pData == ffi.NULL else "data"
        assert values[0].sType == 41
        if phase == incomplete_phase and attempts == 1:
            return vk.VK_INCOMPLETE
        values[0].dataSize = 3
        if phase == "data":
            ffi.buffer(values[0].pData, 3)[:] = b"all"
        return vk.VK_SUCCESS

    records, storage = artifacts._records(
        vk, query, None, None, "VkPipelineExecutableInternalRepresentationKHR", 41, data=True
    )
    assert attempts == 2
    assert bytes(ffi.buffer(records[0].pData, 3)) == b"all"
    assert storage


def test_incomplete_is_bounded_and_native_errors_are_not_silenced(vk):
    calls = 0

    def incomplete(*_args):
        nonlocal calls
        calls += 1
        return vk.VK_INCOMPLETE

    with pytest.raises(RuntimeError, match="no truncated export"):
        artifacts._records(vk, incomplete, None, None, "VkPipelineExecutablePropertiesKHR", 37)
    assert calls == artifacts._ATTEMPTS
    with pytest.raises(RuntimeError, match="VkResult -4"):
        artifacts._records(vk, lambda *_: -4, None, None, "VkPipelineExecutablePropertiesKHR", 37)


@pytest.mark.parametrize(
    ("code", "field", "value", "name"),
    [
        (0, "b32", 1, "BOOL32"),
        (1, "i64", -7, "INT64"),
        (2, "u64", 2**63, "UINT64"),
        (3, "f64", 1.25, "FLOAT64"),
    ],
)
def test_statistics_preserve_formats_and_full_names(vk, code, field, value, name):
    stat = vk.ffi.new("VkPipelineExecutableStatisticKHR *")
    stat.name = b"full reported name"
    stat.description = b"full reported description"
    stat.format = code
    setattr(stat.value, field, value)
    result = artifacts._statistic(vk, stat)
    assert result == {
        "name": "full reported name",
        "description": "full reported description",
        "format": name,
        "format_code": code,
        "value": value,
        "raw_value_bytes_hex": bytes(
            vk.ffi.buffer(vk.ffi.addressof(stat.value), vk.ffi.sizeof(stat.value))
        ).hex(),
    }


def test_real_binding_statistic_abi_matches_independent_native_layout():
    # Loading the optional binding does not create an instance/device or issue
    # driver calls. This specifically checks its shipped CFFI declarations.
    vk = pytest.importorskip("vulkan")

    class Value(ctypes.Union):
        _fields_ = [
            ("b32", ctypes.c_uint32),
            ("i64", ctypes.c_int64),
            ("u64", ctypes.c_uint64),
            ("f64", ctypes.c_double),
        ]

    class Statistic(ctypes.Structure):
        _fields_ = [
            ("sType", ctypes.c_int),
            ("pNext", ctypes.c_void_p),
            ("name", ctypes.c_char * 256),
            ("description", ctypes.c_char * 256),
            ("format", ctypes.c_int),
            ("value", Value),
        ]

    abi = artifacts._statistic_abi(vk)
    assert abi["statistic_struct_bytes"] == ctypes.sizeof(Statistic)
    assert abi["statistic_struct_alignment"] == ctypes.alignment(Statistic)
    assert abi["statistic_offsets"] == {
        name: getattr(Statistic, name).offset for name, _ in Statistic._fields_
    }
    records = vk.ffi.new("VkPipelineExecutableStatisticKHR[]", 5)
    assert bytes(vk.ffi.buffer(records)) == bytes(vk.ffi.sizeof(records))
    for index, value in enumerate([60, 7296, 0, 68719476736, 4096]):
        address = int(vk.ffi.cast("uintptr_t", vk.ffi.addressof(records, index)))
        native = Statistic.from_address(address)
        native.sType = vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR
        native.format = vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR
        native.value.u64 = value
        result = artifacts._statistic(vk, records[index])
        assert result["value"] == value
        assert result["raw_value_bytes_hex"] == value.to_bytes(8, sys.byteorder).hex()


def test_suspect_local_memory_is_preserved_and_explicitly_unverified():
    statistic = {
        "name": "Local Memory Size",
        "description": "Size in bytes required per-thread for local memory",
        "format": "UINT64",
        "value": 68719476736,
    }
    limitations, suspect = artifacts._evidence_limits([statistic], [])
    assert statistic["value"] == 68719476736
    assert suspect[0]["reported_value"] == 68719476736
    assert suspect[0]["diagnostic_threshold_bytes"] == 2**32
    assert "not a Vulkan or hardware limit" in suspect[0]["reason"]
    assert any("no internal representations" in item for item in limitations)
    statistic["value"] = 16
    assert artifacts._evidence_limits([statistic], ["representation"])[1] == []


@pytest.mark.parametrize("supported", [True, False])
def test_backend_enables_feature_and_restores_hooks(monkeypatch, vk, supported):
    from openpacho import _vulkan as runtime

    original_next = vk.ffi.new("char[]", b"existing next")
    requested_extensions = []
    created_devices = []

    def instance(_binding, extensions=()):
        requested_extensions.extend(extensions)
        return "instance"

    def create_device(physical, info, allocator):
        assert physical == "physical" and allocator is None
        feature = vk.ffi.cast(
            "VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR *", info.pNext
        )
        assert feature.pipelineExecutableInfo == 1
        assert feature.pNext == original_next
        names = [
            vk.ffi.string(info.ppEnabledExtensionNames[i]).decode()
            for i in range(info.enabledExtensionCount)
        ]
        assert names == ["existing extension", artifacts._EXTENSION]
        created_devices.append(physical)
        return "device"

    def query(physical, features):
        assert physical == "physical"
        assert features.sType == 21
        feature = vk.ffi.cast(
            "VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR *", features.pNext
        )
        assert feature.sType == 20
        feature.pipelineExecutableInfo = int(supported)

    def backend_constructor(device):
        assert device == "2"
        runtime._instance(vk, [])
        text = vk.ffi.new("char[]", b"existing extension")
        names = vk.ffi.new("char *[]", [text])
        info = vk.ffi.new("VkDeviceCreateInfo *")
        info.pNext = original_next
        info.enabledExtensionCount = 1
        info.ppEnabledExtensionNames = names
        result = vk.vkCreateDevice("physical", info, None)
        assert info.pNext == original_next
        assert info.enabledExtensionCount == 1
        return SimpleNamespace(_device=result)

    vk.vkEnumerateInstanceExtensionProperties = lambda _: [
        SimpleNamespace(extensionName=artifacts._PROPERTIES_EXTENSION)
    ]
    vk.vkEnumerateDeviceExtensionProperties = lambda *_: [
        SimpleNamespace(extensionName=artifacts._EXTENSION)
    ]
    vk.vkCreateDevice = create_device
    monkeypatch.setattr(runtime, "_load", lambda: vk)
    monkeypatch.setattr(runtime, "_instance", instance)
    monkeypatch.setattr(runtime, "VulkanBackend", backend_constructor)
    monkeypatch.setattr(artifacts, "_procedure", lambda *_args, **_kwargs: query)
    if supported:
        backend = artifacts.create_backend(device="2")
        assert backend._executable_capture_metadata["pipelineExecutableInfo"] is True
        assert created_devices == ["physical"]
    else:
        with pytest.raises(artifacts.CaptureUnavailableError, match="pipelineExecutableInfo=false"):
            artifacts.create_backend(device="2")
        assert not created_devices
    assert requested_extensions == [artifacts._PROPERTIES_EXTENSION]
    assert runtime._instance is instance
    assert vk.vkCreateDevice is create_device


def test_export_restores_hooks_when_pipeline_fails(monkeypatch, tmp_path, vk):
    from openpacho import _vulkan as runtime

    original_run = runtime.subprocess.run
    original_create = lambda *_: None
    vk.vkCreateComputePipelines = original_create

    def fail_pipeline(source, bindings, push_size):
        assert (source, bindings, push_size) == ("shader", 3, 0)
        assert runtime.subprocess.run is not original_run
        assert vk.vkCreateComputePipelines is not original_create
        raise RuntimeError("pipeline failed")

    backend = SimpleNamespace(
        _executable_capture_metadata={"pipelineExecutableInfo": True},
        vk=vk,
        device="fake",
        _properties=SimpleNamespace(deviceID=1, vendorID=2, driverVersion=3, apiVersion=4),
        _lock=RLock(),
        _pipelines={},
        _pipeline=fail_pipeline,
    )
    with pytest.raises(RuntimeError, match="pipeline failed"):
        artifacts.export_pipeline(backend, "shader", tmp_path)
    assert runtime.subprocess.run is original_run
    assert vk.vkCreateComputePipelines is original_create
    assert not (tmp_path / "metadata.json").exists()


def test_unconfigured_backend_fails_before_writing(tmp_path):
    with pytest.raises(artifacts.CaptureUnavailableError, match="create_backend"):
        artifacts.export_pipeline(SimpleNamespace(), "shader", tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_export_preserves_exact_compiler_bytes_and_all_named_representations(
    monkeypatch, tmp_path, vk
):
    from openpacho import _vulkan as runtime

    compiler = tmp_path / "compiler.exe"
    compiler.write_bytes(b"simulated compiler")
    shader = tmp_path / "compute.comp"
    shader.write_bytes(b"exact source\r\n")
    spirv = b"\x03\x02\x23\x07" + b"\x00" * 16
    binary = tmp_path / "compute.spv"
    args = [str(compiler), "-V", "--target-env", "vulkan1.0", str(shader), "-o", str(binary)]

    def compiler_run(actual_args, **kwargs):
        assert actual_args == args
        binary.write_bytes(spirv)
        return SimpleNamespace(returncode=0, stdout="compiled", stderr="")

    def create_pipeline(device, cache, count, infos, allocator):
        assert infos[0].flags == 8 | 64 | 128
        return [123]

    def pipeline(source, bindings, push_size):
        runtime.subprocess.run(args, check=False)
        info = SimpleNamespace(flags=8)
        result = vk.vkCreateComputePipelines(None, None, 1, [info], None)
        assert info.flags == 8
        return result[0], None, None

    def records(vk, function, device, info, ctype, stype, *, data=False):
        values = vk.ffi.new(ctype + "[]", 1)
        item = values[0]
        item.name = b"A vendor-provided name, not inferred ISA"
        item.description = b"Complete vendor description"
        if ctype == "VkPipelineExecutablePropertiesKHR":
            assert info.sType == 30 and info.pipeline == 123
            item.stages, item.subgroupSize = 32, 32
        elif ctype == "VkPipelineExecutableStatisticKHR":
            assert info.sType == 31 and info.executableIndex == 0
            item.format, item.value.u64 = 2, 71
        else:
            assert data
            payload = vk.ffi.new("char[]", b"text with terminator")
            item.isText, item.dataSize, item.pData = 1, 21, payload
            return [item], [values, payload]
        return [item], [values]

    monkeypatch.setattr(runtime.subprocess, "run", compiler_run)
    vk.vkCreateComputePipelines = create_pipeline
    monkeypatch.setattr(artifacts, "_procedure", lambda *_args, **_kwargs: "unused")
    monkeypatch.setattr(artifacts, "_records", records)
    backend = SimpleNamespace(
        _executable_capture_metadata={"pipelineExecutableInfo": True},
        vk=vk,
        device="fake",
        _device=None,
        _properties=SimpleNamespace(deviceID=1, vendorID=2, driverVersion=3, apiVersion=4),
        _lock=RLock(),
        _pipelines={},
        _pipeline=pipeline,
    )
    output = tmp_path / "export"
    metadata = artifacts.export_pipeline(backend, "exact source\n", output)
    assert (output / "source.comp").read_bytes() == shader.read_bytes()
    assert (output / "shader.spv").read_bytes() == spirv
    assert metadata["source_string_sha256"] != metadata["source"]["sha256"]
    assert metadata["compiler"]["arguments"] == args
    executable = metadata["executables"][0]
    assert executable["statistics"][0]["value"] == 71
    assert executable["representations"][0]["name"] == "A vendor-provided name, not inferred ISA"
    assert (
        output / executable["representations"][0]["file"]
    ).read_bytes() == b"text with terminator\x00"
    assert json.loads((output / "metadata.json").read_text()) == metadata
    assert runtime.subprocess.run is compiler_run
    assert vk.vkCreateComputePipelines is create_pipeline
