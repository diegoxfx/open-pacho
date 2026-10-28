"""Vulkan recording templates must never cache invocation values or live leases."""

import os
from collections import OrderedDict

import numpy as np
import pytest

import openpacho as pacho
from openpacho._array import ndarray
from openpacho._vulkan import VulkanBackend, _Buffer
from openpacho.exceptions import BackendUnavailableError


@pytest.fixture(scope="module")
def backend(request):
    try:
        value = VulkanBackend(
            os.environ.get("PACHO_VULKAN_DEVICE"),
            _descriptor_mode=getattr(request, "param", "auto"),
        )
    except BackendUnavailableError as exc:
        if os.environ.get("PACHO_REQUIRE_VULKAN", "").lower() in ("1", "true", "yes"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    yield value
    value.synchronize()


def array(backend, host):
    return ndarray._wrap(backend.upload(host), host.shape, host.dtype, backend)


def test_template_metadata_lru_and_scalar_alignment_without_driver():
    backend = object.__new__(VulkanBackend)
    backend._device = backend._instance = None
    backend._dispatch_templates = OrderedDict()
    backend._max_push_constants_size = 128
    backend._max_workgroup_count = (65535, 65535, 65535)
    backend.fp64 = backend.int64 = True
    dtypes = (np.dtype("float64"),)
    scalars = tuple(map(np.dtype, ("float32", "float64", "bool")))
    record = backend._dispatch_template("x0[i]=pc.s1;", 7, dtypes, scalars, False, "")
    assert [(offset, dtype.itemsize) for offset, dtype in record[1]] == [(0, 4), (8, 8), (16, 4)]
    assert record[2:] == (20, 1, 1)
    assert backend._dispatch_template("x0[i]=pc.s1;", 7, dtypes, scalars, False, "") is record
    for count in range(128):
        backend._dispatch_template("x0[i]=0;", count + 1, dtypes, (), False, "")
    assert len(backend._dispatch_templates) == 128
    assert record not in backend._dispatch_templates.values()
    with pytest.raises(NotImplementedError, match="push-constant"):
        backend._dispatch_template("", 1, dtypes, (np.dtype("float64"),) * 17, False, "")
    backend._max_workgroup_count = (1, 1, 1)
    with pytest.raises(ValueError, match="workgroup"):
        backend._dispatch_template("", 65, dtypes, (), False, "")


@pytest.mark.parametrize("backend", ["sets"], indirect=True)
def test_cached_descriptor_records_rebind_and_remain_distinct_in_flight(backend, monkeypatch):
    backend.release_cached_memory()
    host = np.arange(97, dtype=np.float32)
    a, b = array(backend, host), array(backend, -host)
    operation = pacho.compile(lambda x, alpha: x * alpha + 1)
    pacho.asnumpy(operation(a, 2.0))
    backend.synchronize()
    updates = []
    original = backend.vk.vkUpdateDescriptorSets

    def capture(device, count, writes, copy_count, copies):
        updates.append((writes, tuple(writes[i].pBufferInfo[0].buffer for i in range(count))))
        return original(device, count, writes, copy_count, copies)

    monkeypatch.setattr(backend.vk, "vkUpdateDescriptorSets", capture)
    first = operation(a, 3.0)
    second = operation(b, -5.0)
    assert updates[0][0] is not updates[1][0]
    np.testing.assert_array_equal(pacho.asnumpy(first), host * 3 + 1)
    np.testing.assert_array_equal(pacho.asnumpy(second), host * 5 + 1)
    previous_records = [entry[0] for entry in updates]
    third = operation(b, 7.0)
    assert any(updates[-1][0] is previous for previous in previous_records)
    assert updates[-1][1][0] == b._data.handle
    np.testing.assert_array_equal(pacho.asnumpy(third), -host * 7 + 1)
    # Prior public results remain independently owned after descriptor rebinding.
    np.testing.assert_array_equal(pacho.asnumpy(first), host * 3 + 1)


def test_warm_dispatch_uses_prebuilt_native_records(backend, monkeypatch):
    a = array(backend, np.arange(67, dtype=np.float32))
    pacho.asnumpy(a + a)
    backend.synchronize()

    def unexpected(*args, **kwargs):
        pytest.fail("warm dispatch rebuilt an invariant native record")

    for name in (
        "VkWriteDescriptorSet",
        "VkDescriptorBufferInfo",
        "VkCommandBufferBeginInfo",
        "VkMemoryBarrier",
        "VkSubmitInfo",
        "VkBufferCopy",
    ):
        monkeypatch.setattr(backend.vk, name, unexpected)
    np.testing.assert_array_equal(pacho.asnumpy(a + a), np.arange(67, dtype=np.float32) * 2)


@pytest.mark.parametrize("backend", ["auto", "sets"], indirect=True)
def test_descriptor_rebinding_updates_buffer_range(backend, monkeypatch):
    backend.release_cached_memory()
    sizes = []
    method = (
        "vkCmdPushDescriptorSetKHR" if backend._push_descriptor_limit else "vkUpdateDescriptorSets"
    )
    original = getattr(backend.vk, method)

    def capture(*args):
        count, writes = args[4:6] if backend._push_descriptor_limit else args[1:3]
        sizes.append(tuple(int(writes[i].pBufferInfo[0].range) for i in range(count)))
        return original(*args)

    monkeypatch.setattr(backend.vk, method, capture)
    for count in (31, 509, 3):
        host = np.arange(count, dtype=np.float32)
        a = array(backend, host)
        np.testing.assert_array_equal(pacho.asnumpy(a + a), host * 2)
        assert sizes[-1] == (host.nbytes,) * 3


def test_reused_capacity_keeps_logical_descriptor_and_transfer_extents(backend, monkeypatch):
    backend.release_cached_memory()
    leases = [backend.empty((129,), np.dtype("float32")) for _ in range(3)]
    leases.extend(_Buffer(backend, 516, host=True, readback=readback) for readback in (False, True))
    for lease in leases:
        lease.close()
    ranges, copies = [], []
    method = (
        "vkCmdPushDescriptorSetKHR" if backend._push_descriptor_limit else "vkUpdateDescriptorSets"
    )
    original_update = getattr(backend.vk, method)
    original_copy = backend.vk.vkCmdCopyBuffer

    def update(*args):
        count, writes = args[4:6] if backend._push_descriptor_limit else args[1:3]
        ranges.append(tuple(int(writes[i].pBufferInfo[0].range) for i in range(count)))
        return original_update(*args)

    def copy(command, source, destination, count, regions):
        copies.extend(int(regions[i].size) for i in range(count))
        return original_copy(command, source, destination, count, regions)

    monkeypatch.setattr(backend.vk, method, update)
    monkeypatch.setattr(backend.vk, "vkCmdCopyBuffer", copy)
    host = np.arange(127, dtype=np.float32)
    a = array(backend, host)
    first, second = a + a, a * 3
    for value in (a, first, second):
        assert value._data.size == host.nbytes
        assert value._data.capacity == 516
    assert len({value._data.handle for value in (a, first, second)}) == 3
    np.testing.assert_array_equal(pacho.asnumpy(first), host * 2)
    np.testing.assert_array_equal(pacho.asnumpy(second), host * 3)
    assert len(ranges) == 2 and all(size == host.nbytes for record in ranges for size in record)
    assert copies == [host.nbytes] * 3
