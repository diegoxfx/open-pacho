"""Completed stale size classes cannot force permanent native-allocation churn."""

import gc
import weakref
from collections import OrderedDict
from threading import RLock
from types import SimpleNamespace

import pytest

from openpacho._vulkan import VulkanBackend, _Buffer


def pool(limit=128 * 1024**2):
    backend = object.__new__(VulkanBackend)
    freed = []
    backend.vk = SimpleNamespace(
        vkDestroyBuffer=lambda device, handle, _: freed.append(("buffer", handle)),
        vkFreeMemory=lambda device, handle, _: freed.append(("memory", handle)),
    )
    backend._lock = RLock()
    backend._device = "device"
    backend._instance = None
    backend._closing = False
    backend._buffer_cache = OrderedDict()
    backend._mapped_memory = {}
    backend._cached_bytes = backend._pool_items = 0
    backend._pool_limit = limit
    return backend, freed


def recycle(backend, key, handle, allocation=None):
    size, host, readback = key
    backend._recycle_buffer(size, host, handle, f"mem-{handle}", allocation or size, readback)


def test_release_detaches_cache_before_reentrant_buffer_return():
    backend, freed = pool()
    backend.synchronize = lambda: None
    backend._free_descriptor_pools, backend._free_commands = {}, []
    destroy = backend.vk.vkDestroyBuffer

    def reentrant_destroy(device, handle, allocator):
        if handle == "old":
            recycle(backend, (600, False, False), "returned")
        destroy(device, handle, allocator)

    backend.vk.vkDestroyBuffer = reentrant_destroy
    try:
        recycle(backend, (512, False, False), "old")
        backend.release_cached_memory()
        assert freed == [("buffer", "old"), ("memory", "mem-old")]
        assert backend._pool_items == 1 and backend._cached_bytes == 600
        backend.release_cached_memory()
        assert backend._pool_items == backend._cached_bytes == 0
        assert freed[-2:] == [("buffer", "returned"), ("memory", "mem-returned")]
    finally:
        backend._device = None


@pytest.mark.parametrize("evict_selected", [False, True])
def test_size_scan_tolerates_reentrant_recycling_and_eviction(evict_selected):
    backend, _ = pool(limit=1024 if evict_selected else 4096)
    triggered = False

    class ReentrantKey(tuple):
        def __getitem__(self, index):
            nonlocal triggered
            if not triggered:
                triggered = True
                # Model a _Buffer finalizer triggered by an allocation during
                # candidate selection, while the same thread holds the RLock.
                recycle(backend, (600, False, False), "returned")
            return super().__getitem__(index)

    try:
        recycle(backend, (512, False, False), "original")
        entries = backend._buffer_cache.pop((512, False, False))
        backend._buffer_cache[ReentrantKey((512, False, False))] = entries
        result = backend._take_buffer(500, False)
        assert triggered
        assert result[0] == ("returned" if evict_selected else "original")
        assert result[3] == (600 if evict_selected else 512)
        assert backend._pool_items == (0 if evict_selected else 1)
        assert backend._cached_bytes == (0 if evict_selected else 600)
    finally:
        backend._device = None


def test_new_size_becomes_reusable_when_old_size_fills_item_budget():
    backend, freed = pool()
    try:
        for i in range(128):
            recycle(backend, (512, False, False), str(i))
        recycle(backend, (516, False, False), "new")
        assert backend._pool_items == 128
        assert backend._cached_bytes == 127 * 512 + 516
        assert len(freed) == 2
        for _ in range(5):
            handles = backend._take_buffer(516, False)
            assert handles == ("new", "mem-new", 516, 516)
            recycle(backend, (516, False, False), "new")
        assert len(freed) == 2  # Warm calls do not allocate/free again.
    finally:
        backend._device = None


def test_byte_eviction_uses_allocation_size_and_keeps_recent_class():
    backend, freed = pool(limit=1024)
    try:
        recycle(backend, (64, False, False), "old", 512)
        recycle(backend, (32, True, False), "upload", 256)
        recycle(backend, (32, True, True), "readback", 256)
        recycle(backend, (128, False, False), "new", 512)
        assert freed == [("buffer", "old"), ("memory", "mem-old")]
        assert backend._cached_bytes == 1024 and backend._pool_items == 3
        assert backend._take_buffer(32, True, False)[0] == "upload"
        assert backend._take_buffer(32, True, True)[0] == "readback"
        # Taking a live lease removes it from the cache and its eviction budget.
        assert backend._cached_bytes == 512 and backend._pool_items == 1
    finally:
        backend._device = None


def test_oversized_or_closing_lease_does_not_evict_useful_cache():
    backend, freed = pool(limit=1024)
    try:
        recycle(backend, (64, False, False), "keep", 512)
        recycle(backend, (2048, True, False), "oversized")
        backend._closing = True
        recycle(backend, (64, False, False), "closing", 512)
        assert backend._cached_bytes == 512 and backend._pool_items == 1
        assert len(freed) == 4
        assert backend._take_buffer(64, False)[0] == "keep"
    finally:
        backend._device = None


def test_take_refreshes_remaining_class_without_exposing_live_lease_to_eviction():
    backend, freed = pool(limit=1536)
    try:
        recycle(backend, (64, False, False), "retained", 512)
        recycle(backend, (64, False, False), "live", 512)
        recycle(backend, (32, False, False), "older", 512)
        live = backend._take_buffer(64, False)
        recycle(backend, (128, False, False), "incoming", 1024)
        assert live == ("live", "mem-live", 512, 64)
        assert freed == [("buffer", "older"), ("memory", "mem-older")]
        assert backend._take_buffer(64, False)[0] == "retained"
        assert backend._cached_bytes == 1024 and backend._pool_items == 1
    finally:
        backend._device = None


def test_exact_capacity_precedes_other_eligible_sizes_and_best_fit_is_smallest():
    backend, freed = pool()
    try:
        recycle(backend, (516, False, False), "exact", 1024)
        recycle(backend, (520, False, False), "larger", 2048)
        recycle(backend, (512, False, False), "too-small", 4096)
        recycle(backend, (800, False, False), "largest", 1024)
        assert backend._take_buffer(516, False) == ("exact", "mem-exact", 1024, 516)
        assert backend._take_buffer(513, False) == ("larger", "mem-larger", 2048, 520)
        assert backend._take_buffer(513, False) == ("largest", "mem-largest", 1024, 800)
        # The driver's allocation alignment cannot enlarge the created VkBuffer.
        assert backend._take_buffer(513, False) is None
        assert backend._cached_bytes == 4096 and backend._pool_items == 1
        assert not freed
    finally:
        backend._device = None


@pytest.mark.parametrize("requested,capacity", [(4, 512), (256, 512), (300, 600)])
def test_best_fit_bound_includes_limit_but_rejects_larger_capacity(requested, capacity):
    backend, freed = pool()
    try:
        recycle(backend, (capacity + 1, False, False), "too-large")
        assert backend._take_buffer(requested, False) is None
        recycle(backend, (capacity, False, False), "bounded")
        assert backend._take_buffer(requested, False)[3] == capacity
        assert backend._take_buffer(requested, False) is None
        assert not freed
    finally:
        backend._device = None


def test_best_fit_preserves_host_and_readback_memory_classes():
    backend, freed = pool()
    try:
        recycle(backend, (520, False, False), "device")
        recycle(backend, (516, True, True), "readback")
        recycle(backend, (524, True, False), "upload")
        assert backend._take_buffer(512, True, False)[0] == "upload"
        assert backend._take_buffer(512, True, False) is None
        assert backend._take_buffer(512, False, False)[0] == "device"
        assert backend._take_buffer(512, False, False) is None
        assert backend._take_buffer(512, True, True)[0] == "readback"
        assert not freed
    finally:
        backend._device = None


def test_nearby_odd_shapes_keep_logical_size_and_recycle_original_capacity():
    backend, freed = pool()
    capacity = 4097 * 129 * 4
    allocation = capacity + 256
    try:
        recycle(backend, (capacity, False, False), "matrix", allocation)
        for size in (4095 * 127 * 4, 4096 * 128 * 4, capacity):
            lease = _Buffer(backend, size)
            assert lease.size == size  # Descriptor range remains the logical extent.
            assert lease.capacity == capacity
            assert lease.allocation_size == allocation
            assert lease.handle == "matrix"
            assert backend._pool_items == backend._cached_bytes == 0
            lease.close()
            assert list(backend._buffer_cache) == [(capacity, False, False)]
            assert backend._pool_items == 1 and backend._cached_bytes == allocation
        assert not freed  # A fake driver has no allocator: every lease is a cache hit.
    finally:
        backend._device = None


def test_new_buffer_capacity_comes_from_creation_size_not_memory_requirements():
    backend, freed = pool()
    created = []
    backend._memory_type_flags = (1,)
    backend.vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 1
    backend.vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 2
    backend.vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT = 4
    backend.vk.VK_SHARING_MODE_EXCLUSIVE = 0
    backend.vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1
    backend.vk.VkBufferCreateInfo = lambda **fields: SimpleNamespace(**fields)
    backend.vk.vkCreateBuffer = lambda device, info, _: created.append(info.size) or "new"
    backend.vk.vkGetBufferMemoryRequirements = lambda device, handle: SimpleNamespace(
        size=4096, memoryTypeBits=1
    )
    backend.vk.VkMemoryAllocateInfo = lambda **fields: SimpleNamespace(**fields)
    backend.vk.vkAllocateMemory = lambda device, info, _: "mem-new"
    backend.vk.vkBindBufferMemory = lambda *args: None
    try:
        lease = _Buffer(backend, 516)
        assert created == [516]
        assert lease.size == lease.capacity == 516
        assert lease.allocation_size == 4096
        lease.close()
        assert backend._take_buffer(1024, False) is None
        assert backend._take_buffer(513, False) == ("new", "mem-new", 4096, 516)
        assert not freed
    finally:
        backend._device = None


def test_recorded_and_live_references_hold_capacity_outside_reusable_cache():
    backend, freed = pool()
    backend._recording_command = "command"
    backend._recording_references = []
    backend._recording_pools = []
    backend._queued_operations = 0
    backend._batch_depth = 1
    try:
        recycle(backend, (516, False, False), "held", 1024)
        live = _Buffer(backend, 512)
        reference = weakref.ref(live)
        backend._submit("command", [live])
        del live
        gc.collect()
        assert reference() is not None
        assert backend._take_buffer(512, False) is None
        retained = reference()
        backend._recording_references.clear()
        gc.collect()
        assert backend._take_buffer(512, False) is None
        del retained
        gc.collect()
        assert reference() is None
        assert backend._take_buffer(512, False) == ("held", "mem-held", 1024, 516)
        assert not freed
    finally:
        backend._device = None
