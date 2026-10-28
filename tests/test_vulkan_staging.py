"""Mappings follow completed allocations, never in-flight staging leases."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_vulkan import vk_backend  # noqa: F401
from test_vulkan_pool_eviction import pool, recycle

import openpacho as p


def test_mapping_reuse_eviction_and_failed_map():
    backend, freed = pool(limit=512)
    backend._staging_mode = "persistent"
    mapped = bytearray(512)
    calls = []
    backend.vk.vkMapMemory = lambda *a: calls.append(("map", a[3])) or mapped
    backend.vk.vkUnmapMemory = lambda *a: calls.append(("unmap", a[1]))
    staging = SimpleNamespace(host=True, capacity=512, memory="mem-old")
    try:
        for _ in range(2):
            with backend._map_staging(staging) as value:
                assert value is mapped
        assert calls == [("map", 512)]
        recycle(backend, (512, True, False), "old")
        recycle(backend, (512, True, False), "new")
        assert calls[-1] == ("unmap", "mem-old")
        assert backend._mapped_memory == {}
        assert freed == [("buffer", "old"), ("memory", "mem-old")]

        def fail(*args):
            raise RuntimeError("map failed")

        backend.vk.vkMapMemory = fail
        with pytest.raises(RuntimeError, match="map failed"), backend._map_staging(staging):
            pass
        assert backend._mapped_memory == {}
    finally:
        backend._device = None


@pytest.mark.vulkan
@pytest.mark.parametrize("mode", ["readback", "persistent", "transient"])
def test_gpu_staging_reuse_pending_uploads_and_release(vk_backend, mode):  # noqa: F811 - imported fixture
    backend = vk_backend
    backend.release_cached_memory()
    previous_mode = backend._staging_mode
    backend._staging_mode = mode
    counters = {"map": 0, "unmap": 0}
    originals = backend.vk.vkMapMemory, backend.vk.vkUnmapMemory

    def counted(index, name):
        def call(*args):
            counters[name] += 1
            return originals[index](*args)

        return call

    backend.vk.vkMapMemory = counted(0, "map")
    backend.vk.vkUnmapMemory = counted(1, "unmap")
    try:
        for _ in range(2):
            # Odd shapes share larger completed buffers. Multiple uploads keep
            # separate leases until synchronization; mutation of host is safe.
            arrays = []
            expected = []
            for n in (129, 127, 128):
                host = np.arange(n, dtype=np.float32)
                expected.append(host.copy())
                arrays.append(p.asarray(host, backend=backend))
                host[:] = -99
            for a, ref in zip(arrays, expected):
                np.testing.assert_array_equal(p.asnumpy(a), ref)
            del a, arrays
            backend.synchronize()
            if _ == 0:
                warmed = counters["map"]
        assert counters["map"] == warmed + {"persistent": 0, "readback": 3, "transient": 6}[mode]
        assert counters["unmap"] == {"persistent": 0, "readback": 6, "transient": 12}[mode]
        backend.release_cached_memory()
        assert counters["unmap"] == counters["map"]
        assert backend._mapped_memory == {}
    finally:
        backend.release_cached_memory()
        backend.vk.vkMapMemory, backend.vk.vkUnmapMemory = originals
        backend._staging_mode = previous_mode
