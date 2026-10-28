"""Completion markers may release leases only after a successful wait."""

import os
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

import openpacho as p
from openpacho._vulkan import VulkanBackend


def fake_completion(failure=None):
    b = object.__new__(VulkanBackend)
    b._device, b._instance, b._queue = None, None, "queue"
    b._lock = RLock()
    b._completion_mode, b._completion_fence = "fence", None
    b._completion_failed = False
    b._recording_command = None
    b._recorded_commands = []
    b._queued_operations = 1
    b._free_commands = []
    b._free_descriptor_pools = {}
    b._pending = [("command", [], [object()])]
    calls = []

    def command(name, result=None):
        def call(*args):
            calls.append((name, args))
            if failure == name:
                raise RuntimeError("injected completion failure")
            return result

        return call

    b.vk = SimpleNamespace(
        ffi=SimpleNamespace(NULL=None, new=lambda t, count: [None] * count),
        VK_TRUE=1,
        VkFenceCreateInfo=lambda: "info",
        vkCreateFence=command("create", "fence"),
        vkQueueSubmit=command("submit"),
        vkWaitForFences=command("wait"),
        vkResetFences=command("reset"),
        vkQueueWaitIdle=command("idle"),
    )
    return b, calls


def test_reuses_one_marker_and_elides_only_empty_queue_waits():
    b, calls = fake_completion()
    b.synchronize()
    assert [name for name, _ in calls] == ["create", "submit", "wait", "reset"]
    assert calls[1][1] == ("queue", 0, None, "fence")
    assert calls[2][1][-1] == 2**64 - 1
    assert not b._pending and b._free_commands == ["command"]
    b.synchronize()
    assert len(calls) == 4
    b._pending = [("second", [], [object()])]
    b.synchronize()
    assert [name for name, _ in calls[4:]] == ["submit", "wait", "reset"]


@pytest.mark.parametrize("failure", ["submit", "wait", "reset"])
def test_failed_completion_retains_resources_and_prevents_resubmission(failure):
    b, calls = fake_completion(failure)
    pending = b._pending
    with pytest.raises(RuntimeError, match="injected"):
        b.synchronize()
    assert b._pending is pending and len(b._pending) == 1
    assert not b._free_commands
    assert b._completion_failed
    count = len(calls)
    with pytest.raises(RuntimeError, match="cannot submit"):
        b.synchronize()
    assert len(calls) == count


@pytest.mark.vulkan
@pytest.mark.parametrize("mode", ["queue", "fence"])
def test_real_completion_preserves_pending_chains_batches_and_reads(mode):
    try:
        b = VulkanBackend(os.environ.get("PACHO_VULKAN_DEVICE"), _completion_mode=mode)
    except p.BackendUnavailableError as exc:
        if os.environ.get("PACHO_REQUIRE_VULKAN") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    try:

        @p.compile
        def update(a):
            x = a + 1
            return x, p.sum(x * x)

        a = p.asarray(np.arange(129, dtype=np.float32), backend=b)
        saved = []
        for i in range(80):
            a, total = update(a)
            if i % 17 == 0:
                saved.append((a, total, i + 1))
        b.release_cached_memory()
        for values, total, count in saved:
            expected = np.arange(129, dtype=np.float32) + count
            np.testing.assert_array_equal(p.asnumpy(values), expected)
            np.testing.assert_allclose(total.item(), (expected * expected).sum(), rtol=1e-6)
        del a, total, values, saved
        assert not b._pending
    finally:
        b.close()
