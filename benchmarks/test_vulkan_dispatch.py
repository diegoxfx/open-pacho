"""Diagnostic clocks, query lifetime and failures without a GPU dependency."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.vulkan_dispatch import Diagnostic, solver_validation, timestamp_seconds
from openpacho._vulkan_commands import RESULT_COMMANDS, VOID_COMMANDS


def test_timestamp_wrap_uses_valid_bits_and_period():
    assert timestamp_seconds(250, 5, 8, 2) == pytest.approx(22e-9)
    assert timestamp_seconds((1 << 64) - 3, 5, 64, 1) == pytest.approx(8e-9)


def test_solver_validation_checks_true_residual_independently():
    a, b = np.eye(3), np.ones(3)
    assert solver_validation(a, b, b, 1e-4)["residual_float64"] == 0
    with pytest.raises(AssertionError, match="residual"):
        solver_validation(a, b, np.zeros(3), 1e-4)
    with pytest.raises(AssertionError, match="residual"):
        solver_validation(a, b, np.full(3, np.nan), 1e-4)


@pytest.mark.parametrize("bits,period", [(0, 1), (65, 1), (8, 0)])
def test_timestamp_rejects_unavailable_capability(bits, period):
    with pytest.raises(ValueError):
        timestamp_seconds(0, 1, bits, period)


def fake_backend(timestamps=True):
    events = []
    vk = SimpleNamespace()
    for name in (
        VOID_COMMANDS
        + RESULT_COMMANDS
        + (
            "vkMapMemory",
            "vkAllocateMemory",
            "vkFreeMemory",
            "vkCreateBuffer",
            "vkDestroyBuffer",
            "vkBindBufferMemory",
            "vkGetBufferMemoryRequirements",
            "vkAllocateDescriptorSets",
            "vkCreateDescriptorPool",
            "vkDestroyDescriptorPool",
            "vkCmdResetQueryPool",
            "vkCmdWriteTimestamp",
        )
    ):
        setattr(vk, name, lambda *args: None)
    vk.ffi = SimpleNamespace(new=lambda _type, count: [0] * count)
    vk.VK_QUERY_TYPE_TIMESTAMP = 0
    vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 1
    vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 2
    vk.VK_QUERY_RESULT_64_BIT = 1
    vk.VK_QUERY_RESULT_WAIT_BIT = 2
    vk.VkQueryPoolCreateInfo = lambda **kwargs: kwargs
    vk.vkCreateQueryPool = lambda *args: "pool"
    vk.vkDestroyQueryPool = lambda *args: events.append("destroy")

    def results(device, pool, first, count, size, data, stride, flags):
        data[:] = [250, 5] * (count // 2)

    vk.vkGetQueryPoolResults = results
    backend = SimpleNamespace(
        vk=vk,
        _device="device",
        _cached_bytes=128,
        _timestamp_valid_bits=8 if timestamps else 0,
        _timestamp_period=2,
    )

    def synchronize():
        events.append("sync")
        vk.vkQueueWaitIdle("queue")

    backend.synchronize = synchronize
    return backend, events


@pytest.mark.parametrize("timestamps", [False, True])
def test_diagnostic_counts_copies_and_restores_functions(timestamps):
    backend, events = fake_backend(timestamps)
    copy = backend.vk.vkCmdCopyBuffer

    def operation():
        backend.vk.vkCmdCopyBuffer("command", "a", "b", 1, [SimpleNamespace(size=516)])
        return 7

    value, report = Diagnostic(backend).run(operation)
    assert value == 7
    assert report["copy_bytes"] == 516
    assert report["command_counts"]["vkCmdCopyBuffer"] == 1
    assert report["gpu_timestamps_supported"] == timestamps
    assert backend.vk.vkCmdCopyBuffer is copy
    if timestamps:
        assert report["gpu_intervals"][0]["seconds"] == pytest.approx(22e-9)
        assert events[-2:] == ["sync", "destroy"]
    else:
        assert report["gpu_intervals"] == [] and "destroy" not in events


def test_timestamp_capacity_failure_restores_hooks_and_waits_before_destroy():
    backend, events = fake_backend()
    original = backend.vk.vkCmdDispatch

    def operation():
        for _ in range(2):
            backend.vk.vkCmdDispatch("command", 1, 1, 1)

    with pytest.raises(RuntimeError, match="capacity"):
        Diagnostic(backend, max_commands=1).run(operation)
    assert backend.vk.vkCmdDispatch is original
    assert events[-2:] == ["sync", "destroy"]
