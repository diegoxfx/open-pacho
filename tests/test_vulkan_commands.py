"""Typed Vulkan commands preserve device ownership, errors and array semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

import openpacho as p
from openpacho._vulkan_commands import RESULT_COMMANDS, VOID_COMMANDS, DeviceCommands


def fake_vk(missing=None, result=0):
    calls = []

    def resolve(device, name):
        name = name.decode()
        if name == missing:
            return None

        def function(*args):
            calls.append((device, name, args))
            return result if name in RESULT_COMMANDS else None

        return function

    class DeviceLost(Exception):
        pass

    vk = SimpleNamespace(
        ffi=SimpleNamespace(NULL=None, cast=lambda signature, pointer: pointer),
        lib=SimpleNamespace(vkGetDeviceProcAddr=resolve),
        exception_codes={-4: DeviceLost},
    )
    for name in VOID_COMMANDS + RESULT_COMMANDS:
        setattr(vk, name, lambda *args: "wrapped")
    return vk, calls, DeviceLost


def test_command_tables_are_device_local_and_do_not_patch_binding():
    vk, calls, _ = fake_vk()
    first, second = DeviceCommands(vk, "first"), DeviceCommands(vk, "second")
    assert first.mode == second.mode == "direct"
    first.vkCmdDispatch("command1", 1, 2, 3)
    second.vkCmdDispatch("command2", 2, 3, 4)
    assert calls == [
        ("first", "vkCmdDispatch", ("command1", 1, 2, 3)),
        ("second", "vkCmdDispatch", ("command2", 2, 3, 4)),
    ]
    assert vk.vkCmdDispatch() == "wrapped"


def test_missing_entrypoint_uses_whole_wrapped_table_or_fails_when_required():
    vk, calls, _ = fake_vk(missing=RESULT_COMMANDS[-1])
    table = DeviceCommands(vk, "device")
    assert table.mode == "wrapped" and RESULT_COMMANDS[-1] in table.reason
    assert table.vkCmdDispatch() == "wrapped"
    assert calls == []
    with pytest.raises(RuntimeError, match="Missing Vulkan"):
        DeviceCommands(vk, "device", "direct")
    with pytest.raises(ValueError, match="command mode"):
        DeviceCommands(vk, "device", "invalid")


@pytest.mark.parametrize("name", RESULT_COMMANDS)
def test_vulkan_errors_propagate_without_runtime_fallback(name):
    vk, calls, error = fake_vk(result=-4)
    table = DeviceCommands(vk, "device")
    with pytest.raises(error):
        getattr(table, name)("handle")
    assert table.mode == "direct"
    assert calls == [("device", name, ("handle",))]


def test_unknown_result_is_reported_and_success_returns_none():
    vk, _, _ = fake_vk(result=-123456)
    with pytest.raises(RuntimeError, match="vkQueueSubmit.*-123456"):
        DeviceCommands(vk, "device").vkQueueSubmit()
    vk, _, _ = fake_vk()
    assert DeviceCommands(vk, "device").vkQueueSubmit() is None


@pytest.mark.vulkan
@pytest.mark.parametrize("mode", ["wrapped", "direct"])
def test_typed_and_wrapped_execution_keep_inflight_outputs_and_copy_extents(mode):
    import os

    from openpacho._vulkan import VulkanBackend

    try:
        backend = VulkanBackend(os.environ.get("PACHO_VULKAN_DEVICE"), _command_mode=mode)
    except p.BackendUnavailableError as exc:
        if os.environ.get("PACHO_REQUIRE_VULKAN") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    assert backend.vk.mode == mode
    try:

        @p.compile
        def step(a, b, scale):
            c = a @ b
            return c * scale + 1, p.sum(c * c, axis=1)

        rng = np.random.default_rng(941)
        saved = []
        # Cross the 64-operation drain with live outputs and non-tile dimensions.
        for index in range(24):
            host = rng.integers(-3, 4, (17, 33)).astype(np.float32)
            a = p.asarray(host, backend=backend)
            b = p.asarray(host.T, backend=backend)
            actual = step(a, b, float(index + 1))
            expected = host @ host.T
            saved.append((actual, expected * (index + 1) + 1, (expected * expected).sum(axis=1)))
        backend.release_cached_memory()
        for (out, norm), expected, square in saved:
            np.testing.assert_array_equal(p.asnumpy(out), expected)
            np.testing.assert_array_equal(p.asnumpy(norm), square)
        np.testing.assert_array_equal(
            p.asnumpy(p.asarray([True, False], backend=backend)), [True, False]
        )
    finally:
        backend.close()
