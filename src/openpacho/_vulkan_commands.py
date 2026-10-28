"""Private typed Vulkan command boundary, resolved once per logical device.

The backend passes owned CFFI handles, native arrays/pointers and ffi.NULL.
No generic Python argument conversion, process-global patch or new binary is
needed. Device-specific entry points must never be shared between VkDevices.
Resource ownership and synchronization remain the backend's responsibility.
"""

VOID_COMMANDS = (
    "vkUpdateDescriptorSets",
    "vkCmdPipelineBarrier",
    "vkCmdBindPipeline",
    "vkCmdBindDescriptorSets",
    "vkCmdPushConstants",
    "vkCmdDispatch",
    "vkCmdCopyBuffer",
    "vkUnmapMemory",
)
RESULT_COMMANDS = (
    "vkResetCommandBuffer",
    "vkBeginCommandBuffer",
    "vkEndCommandBuffer",
    "vkQueueSubmit",
    "vkQueueWaitIdle",
    "vkWaitForFences",
    "vkResetFences",
)


def _checked(function, name, exceptions):
    def call(*args):
        result = function(*args)
        if result != 0:
            exception = exceptions.get(result)
            if exception is None:
                raise RuntimeError(f"{name} failed with VkResult {result}")
            raise exception

    return call


class DeviceCommands:
    """Instance-local command table; other Vulkan attributes are cached on demand.

    ``wrapped`` preserves the generated binding as a diagnostic control.
    ``direct`` requires the typed ABI. ``auto`` uses the same generated binding
    if entry-point resolution is unavailable; execution errors never fall back.
    """

    def __init__(self, vk, device, mode="auto"):
        if mode not in ("auto", "wrapped", "direct"):
            raise ValueError("command mode must be auto, wrapped or direct")
        self._vk = vk
        self.mode = "wrapped"
        self.reason = None
        if mode == "wrapped":
            return
        try:
            functions = {}
            for name in VOID_COMMANDS + RESULT_COMMANDS:
                pointer = vk.lib.vkGetDeviceProcAddr(device, name.encode("ascii"))
                if pointer == vk.ffi.NULL:
                    raise RuntimeError(f"Missing Vulkan device entry point: {name}")
                functions[name] = vk.ffi.cast("PFN_" + name, pointer)
        except (AttributeError, TypeError, RuntimeError) as exc:
            if mode == "direct":
                raise
            self.reason = str(exc)
            return
        for name, function in functions.items():
            self.__dict__[name] = (
                _checked(function, name, vk.exception_codes)
                if name in RESULT_COMMANDS
                else function
            )
        self.mode = "direct"

    def __getattr__(self, name):
        value = getattr(self._vk, name)
        setattr(self, name, value)
        return value
