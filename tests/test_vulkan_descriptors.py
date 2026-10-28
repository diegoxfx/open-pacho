"""Push-descriptor snapshot semantics and the portable descriptor-set path."""

import os

import numpy as np
import pytest

import openpacho as pacho
from openpacho._vulkan import VulkanBackend
from openpacho.exceptions import BackendUnavailableError


@pytest.fixture(params=("auto", "sets"))
def descriptors(request):
    try:
        backend = VulkanBackend(
            os.environ.get("PACHO_VULKAN_DEVICE"), _descriptor_mode=request.param
        )
    except BackendUnavailableError as exc:
        if os.environ.get("PACHO_REQUIRE_VULKAN") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    yield backend
    backend.close()


def test_descriptor_snapshot_rebinding_batch_limit_and_live_results(descriptors, monkeypatch):
    backend = descriptors
    calls = []
    method = (
        "vkCmdPushDescriptorSetKHR" if backend._push_descriptor_limit else "vkUpdateDescriptorSets"
    )
    original = getattr(backend.vk, method)

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(backend.vk, method, counted)
    host = np.arange(33 * 17, dtype=np.float32).reshape(33, 17) / 17
    a = pacho.asarray(host, backend=backend)
    output = []
    with backend._batch_commands():
        for i in range(70):
            output.append((a + i) * (i + 1))
    assert len(calls) == 140
    backend.release_cached_memory()
    for i, value in enumerate(output):
        np.testing.assert_array_equal(pacho.asnumpy(value), (host + i) * (i + 1))


@pytest.mark.parametrize("descriptors", ["auto"], indirect=True)
def test_descriptor_limit_falls_back_for_larger_binding_count(descriptors, monkeypatch):
    backend = descriptors
    if not backend._push_descriptor_limit:
        pytest.skip("push-descriptor branch requires the optional extension")
    # Simulate a smaller advertised limit before preparing any pipelines.
    monkeypatch.setattr(backend, "_push_descriptor_limit", 2)
    a = pacho.asarray(np.arange(33, dtype=np.float32), backend=backend)
    first = a + 2  # input/output: push
    second = a + first  # three buffers: ordinary descriptor set
    third = second + 7  # push again, same shared host records rebound
    np.testing.assert_array_equal(pacho.asnumpy(third), np.arange(33, dtype=np.float32) * 2 + 9)
    assert {key[1] for key in backend._descriptor_layouts} == {False, True}


def test_invalid_descriptor_mode_fails_before_loading_vulkan():
    with pytest.raises(ValueError, match="descriptor mode"):
        VulkanBackend(_descriptor_mode="unknown")
