from __future__ import annotations

import sys
import types

import torch
from torch.utils import generate_methods_for_privateuse1_backend, rename_privateuse1_backend

from .runtime import (
    VULKAN_DEVICE_ID,
    normalize_vulkan_device,
    probe_ncnn_vulkan_runtime,
)


def _get_probe():
    return probe_ncnn_vulkan_runtime()


def _get_device_index(device: int | str | torch.device | None) -> int:
    if device is None:
        return 0
    if isinstance(device, int):
        return max(device, 0)
    if isinstance(device, torch.device):
        return 0 if device.index is None else int(device.index)

    normalized_device = normalize_vulkan_device(str(device))
    if normalized_device is None:
        return 0
    _, _, device_index = normalized_device.partition(":")
    return int(device_index or 0)


def _build_vulkan_device_module() -> types.ModuleType:
    module = types.ModuleType("torch.vulkan")

    def is_available() -> bool:
        return _get_probe().available

    def device_count() -> int:
        return len(_get_probe().devices)

    def current_device() -> int:
        return 0

    def get_device_name(device: int | str | torch.device | None = None) -> str:
        probe = _get_probe()
        if not probe.devices:
            return "NCNN Vulkan"
        device_index = _get_device_index(device)
        if 0 <= device_index < len(probe.devices):
            return probe.devices[device_index]
        return probe.devices[0]

    module.is_available = is_available
    module.device_count = device_count
    module.current_device = current_device
    module.get_device_name = get_device_name
    return module


def bootstrap_vulkan_privateuse1_backend() -> None:
    rename_privateuse1_backend("vulkan")

    if not hasattr(torch.Tensor, "is_vulkan"):
        generate_methods_for_privateuse1_backend()

    if not hasattr(torch, "vulkan"):
        module = _build_vulkan_device_module()
        sys.modules.setdefault("torch.vulkan", module)
        torch._register_device_module("vulkan", module)


def get_vulkan_privateuse1_device(
    device: str | torch.device | None = None,
) -> torch.device:
    bootstrap_vulkan_privateuse1_backend()
    normalized_device = normalize_vulkan_device(device or VULKAN_DEVICE_ID)
    return torch.device(normalized_device or VULKAN_DEVICE_ID)
