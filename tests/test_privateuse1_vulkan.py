from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import torch


def test_importing_vulkan_extension_bootstraps_privateuse1(monkeypatch) -> None:
    from lada.extensions.vulkan import privateuse1

    calls: list[str] = []

    monkeypatch.setattr(
        privateuse1,
        "bootstrap_vulkan_privateuse1_backend",
        lambda: calls.append("bootstrapped"),
    )
    sys.modules.pop("lada.extensions.vulkan", None)

    importlib.import_module("lada.extensions.vulkan")

    assert calls == ["bootstrapped"]


def test_privateuse1_bootstrap_exposes_vulkan_torch_device(monkeypatch) -> None:
    from lada.extensions.vulkan import privateuse1

    monkeypatch.setattr(
        privateuse1,
        "probe_ncnn_vulkan_runtime",
        lambda: SimpleNamespace(available=True, devices=("NCNN Vulkan GPU",)),
    )

    privateuse1.bootstrap_vulkan_privateuse1_backend()
    device = privateuse1.get_vulkan_privateuse1_device()

    assert str(device) == "vulkan:0"
    assert device.type == "vulkan"
    assert torch.vulkan.is_available() is True
    assert torch.vulkan.device_count() == 1
    assert torch.vulkan.get_device_name(0) == "NCNN Vulkan GPU"
