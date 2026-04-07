from __future__ import annotations

import ctypes
import ctypes.util
import sys
from dataclasses import dataclass

import torch

from lada.utils.os_utils import gpu_has_fp16_acceleration, has_mps


@dataclass(frozen=True)
class ComputeTarget:
    id: str
    description: str
    runtime: str
    available: bool
    torch_device: str | None
    notes: str = ""
    experimental: bool = False


class UnsupportedComputeTargetError(RuntimeError):
    """Raised when a requested compute target cannot back the current pipeline."""


def _has_vulkan_loader() -> bool:
    probe_name = "vulkan-1" if sys.platform == "win32" else "vulkan"
    if ctypes.util.find_library(probe_name) is not None:
        return True

    load_name = "vulkan-1.dll" if sys.platform == "win32" else "libvulkan.so.1"
    loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
    try:
        loader(load_name)
        return True
    except OSError:
        return False


def _has_ncnn_vulkan_runtime() -> bool:
    try:
        from lada.models.basicvsrpp.ncnn_vulkan import (
            import_ncnn_module,
            ncnn_has_lada_basicvsrpp_clip_runner,
            ncnn_has_lada_custom_layer,
            ncnn_has_lada_gridsample_layer,
            ncnn_has_lada_vulkan_net_runner,
            ncnn_has_lada_yolo_attention_layer,
            ncnn_has_lada_yolo_seg_postprocess_vulkan_layer,
        )

        ncnn = import_ncnn_module()
        return (
            ncnn.get_gpu_count() > 0
            and ncnn_has_lada_custom_layer(ncnn)
            and ncnn_has_lada_gridsample_layer(ncnn)
            and ncnn_has_lada_yolo_attention_layer(ncnn)
            and ncnn_has_lada_yolo_seg_postprocess_vulkan_layer(ncnn)
            and ncnn_has_lada_vulkan_net_runner(ncnn)
            and ncnn_has_lada_basicvsrpp_clip_runner(ncnn)
        )
    except Exception:
        return False


def normalize_compute_target_id(target_id: str | None) -> str:
    if not target_id:
        return get_default_compute_target()
    lowered = target_id.lower()
    if lowered == "vulkan":
        return "vulkan:0"
    return lowered


def get_default_compute_target() -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    if has_mps():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.device_count() > 0:
        return "xpu:0"
    return "cpu"


def get_compute_targets(include_experimental: bool = False) -> list[ComputeTarget]:
    targets = [
        ComputeTarget(
            id="cpu",
            description="CPU",
            runtime="torch",
            available=True,
            torch_device="cpu",
        )
    ]

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            targets.append(
                ComputeTarget(
                    id=f"cuda:{index}",
                    description=torch.cuda.get_device_properties(index).name,
                    runtime="torch",
                    available=True,
                    torch_device=f"cuda:{index}",
                )
            )

    if has_mps():
        targets.append(
            ComputeTarget(
                id="mps",
                description="Apple MPS (Metal)",
                runtime="torch",
                available=True,
                torch_device="mps",
            )
        )

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        for index in range(torch.xpu.device_count()):
            targets.append(
                ComputeTarget(
                    id=f"xpu:{index}",
                    description=torch.xpu.get_device_name(index),
                    runtime="torch",
                    available=True,
                    torch_device=f"xpu:{index}",
                )
            )

    if include_experimental:
        has_loader = _has_vulkan_loader()
        has_ncnn_runtime = _has_ncnn_vulkan_runtime()
        loader_note = "" if has_loader else " Vulkan loader not detected on this machine."
        ncnn_note = (
            ""
            if has_ncnn_runtime
            else (
                " Build the local ncnn Vulkan runtime with Lada custom operators "
                "(for example via 'scripts/build_ncnn_vulkan_runtime.ps1')."
            )
        )
        targets.append(
            ComputeTarget(
                id="vulkan:0",
                description="Vulkan",
                runtime="vulkan",
                available=has_loader and has_ncnn_runtime,
                torch_device=None,
                notes=(
                    "Vulkan backend backed by the local ncnn runtime and Lada custom operators. "
                    "Mosaic detection runs through fused YOLO segmentation postprocess on GPU, and "
                    "BasicVSR++ restoration runs through the modular 5-frame / recurrent Vulkan graph. "
                    f"DeepMosaics is not supported.{loader_note}{ncnn_note}"
                ),
                experimental=True,
            )
        )

    return targets


def get_compute_target(target_id: str, include_experimental: bool = True) -> ComputeTarget | None:
    normalized_target_id = normalize_compute_target_id(target_id)
    for target in get_compute_targets(include_experimental=include_experimental):
        if target.id == normalized_target_id:
            return target
    return None


def is_compute_target_available(target_id: str) -> bool:
    target = get_compute_target(target_id, include_experimental=True)
    return target is not None and target.available


def describe_compute_target_issue(target_id: str) -> str | None:
    normalized_target_id = normalize_compute_target_id(target_id)
    target = get_compute_target(normalized_target_id, include_experimental=True)
    if target is None:
        return f"Unknown compute target '{target_id}'. Use --list-devices to inspect supported targets."
    if target.available:
        return None
    if target.notes:
        return target.notes
    return f"Compute target '{normalized_target_id}' is not available."


def resolve_torch_device(target_id: str) -> torch.device:
    normalized_target_id = normalize_compute_target_id(target_id)
    target = get_compute_target(normalized_target_id, include_experimental=True)
    if target is None:
        raise UnsupportedComputeTargetError(
            f"Unknown compute target '{target_id}'. Use --list-devices to inspect supported targets."
        )
    if not target.available:
        raise UnsupportedComputeTargetError(
            target.notes or f"Compute target '{normalized_target_id}' is not available."
        )
    if target.torch_device is None:
        raise UnsupportedComputeTargetError(
            f"Compute target '{normalized_target_id}' does not map to a torch.device."
        )
    return torch.device(target.torch_device)


def default_fp16_enabled_for_compute_target(target_id: str | None) -> bool:
    """Return the default fp16 toggle for one resolved compute target."""
    normalized_target_id = normalize_compute_target_id(target_id)
    target = get_compute_target(normalized_target_id, include_experimental=True)
    if target is None:
        return gpu_has_fp16_acceleration()
    if target.runtime == "vulkan":
        # The local NCNN Vulkan path is designed around fp16 artifacts; callers can still opt out
        # with `--no-fp16` when they need conservative compatibility.
        return True
    if target.torch_device is None:
        return False
    return gpu_has_fp16_acceleration(torch.device(target.torch_device))
