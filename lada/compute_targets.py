from __future__ import annotations
from dataclasses import dataclass

import torch

from lada.extensions.runtime_registry import (
    get_runtime_extension,
    iter_runtime_extensions,
)
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

    for extension in iter_runtime_extensions():
        get_targets = extension.get_compute_targets
        if get_targets is None:
            continue
        for target in get_targets():
            if include_experimental or not getattr(target, "experimental", False):
                targets.append(target)

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
    extension = get_runtime_extension(target.runtime)
    if extension is not None and extension.default_fp16_enabled is not None:
        return bool(extension.default_fp16_enabled(normalized_target_id))
    if target.torch_device is None:
        return False
    return gpu_has_fp16_acceleration(torch.device(target.torch_device))


def configure_compute_target_device_info(
    target_id: str | None,
    *,
    show: bool | None = None,
) -> bool | None:
    """Allow runtime extensions to configure optional one-time device info logging."""
    normalized_target_id = normalize_compute_target_id(target_id)
    runtime_hint = normalized_target_id.split(":", 1)[0]
    extension = get_runtime_extension(runtime_hint)
    if extension is None:
        target = get_compute_target(normalized_target_id, include_experimental=True)
        runtime = target.runtime if target is not None else runtime_hint
        extension = get_runtime_extension(runtime)
    if extension is None or extension.configure_device_info is None:
        return None
    return extension.configure_device_info(show)
