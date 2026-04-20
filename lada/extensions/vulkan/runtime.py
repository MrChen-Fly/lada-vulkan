from __future__ import annotations

import ctypes
import ctypes.util
import sys
from dataclasses import dataclass

from lada.extensions.runtime_registry import ComputeTarget, UnsupportedComputeTargetError

from .ncnn import (
    _iter_local_ncnn_runtime_dirs,
    configure_ncnn_vulkan_device_info,
    get_ncnn_gpu_count,
    import_ncnn_module,
    ncnn_has_lada_basicvsrpp_clip_runner,
    ncnn_has_lada_custom_layer,
    ncnn_has_lada_gridsample_layer,
    ncnn_has_lada_vulkan_net_runner,
    ncnn_has_lada_yolo_attention_layer,
    ncnn_has_lada_yolo_seg_postprocess_vulkan_layer,
)

VULKAN_DEVICE_ID = "vulkan:0"
_VULKAN_DEVICE_ALIASES = {
    "vulkan",
    "vulkan:0",
}


@dataclass(frozen=True)
class NcnnVulkanRuntimeProbe:
    runtime_importable: bool
    vulkan_loader_available: bool
    custom_runtime_supported: bool
    devices: tuple[str, ...]
    error: str | None = None

    @property
    def available(self) -> bool:
        return (
            self.runtime_importable
            and self.vulkan_loader_available
            and self.custom_runtime_supported
            and bool(self.devices)
        )


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


def _supports_local_runtime(ncnn_module: object) -> bool:
    return (
        get_ncnn_gpu_count(ncnn_module) > 0
        and ncnn_has_lada_custom_layer(ncnn_module)
        and ncnn_has_lada_gridsample_layer(ncnn_module)
        and ncnn_has_lada_yolo_attention_layer(ncnn_module)
        and ncnn_has_lada_yolo_seg_postprocess_vulkan_layer(ncnn_module)
        and ncnn_has_lada_vulkan_net_runner(ncnn_module)
        and ncnn_has_lada_basicvsrpp_clip_runner(ncnn_module)
    )


def _resolve_device_name(ncnn_module: object, device_index: int) -> str:
    get_gpu_info = getattr(ncnn_module, "get_gpu_info", None)
    if callable(get_gpu_info):
        try:
            info = get_gpu_info(int(device_index))
            device_name = getattr(info, "device_name", None)
            if callable(device_name):
                resolved = device_name()
            else:
                resolved = device_name
            if resolved:
                return str(resolved)
        except Exception:
            pass
    return f"NCNN Vulkan GPU {device_index}"


def _summarize_runtime_import_origin(ncnn_module: object) -> list[str]:
    notes: list[str] = []
    module_file = getattr(ncnn_module, "__file__", None)
    runtime_dirs = tuple(str(path.resolve()) for path in _iter_local_ncnn_runtime_dirs())

    if module_file:
        notes.append(f"Imported ncnn from '{module_file}'.")
        resolved_module_file = str(module_file).replace("/", "\\").lower()
        if runtime_dirs:
            normalized_dirs = tuple(path.replace("/", "\\").lower() for path in runtime_dirs)
            if not any(resolved_module_file.startswith(path) for path in normalized_dirs):
                notes.append(
                    "The imported ncnn module is outside the detected local runtime directories."
                )
        else:
            notes.append("No local ncnn runtime directories were found.")
    elif not runtime_dirs:
        notes.append("No local ncnn runtime directories were found.")

    return notes


def probe_ncnn_vulkan_runtime() -> NcnnVulkanRuntimeProbe:
    loader_available = _has_vulkan_loader()
    runtime_importable = False
    custom_runtime_supported = False
    devices: tuple[str, ...] = ()
    notes: list[str] = []

    if not loader_available:
        notes.append("Vulkan loader not detected on this machine.")

    ncnn_module = None
    try:
        ncnn_module = import_ncnn_module()
        runtime_importable = True
    except Exception as exc:
        notes.append(f"{type(exc).__name__}: {exc}")

    if loader_available and runtime_importable and ncnn_module is not None:
        try:
            gpu_count = get_ncnn_gpu_count(ncnn_module)
            devices = tuple(
                _resolve_device_name(ncnn_module, device_index)
                for device_index in range(max(gpu_count, 0))
            )
            custom_runtime_supported = _supports_local_runtime(ncnn_module)
            if not devices:
                notes.append("NCNN runtime did not report any Vulkan devices.")
            if devices and not custom_runtime_supported:
                notes.append(
                    "The local ncnn runtime is missing one or more required Lada custom operators."
                )
                notes.extend(_summarize_runtime_import_origin(ncnn_module))
        except Exception as exc:
            notes.append(f"NCNN Vulkan probe failed: {type(exc).__name__}: {exc}")

    return NcnnVulkanRuntimeProbe(
        runtime_importable=runtime_importable,
        vulkan_loader_available=loader_available,
        custom_runtime_supported=custom_runtime_supported,
        devices=devices,
        error=" ".join(notes) if notes else None,
    )


def _build_runtime_note(probe: NcnnVulkanRuntimeProbe) -> str:
    note = (
        "Vulkan backend backed by the local ncnn runtime and Lada custom operators. "
        "Mosaic detection runs through fused YOLO segmentation postprocess on GPU, and "
        "BasicVSR++ restoration runs through the modular 5-frame / recurrent Vulkan graph. "
        "DeepMosaics is not supported."
    )
    if not probe.runtime_importable:
        note += " Install the optional Python package 'ncnn' or point LADA_LOCAL_NCNN_RUNTIME_DIR to the local runtime build."
    elif not probe.custom_runtime_supported:
        note += " Build the local ncnn Vulkan runtime with the required Lada custom operators."
    if probe.error:
        note += f" Probe details: {probe.error}"
    return note


def normalize_vulkan_device(device: str | None) -> str | None:
    if device is None:
        return None
    normalized_device = str(device).strip().lower()
    if normalized_device in _VULKAN_DEVICE_ALIASES:
        return VULKAN_DEVICE_ID
    return str(device).strip()


def is_vulkan_device(device: str | None) -> bool:
    return normalize_vulkan_device(device) == VULKAN_DEVICE_ID


def get_vulkan_target() -> ComputeTarget:
    probe = probe_ncnn_vulkan_runtime()
    description = "Vulkan (NCNN)"
    if probe.devices:
        description = f"Vulkan (NCNN) - {probe.devices[0]}"
    return ComputeTarget(
        id=VULKAN_DEVICE_ID,
        description=description,
        runtime="vulkan",
        available=probe.available,
        torch_device=None,
        notes=_build_runtime_note(probe),
        experimental=False,
    )


def get_vulkan_compute_targets() -> list[ComputeTarget]:
    return [get_vulkan_target()]


def describe_vulkan_issue(device: str | None) -> str | None:
    if not is_vulkan_device(device):
        return None
    target = get_vulkan_target()
    if target.available:
        return None
    return target.notes or f"Compute target '{target.id}' is not available."


def require_vulkan_target(device: str | None) -> ComputeTarget:
    if not is_vulkan_device(device):
        raise UnsupportedComputeTargetError(
            f"Unknown compute target '{device}'. Use --list-devices to inspect supported targets."
        )
    target = get_vulkan_target()
    if not target.available:
        raise UnsupportedComputeTargetError(
            target.notes or f"Compute target '{target.id}' is not available."
        )
    return target


def default_vulkan_fp16_enabled(_target_id: str) -> bool:
    return True


def configure_vulkan_device_info(show: bool | None = None) -> bool:
    return bool(configure_ncnn_vulkan_device_info(show=show))
