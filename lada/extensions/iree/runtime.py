from __future__ import annotations

import ctypes
import ctypes.util
import sys
from dataclasses import dataclass

_SHOW_VULKAN_IREE_DEVICE_INFO = False


@dataclass(frozen=True)
class IreeVulkanRuntimeProbe:
    runtime_importable: bool
    compiler_importable: bool
    vulkan_loader_available: bool
    devices: tuple[str, ...]
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.runtime_importable and self.vulkan_loader_available and bool(self.devices)


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


def probe_iree_vulkan_runtime() -> IreeVulkanRuntimeProbe:
    runtime_importable = False
    compiler_importable = False
    loader_available = _has_vulkan_loader()
    devices: tuple[str, ...] = ()
    notes: list[str] = []

    if not loader_available:
        notes.append("Vulkan loader not detected on this machine.")

    runtime_module = None
    try:
        import iree.runtime as runtime_module  # type: ignore[no-redef]

        runtime_importable = True
    except Exception as exc:
        notes.append(f"{type(exc).__name__}: {exc}")

    try:
        import iree.compiler  # noqa: F401

        compiler_importable = True
    except Exception as exc:
        notes.append(f"{type(exc).__name__}: {exc}")

    if runtime_importable and loader_available and runtime_module is not None:
        try:
            driver = runtime_module.get_driver("vulkan")
            device_infos = driver.query_available_devices()
            devices = tuple(str(info.get("name", info.get("path", "unknown"))) for info in device_infos)
            if not devices:
                notes.append("IREE runtime did not report any Vulkan devices.")
        except Exception as exc:
            notes.append(f"IREE Vulkan device probe failed: {type(exc).__name__}: {exc}")

    return IreeVulkanRuntimeProbe(
        runtime_importable=runtime_importable,
        compiler_importable=compiler_importable,
        vulkan_loader_available=loader_available,
        devices=devices,
        error=" ".join(notes) if notes else None,
    )


def _build_runtime_note(probe: IreeVulkanRuntimeProbe) -> str:
    note = (
        "Experimental Vulkan backend backed by IREE. "
        "The implementation is intentionally isolated under 'lada/extensions/iree/' "
        "so upstream Lada syncs only need a small runtime-registration touchpoint. "
        "This landing is the architecture skeleton for the YOLO pilot; BasicVSR++ "
        "still requires a stage-by-stage audit against the Torch/CUDA baseline."
    )
    if not probe.runtime_importable or not probe.compiler_importable:
        note += " Install the IREE Python packages (`iree-base-runtime`, `iree-base-compiler`, `iree-turbine`)."
    if probe.error:
        note += f" Probe details: {probe.error}"
    return note


def get_vulkan_iree_compute_targets() -> list[object]:
    from lada.compute_targets import ComputeTarget

    probe = probe_iree_vulkan_runtime()
    description = "Vulkan (IREE)"
    if probe.devices:
        description = f"Vulkan (IREE) - {probe.devices[0]}"
    return [
        ComputeTarget(
            id="vulkan-iree:0",
            description=description,
            runtime="vulkan-iree",
            available=probe.available,
            torch_device=None,
            notes=_build_runtime_note(probe),
            experimental=True,
        )
    ]


def default_vulkan_iree_fp16_enabled(_target_id: str) -> bool:
    return False


def configure_vulkan_iree_device_info(show: bool | None = None) -> bool:
    global _SHOW_VULKAN_IREE_DEVICE_INFO
    if show is not None:
        _SHOW_VULKAN_IREE_DEVICE_INFO = bool(show)
    return _SHOW_VULKAN_IREE_DEVICE_INFO
