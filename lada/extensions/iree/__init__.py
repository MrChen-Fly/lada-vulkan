from __future__ import annotations

from lada.extensions.runtime_registry import RuntimeExtension, register_runtime_extension

from .detection import build_vulkan_iree_detection_model
from .restoration import build_vulkan_iree_restoration_model
from .runtime import (
    configure_vulkan_iree_device_info,
    default_vulkan_iree_fp16_enabled,
    get_vulkan_iree_compute_targets,
)


def register_extension() -> None:
    register_runtime_extension(
        RuntimeExtension(
            runtime="vulkan-iree",
            get_compute_targets=get_vulkan_iree_compute_targets,
            build_detection_model=build_vulkan_iree_detection_model,
            build_restoration_model=build_vulkan_iree_restoration_model,
            default_fp16_enabled=default_vulkan_iree_fp16_enabled,
            configure_device_info=configure_vulkan_iree_device_info,
        )
    )
