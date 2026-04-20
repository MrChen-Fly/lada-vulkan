# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from .audit import (
    NcnnVulkanLayerAudit,
    audit_ncnn_vulkan_support,
    summarize_ncnn_vulkan_audits,
)
from .capabilities import (
    is_ncnn_vulkan_tensor,
    ncnn_has_lada_basicvsrpp_clip_runner,
    ncnn_has_lada_custom_layer,
    ncnn_has_lada_gridsample_layer,
    ncnn_has_lada_vulkan_net_runner,
    ncnn_has_lada_yolo_attention_layer,
    ncnn_has_lada_yolo_seg_postprocess_layer,
    ncnn_has_lada_yolo_seg_postprocess_vulkan_layer,
    register_lada_custom_layers,
)
from .device import (
    configure_ncnn_vulkan_device_info,
    ensure_ncnn_gpu_instance,
    get_ncnn_gpu_count,
    set_ncnn_vulkan_device,
)
from .loader import (
    _get_local_ncnn_runtime_dir,
    _iter_local_ncnn_runtime_dirs,
    import_ncnn_module,
)
from .runners import (
    NcnnVulkanBasicvsrppClipRunner,
    NcnnVulkanModuleRunner,
)

__all__ = [
    "_get_local_ncnn_runtime_dir",
    "_iter_local_ncnn_runtime_dirs",
    "NcnnVulkanBasicvsrppClipRunner",
    "NcnnVulkanLayerAudit",
    "NcnnVulkanModuleRunner",
    "audit_ncnn_vulkan_support",
    "configure_ncnn_vulkan_device_info",
    "ensure_ncnn_gpu_instance",
    "get_ncnn_gpu_count",
    "import_ncnn_module",
    "is_ncnn_vulkan_tensor",
    "ncnn_has_lada_basicvsrpp_clip_runner",
    "ncnn_has_lada_custom_layer",
    "ncnn_has_lada_gridsample_layer",
    "ncnn_has_lada_vulkan_net_runner",
    "ncnn_has_lada_yolo_attention_layer",
    "ncnn_has_lada_yolo_seg_postprocess_layer",
    "ncnn_has_lada_yolo_seg_postprocess_vulkan_layer",
    "register_lada_custom_layers",
    "set_ncnn_vulkan_device",
    "summarize_ncnn_vulkan_audits",
]
