# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .capabilities import (
    ncnn_has_lada_custom_layer,
    ncnn_has_lada_gridsample_layer,
    ncnn_has_lada_yolo_attention_layer,
    ncnn_has_lada_yolo_seg_postprocess_vulkan_layer,
)
from .loader import import_ncnn_module

_NON_COMPUTE_LAYER_TYPES = frozenset({"Input", "MemoryData", "Noop"})
_CUSTOM_LAYER_SUPPORT_PROBES = {
    "torchvision.deform_conv2d": "deformconv",
    "pnnx.custom_op.torchvision.deform_conv2d": "deformconv",
    "lada.GridSample": "gridsample",
    "pnnx.custom_op.lada.GridSample": "gridsample",
    "lada.YoloAttention": "yolo_attention",
    "pnnx.custom_op.lada.YoloAttention": "yolo_attention",
    "lada.YoloSegPostprocess": "yolo_seg_postprocess",
    "pnnx.custom_op.lada.YoloSegPostprocess": "yolo_seg_postprocess",
}


@dataclass(frozen=True)
class NcnnVulkanLayerAudit:
    """Summarize which ncnn layers can actually run through Vulkan."""

    total_layers: int
    layer_counts: dict[str, int]
    unsupported_layer_counts: dict[str, int]
    unsupported_compute_layer_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_layers": self.total_layers,
            "layer_counts": dict(self.layer_counts),
            "unsupported_layer_counts": dict(self.unsupported_layer_counts),
            "unsupported_compute_layer_counts": dict(self.unsupported_compute_layer_counts),
        }


def _iter_ncnn_param_layer_types(param_path: str | Path) -> list[str]:
    lines = Path(param_path).read_text(encoding="utf-8").splitlines()[2:]
    layer_types: list[str] = []
    for line in lines:
        parts = line.split()
        if parts:
            layer_types.append(parts[0])
    return layer_types


def _layer_supports_vulkan(ncnn_module: Any, layer_type: str) -> bool:
    custom_support = _CUSTOM_LAYER_SUPPORT_PROBES.get(layer_type)
    if custom_support == "deformconv" and ncnn_has_lada_custom_layer(ncnn_module):
        return True
    if custom_support == "gridsample" and ncnn_has_lada_gridsample_layer(ncnn_module):
        return True
    if custom_support == "yolo_attention" and ncnn_has_lada_yolo_attention_layer(ncnn_module):
        return True
    if (
        custom_support == "yolo_seg_postprocess"
        and ncnn_has_lada_yolo_seg_postprocess_vulkan_layer(ncnn_module)
    ):
        return True
    try:
        layer = ncnn_module.create_layer(layer_type)
    except Exception:
        return False
    if layer is None:
        return False
    return bool(layer.support_vulkan)


def audit_ncnn_vulkan_support(
    param_path: str | Path,
    *,
    ncnn_module: Any | None = None,
) -> NcnnVulkanLayerAudit:
    """Inspect an ncnn param file and report layers that will fall back off Vulkan."""
    if ncnn_module is None:
        ncnn_module = import_ncnn_module()

    layer_counts = Counter(_iter_ncnn_param_layer_types(param_path))
    unsupported_layer_counts: dict[str, int] = {}
    unsupported_compute_layer_counts: dict[str, int] = {}
    for layer_type, count in sorted(layer_counts.items()):
        if _layer_supports_vulkan(ncnn_module, layer_type):
            continue
        unsupported_layer_counts[layer_type] = count
        if layer_type not in _NON_COMPUTE_LAYER_TYPES:
            unsupported_compute_layer_counts[layer_type] = count

    return NcnnVulkanLayerAudit(
        total_layers=sum(layer_counts.values()),
        layer_counts=dict(sorted(layer_counts.items())),
        unsupported_layer_counts=unsupported_layer_counts,
        unsupported_compute_layer_counts=unsupported_compute_layer_counts,
    )


def summarize_ncnn_vulkan_audits(
    audits: Mapping[str, NcnnVulkanLayerAudit],
) -> dict[str, Any]:
    """Aggregate per-module Vulkan support audits into one serializable report."""
    total_layers = 0
    unsupported_layer_counts: Counter[str] = Counter()
    unsupported_compute_layer_counts: Counter[str] = Counter()
    module_reports: dict[str, dict[str, Any]] = {}

    for module_name, audit in audits.items():
        total_layers += audit.total_layers
        unsupported_layer_counts.update(audit.unsupported_layer_counts)
        unsupported_compute_layer_counts.update(audit.unsupported_compute_layer_counts)
        module_reports[module_name] = audit.to_dict()

    return {
        "total_layers": total_layers,
        "unsupported_layer_counts": dict(sorted(unsupported_layer_counts.items())),
        "unsupported_compute_layer_counts": dict(sorted(unsupported_compute_layer_counts.items())),
        "modules": module_reports,
    }
