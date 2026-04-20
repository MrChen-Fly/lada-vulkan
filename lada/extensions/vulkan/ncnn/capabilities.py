# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from typing import Any


def ncnn_has_lada_custom_layer(ncnn_module: Any) -> bool:
    """Return whether the ncnn module already embeds the Lada deform-conv custom layer."""
    return bool(getattr(ncnn_module, "has_lada_torchvision_deform_conv2d", False)) and callable(
        getattr(ncnn_module, "register_torchvision_deform_conv2d_layers", None)
    )


def ncnn_has_lada_gridsample_layer(ncnn_module: Any) -> bool:
    """Return whether the ncnn module already embeds the Lada GridSample custom layer."""
    register_fn = getattr(ncnn_module, "register_lada_gridsample_layers", None)
    register_all_fn = getattr(ncnn_module, "register_lada_custom_layers", None)
    return bool(getattr(ncnn_module, "has_lada_gridsample", False)) and (
        callable(register_fn) or callable(register_all_fn)
    )


def ncnn_has_lada_yolo_attention_layer(ncnn_module: Any) -> bool:
    """Return whether the ncnn module already embeds the Lada YOLO attention custom layer."""
    register_fn = getattr(ncnn_module, "register_lada_yolo_attention_layers", None)
    register_all_fn = getattr(ncnn_module, "register_lada_custom_layers", None)
    return bool(getattr(ncnn_module, "has_lada_yolo_attention", False)) and (
        callable(register_fn) or callable(register_all_fn)
    )


def ncnn_has_lada_yolo_seg_postprocess_layer(ncnn_module: Any) -> bool:
    """Return whether the ncnn module already embeds the Lada YOLO postprocess custom layer."""
    register_fn = getattr(ncnn_module, "register_lada_yolo_seg_postprocess_layers", None)
    register_all_fn = getattr(ncnn_module, "register_lada_custom_layers", None)
    return bool(getattr(ncnn_module, "has_lada_yolo_seg_postprocess_layer", False)) and (
        callable(register_fn) or callable(register_all_fn)
    )


def ncnn_has_lada_yolo_seg_postprocess_vulkan_layer(ncnn_module: Any) -> bool:
    """Return whether the YOLO postprocess custom layer also supports Vulkan compute."""
    return ncnn_has_lada_yolo_seg_postprocess_layer(ncnn_module) and bool(
        getattr(ncnn_module, "has_lada_yolo_seg_postprocess_vulkan", False)
    )


def ncnn_has_lada_vulkan_net_runner(ncnn_module: Any) -> bool:
    """Return whether the ncnn module exposes the local Vulkan blob bridge."""
    runner_type = getattr(ncnn_module, "LadaVulkanNetRunner", None)
    return bool(getattr(ncnn_module, "has_lada_vulkan_net_runner", False)) and callable(
        runner_type
    )


def ncnn_has_lada_basicvsrpp_clip_runner(ncnn_module: Any) -> bool:
    """Return whether the ncnn module exposes the native BasicVSR++ clip runner."""
    runner_type = getattr(ncnn_module, "BasicVsrppClipRunner", None)
    return bool(getattr(ncnn_module, "has_lada_basicvsrpp_clip_runner", False)) and callable(
        runner_type
    )


def is_ncnn_vulkan_tensor(ncnn_module: Any, value: Any) -> bool:
    """Return whether ``value`` is a GPU-resident tensor from the local ncnn runtime."""
    tensor_type = getattr(ncnn_module, "LadaVulkanTensor", None)
    return tensor_type is not None and isinstance(value, tensor_type)


def register_lada_custom_layers(net: Any, *, ncnn_module: Any) -> None:
    """Register built-in Lada custom layers on an ncnn.Net when the runtime supports them."""
    register_all_fn = getattr(ncnn_module, "register_lada_custom_layers", None)
    if callable(register_all_fn):
        result = int(register_all_fn(net))
        if result != 0:
            raise RuntimeError(f"Failed to register Lada custom ncnn layers (code={result}).")
        return

    if ncnn_has_lada_custom_layer(ncnn_module):
        register_fn = getattr(ncnn_module, "register_torchvision_deform_conv2d_layers")
        result = int(register_fn(net))
        if result != 0:
            raise RuntimeError(f"Failed to register Lada deform-conv custom layer (code={result}).")

    if ncnn_has_lada_gridsample_layer(ncnn_module):
        register_fn = getattr(ncnn_module, "register_lada_gridsample_layers")
        result = int(register_fn(net))
        if result != 0:
            raise RuntimeError(f"Failed to register Lada GridSample custom layer (code={result}).")

    if ncnn_has_lada_yolo_attention_layer(ncnn_module):
        register_fn = getattr(ncnn_module, "register_lada_yolo_attention_layers")
        result = int(register_fn(net))
        if result != 0:
            raise RuntimeError(
                f"Failed to register Lada YOLO attention custom layer (code={result})."
            )

    if ncnn_has_lada_yolo_seg_postprocess_layer(ncnn_module):
        register_fn = getattr(ncnn_module, "register_lada_yolo_seg_postprocess_layers")
        result = int(register_fn(net))
        if result != 0:
            raise RuntimeError(
                f"Failed to register Lada YOLO postprocess custom layer (code={result})."
            )
