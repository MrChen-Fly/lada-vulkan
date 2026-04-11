from __future__ import annotations

import numpy as np
import torch

from lada.extensions.vulkan.basicvsrpp_ncnn_runtime import is_ncnn_vulkan_tensor
from lada.utils import Image, ImageTensor

from lada.extensions.vulkan.basicvsrpp_export import _build_window_indices


def _frame_to_chw_float32(frame: Image | ImageTensor) -> np.ndarray:
    """Convert one HWC uint8 frame into a contiguous CHW float32 array."""
    if isinstance(frame, np.ndarray):
        return np.ascontiguousarray(
            frame.transpose(2, 0, 1).astype(np.float32, copy=False) / 255.0
        )
    return np.ascontiguousarray(
        frame.permute(2, 0, 1).to(dtype=torch.float32).div(255.0).cpu().numpy()
    )


def _array_to_uint8_frame(array: np.ndarray) -> ImageTensor:
    """Convert one NCNN output array back into an HWC uint8 tensor frame."""
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 3:
        raise RuntimeError(f"Unexpected Vulkan output shape {tuple(array.shape)}.")

    if array.shape[0] == 3:
        chw = array
    elif array.shape[-1] == 3:
        chw = np.transpose(array, (2, 0, 1))
    else:
        raise RuntimeError(
            f"Unable to interpret Vulkan output shape {tuple(array.shape)}."
        )

    return (
        torch.from_numpy(np.ascontiguousarray(chw))
        .mul(255.0)
        .round_()
        .clamp_(0, 255)
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
    )


def _split_clip_restore_output(
    array: np.ndarray,
    *,
    frame_count: int,
) -> list[ImageTensor]:
    """Split one fused clip output tensor into per-frame HWC tensors."""
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 3:
        raise RuntimeError(f"Unexpected fused clip output shape {tuple(array.shape)}.")
    if array.shape[0] != frame_count * 3:
        raise RuntimeError(
            f"Expected fused clip output with {frame_count * 3} channels, "
            f"but got shape {tuple(array.shape)}."
        )

    return [
        _array_to_uint8_frame(array[index * 3 : (index + 1) * 3])
        for index in range(frame_count)
    ]


def _build_restore_clip_inputs(
    lqs: list[object],
    spatial_feats: list[object],
    flows_backward: list[object],
    flows_forward: list[object],
) -> dict[str, object]:
    return {
        "in0": lqs[0],
        "in1": lqs[1],
        "in2": lqs[2],
        "in3": lqs[3],
        "in4": lqs[4],
        "in5": spatial_feats[0],
        "in6": spatial_feats[1],
        "in7": spatial_feats[2],
        "in8": spatial_feats[3],
        "in9": spatial_feats[4],
        "in10": flows_backward[0],
        "in11": flows_backward[1],
        "in12": flows_backward[2],
        "in13": flows_backward[3],
        "in14": flows_forward[0],
        "in15": flows_forward[1],
        "in16": flows_forward[2],
        "in17": flows_forward[3],
    }


def _build_replicated_clip_window(
    frames: list[Image | ImageTensor],
    *,
    center_index: int,
    frame_count: int,
) -> list[Image | ImageTensor]:
    """Build a fixed-size clip window by replicating edge frames."""
    window_indices = _build_window_indices(center_index, len(frames), frame_count)
    return [frames[index] for index in window_indices]


def _zeros_like_runtime_value(ncnn_module: object, value: object) -> np.ndarray:
    """Build a float32 zero tensor that matches one runtime tensor payload."""
    if is_ncnn_vulkan_tensor(ncnn_module, value):
        return np.zeros((value.c * value.elempack, value.h, value.w), dtype=np.float32)

    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise RuntimeError(f"Unexpected runtime tensor shape {tuple(array.shape)}.")
    return np.zeros_like(array, dtype=np.float32)


def _build_backbone_inputs(
    module_name: str,
    feat_current: object,
    feat_prop: object,
    branch_feats: dict[str, list[object]],
    frame_index: int,
) -> dict[str, object]:
    if module_name == "backward_1":
        return {"in0": feat_current, "in1": feat_prop}
    if module_name == "forward_1":
        return {
            "in0": feat_current,
            "in1": branch_feats["backward_1"][frame_index],
            "in2": feat_prop,
        }
    if module_name == "backward_2":
        return {
            "in0": feat_current,
            "in1": branch_feats["backward_1"][frame_index],
            "in2": branch_feats["forward_1"][frame_index],
            "in3": feat_prop,
        }
    return {
        "in0": feat_current,
        "in1": branch_feats["backward_1"][frame_index],
        "in2": branch_feats["forward_1"][frame_index],
        "in3": branch_feats["backward_2"][frame_index],
        "in4": feat_prop,
    }


def _build_step_inputs(
    module_name: str,
    feat_prop: object,
    feat_current: object,
    branch_feats: dict[str, list[object]],
    frame_index: int,
    feat_n2: object,
    flow_n1: object,
    prev_flow_n2: object,
) -> dict[str, object]:
    if module_name == "backward_1":
        return {
            "in0": feat_prop,
            "in1": feat_current,
            "in2": feat_n2,
            "in3": flow_n1,
            "in4": prev_flow_n2,
        }
    if module_name == "forward_1":
        return {
            "in0": feat_prop,
            "in1": feat_current,
            "in2": branch_feats["backward_1"][frame_index],
            "in3": feat_n2,
            "in4": flow_n1,
            "in5": prev_flow_n2,
        }
    if module_name == "backward_2":
        return {
            "in0": feat_prop,
            "in1": feat_current,
            "in2": branch_feats["backward_1"][frame_index],
            "in3": branch_feats["forward_1"][frame_index],
            "in4": feat_n2,
            "in5": flow_n1,
            "in6": prev_flow_n2,
        }
    return {
        "in0": feat_prop,
        "in1": feat_current,
        "in2": branch_feats["backward_1"][frame_index],
        "in3": branch_feats["forward_1"][frame_index],
        "in4": branch_feats["backward_2"][frame_index],
        "in5": feat_n2,
        "in6": flow_n1,
        "in7": prev_flow_n2,
    }


def _build_output_frame_inputs(
    lqs: list[object],
    spatial_feats: list[object],
    branch_feats: dict[str, list[object]],
    frame_index: int,
) -> dict[str, object]:
    return {
        "in0": lqs[frame_index],
        "in1": spatial_feats[frame_index],
        "in2": branch_feats["backward_1"][frame_index],
        "in3": branch_feats["forward_1"][frame_index],
        "in4": branch_feats["backward_2"][frame_index],
        "in5": branch_feats["forward_2"][frame_index],
    }


def _is_gpu_bridge_retryable_error(exc: RuntimeError) -> bool:
    """Match runtime bridge failures that should fall back to CPU downloads."""
    message = str(exc)
    return any(
        marker in message
        for marker in (
            "Failed to record Vulkan extractor output.",
            "Failed to record Vulkan extractor output for module",
            "Failed to clone persistent Vulkan tensor",
            "Failed to extract CPU tensor from ncnn extractor.",
            "Failed to extract CPU tensor from module",
            "GPU blob bridge is unavailable",
            "BasicVSR++ frame preprocess",
        )
    )


def _estimate_feature_area_for_frame(frame: Image | ImageTensor) -> int:
    """Estimate the quarter-resolution feature area used by recurrent fast paths."""
    height, width = int(frame.shape[0]), int(frame.shape[1])
    return ((height + 3) // 4) * ((width + 3) // 4)
