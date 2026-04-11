from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lada.utils.ultralytics_utils import convert_yolo_mask_tensor


def to_numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "data") and hasattr(value, "input_shape"):
        value = value.data
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    return np.ascontiguousarray(array)


def quantize_unit_interval_output(value: Any) -> np.ndarray:
    """Quantize one float output tensor to the uint8 image domain used by restore outputs."""
    array = to_numpy_array(value).astype(np.float32, copy=False)
    return np.ascontiguousarray(np.clip(np.round(array * 255.0), 0.0, 255.0), dtype=np.uint8)


def quantize_image_output(value: Any) -> np.ndarray:
    """Quantize one image-domain output to contiguous uint8 pixels."""
    array = to_numpy_array(value)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    float_array = array.astype(np.float32, copy=False)
    if float_array.size == 0:
        return np.ascontiguousarray(float_array, dtype=np.uint8)

    min_value = float(float_array.min())
    max_value = float(float_array.max())
    # Treat small float overshoot/undershoot as unit-interval image output.
    if min_value >= -0.5 and max_value <= 1.5:
        float_array = float_array * 255.0
    return np.ascontiguousarray(np.clip(np.round(float_array), 0.0, 255.0), dtype=np.uint8)


def tensor_summary(array: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "numel": int(array.size),
    }
    if array.size == 0:
        summary.update({"min": None, "max": None, "mean": None})
        return summary
    float_array = array.astype(np.float64, copy=False)
    summary.update(
        {
            "min": float(float_array.min()),
            "max": float(float_array.max()),
            "mean": float(float_array.mean()),
        }
    )
    return summary


def build_probe(name: str, reference: Any, candidate: Any) -> dict[str, Any]:
    reference_array = to_numpy_array(reference)
    candidate_array = to_numpy_array(candidate)
    probe = {
        "name": name,
        "reference": tensor_summary(reference_array),
        "candidate": tensor_summary(candidate_array),
    }
    if reference_array.shape != candidate_array.shape:
        probe["diff"] = {
            "shape_match": False,
            "reference_shape": list(reference_array.shape),
            "candidate_shape": list(candidate_array.shape),
        }
        return probe
    diff = np.abs(
        reference_array.astype(np.float32, copy=False) - candidate_array.astype(np.float32, copy=False)
    )
    probe["diff"] = {
        "shape_match": True,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
    }
    return probe


def _empty_mask_array(height: int, width: int) -> np.ndarray:
    return np.zeros((0, int(height), int(width)), dtype=np.uint8)


def _sort_boxes_and_masks(boxes: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, masks
    order = sorted(
        range(len(boxes)),
        key=lambda index: (
            -float(boxes[index, 4]),
            float(boxes[index, 5]),
            float(boxes[index, 0]),
            float(boxes[index, 1]),
            float(boxes[index, 2]),
            float(boxes[index, 3]),
        ),
    )
    return boxes[order], masks[order]


def extract_detection_arrays(result: Any) -> tuple[np.ndarray, np.ndarray]:
    boxes = (
        result.boxes.data.detach().cpu().numpy().astype(np.float32, copy=False)
        if result.boxes is not None and len(result.boxes) > 0
        else np.zeros((0, 6), dtype=np.float32)
    )
    masks = (
        np.stack(
            [
                convert_yolo_mask_tensor(mask, result.orig_shape)
                .squeeze(-1)
                .cpu()
                .numpy()
                .astype(np.uint8, copy=False)
                for mask in result.masks
            ],
            axis=0,
        )
        if result.masks is not None and result.masks.data is not None
        else _empty_mask_array(*result.orig_shape)
    )
    return _sort_boxes_and_masks(boxes, masks)
