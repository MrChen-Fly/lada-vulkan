from __future__ import annotations

from lada.compute_targets import UnsupportedComputeTargetError, get_compute_target
from .yolo_model import IreeVulkanYoloSegmentationModel


def _require_available_target(compute_target_id: str) -> None:
    target = get_compute_target(compute_target_id, include_experimental=True)
    if target is None:
        raise UnsupportedComputeTargetError(
            f"Unknown compute target '{compute_target_id}'."
        )
    if not target.available:
        raise UnsupportedComputeTargetError(
            target.notes or f"Compute target '{compute_target_id}' is not available."
        )


def build_vulkan_iree_detection_model(
    *,
    model_path: str,
    compute_target_id: str,
    imgsz: int = 640,
    fp16: bool = False,
    **kwargs,
):
    _require_available_target(compute_target_id)
    _, _, device_index = compute_target_id.partition(":")
    return IreeVulkanYoloSegmentationModel(
        model_path=model_path,
        imgsz=imgsz,
        fp16=fp16,
        device_index=int(device_index or 0),
        **kwargs,
    )
