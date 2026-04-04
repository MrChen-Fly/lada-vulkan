from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from lada.compute_targets import (
    UnsupportedComputeTargetError,
    get_compute_target,
    normalize_compute_target_id,
)
from lada.models.yolo.ncnn_vulkan import NcnnVulkanYoloSegmentationModel
from lada.utils import Image, ImageTensor
from lada.utils.ultralytics_utils import DetectionResult


@runtime_checkable
class MosaicDetectionModel(Protocol):
    runtime: str
    torch_device: torch.device | None
    dtype: torch.dtype | None

    def preprocess(self, imgs: list[Image | ImageTensor]) -> torch.Tensor:
        ...

    def inference_and_postprocess(
        self,
        imgs: torch.Tensor,
        orig_imgs: list[Image | ImageTensor],
    ) -> list[DetectionResult]:
        ...

    def release_cached_memory(self) -> None:
        ...


def build_mosaic_detection_model(
    model_path: str,
    compute_target_id: str,
    imgsz: int = 640,
    fp16: bool = False,
    **kwargs,
) -> MosaicDetectionModel:
    normalized_target_id = normalize_compute_target_id(compute_target_id)
    target = get_compute_target(normalized_target_id, include_experimental=True)
    if target is None:
        raise UnsupportedComputeTargetError(
            f"Unknown compute target '{compute_target_id}'."
        )
    if target.runtime == "torch":
        if target.torch_device is None:
            raise UnsupportedComputeTargetError(
                f"Compute target '{normalized_target_id}' does not expose a torch device."
            )
        from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

        return Yolo11SegmentationModel(
            model_path=model_path,
            device=target.torch_device,
            imgsz=imgsz,
            fp16=fp16,
            **kwargs,
        )
    if target.runtime == "vulkan":
        return NcnnVulkanYoloSegmentationModel(
            model_path=model_path,
            imgsz=imgsz,
            fp16=fp16,
            **kwargs,
        )
    raise UnsupportedComputeTargetError(
        f"Unsupported mosaic detection runtime '{target.runtime}'."
    )
