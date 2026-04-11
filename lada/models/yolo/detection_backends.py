from __future__ import annotations

from typing import Any
from typing import Protocol, runtime_checkable

import torch

from lada.compute_targets import (
    UnsupportedComputeTargetError,
    get_compute_target,
    normalize_compute_target_id,
)
from lada.extensions.runtime_registry import get_runtime_extension
from lada.utils import Image, ImageTensor
from lada.utils.ultralytics_utils import DetectionResult


class _TorchDetectionModelAdapter:
    runtime = "torch"

    def __init__(self, model: object):
        self._model = model
        device = getattr(model, "device", "cpu")
        self.torch_device = torch.device(device)
        self.dtype = getattr(
            model,
            "dtype",
            torch.float16 if bool(getattr(getattr(model, "args", None), "half", False)) else torch.float32,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def prepare_input(self, imgs: torch.Tensor) -> torch.Tensor:
        return imgs.to(device=self.torch_device).to(dtype=self.dtype).div_(255.0)

    def consume_profile(self) -> dict[str, float | int]:
        return {}

    def release_cached_memory(self) -> None:
        if self.torch_device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.torch_device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif (
            self.torch_device.type == "xpu"
            and hasattr(torch, "xpu")
            and hasattr(torch.xpu, "empty_cache")
        ):
            torch.xpu.empty_cache()


@runtime_checkable
class MosaicDetectionModel(Protocol):
    runtime: str
    torch_device: torch.device | None
    dtype: torch.dtype | None

    def preprocess(self, imgs: list[Image | ImageTensor]) -> torch.Tensor | list[Any]:
        ...

    def inference_and_postprocess(
        self,
        imgs: torch.Tensor | list[Any],
        orig_imgs: list[Image | ImageTensor],
    ) -> list[DetectionResult]:
        ...

    def consume_profile(self) -> dict[str, float | int]:
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
        return _TorchDetectionModelAdapter(
            Yolo11SegmentationModel(
                model_path=model_path,
                device=target.torch_device,
                imgsz=imgsz,
                fp16=fp16,
                **kwargs,
            )
        )
    extension = get_runtime_extension(target.runtime)
    if extension is not None and extension.build_detection_model is not None:
        return extension.build_detection_model(
            model_path=model_path,
            compute_target_id=normalized_target_id,
            imgsz=imgsz,
            fp16=fp16,
            **kwargs,
        )
    raise UnsupportedComputeTargetError(
        f"Unsupported mosaic detection runtime '{target.runtime}'."
    )
