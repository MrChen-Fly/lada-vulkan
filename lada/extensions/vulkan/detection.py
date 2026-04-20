from __future__ import annotations


def build_vulkan_detection_model(
    *,
    model_path: str,
    imgsz: int = 640,
    fp16: bool = False,
    **kwargs,
):
    # Registry callers may forward runtime-only metadata that Ultralytics does not accept.
    kwargs.pop("compute_target_id", None)

    from lada.extensions.vulkan.yolo.yolo_ncnn_runtime import (
        NcnnVulkanYoloSegmentationModel,
    )

    return NcnnVulkanYoloSegmentationModel(
        model_path=model_path,
        imgsz=imgsz,
        fp16=fp16,
        **kwargs,
    )
