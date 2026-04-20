from __future__ import annotations

import logging

import torch

from lada import LOG_LEVEL, ModelFiles

from .detection import build_vulkan_detection_model
from .restoration import build_vulkan_restoration_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


def _get_detection_classes(
    mosaic_detection_model_path: str,
    detect_face_mosaics: bool,
) -> list[int] | None:
    if not detect_face_mosaics:
        return None

    detection_model = ModelFiles.get_detection_model_by_path(mosaic_detection_model_path)
    detection_model_name = (
        detection_model.name if detection_model is not None else None
    )
    if detection_model_name == "v2":
        logger.info(
            "Mosaic detection model v2 does not support detecting face mosaics. "
            "Use detection models v3 or newer. Ignoring..."
        )
        return None
    return [0]


def load_vulkan_models(
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_restoration_config_path: str | None,
    mosaic_detection_model_path: str,
    fp16: bool,
    detect_face_mosaics: bool,
):
    classes = _get_detection_classes(
        mosaic_detection_model_path=mosaic_detection_model_path,
        detect_face_mosaics=detect_face_mosaics,
    )
    mosaic_restoration_model, pad_mode = build_vulkan_restoration_model(
        model_name=mosaic_restoration_model_name,
        model_path=mosaic_restoration_model_path,
        config_path=mosaic_restoration_config_path,
        compute_target_id="vulkan:0",
        torch_device=torch.device("cpu"),
        fp16=fp16,
    )
    mosaic_detection_model = build_vulkan_detection_model(
        model_path=mosaic_detection_model_path,
        compute_target_id="vulkan:0",
        classes=classes,
        conf=0.15,
        fp16=fp16,
    )
    return mosaic_detection_model, mosaic_restoration_model, pad_mode
