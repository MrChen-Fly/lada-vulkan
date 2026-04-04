from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from lada import LOG_LEVEL, ModelFiles
from lada.models.yolo.detection_backends import (
    MosaicDetectionModel,
    build_mosaic_detection_model,
)

from .restoration_backends import MosaicRestorationModel, build_mosaic_restoration_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


@dataclass(frozen=True)
class LoadedModels:
    """Bundle the models and padding mode consumed by CLI and GUI entrypoints."""

    detection_model: MosaicDetectionModel
    restoration_model: MosaicRestorationModel
    preferred_pad_mode: str


def _resolve_detection_classes(
    *,
    detect_face_mosaics: bool,
    mosaic_detection_model_path: str,
) -> list[int] | None:
    """Resolve the detection classes to keep for the active detection model."""
    if not detect_face_mosaics:
        return None

    detection_model_name = ModelFiles.get_detection_model_by_path(
        mosaic_detection_model_path,
    )
    if detection_model_name == "v2":
        logger.info(
            "Mosaic detection model v2 does not support detecting face mosaics. "
            "Use detection models v3 or newer. Ignoring..."
        )
        return None

    # class id 0 keeps the NSFW mosaic class and filters SFW face mosaics.
    return [0]


def load_models(
    compute_target_id: str,
    device: torch.device | None,
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_restoration_config_path: str | None,
    mosaic_detection_model_path: str,
    fp16: bool,
    detect_face_mosaics: bool,
) -> LoadedModels:
    """Build and return the detection/restoration models used by one pipeline run."""
    restoration_model, preferred_pad_mode = build_mosaic_restoration_model(
        mosaic_restoration_model_name,
        mosaic_restoration_model_path,
        mosaic_restoration_config_path,
        compute_target_id,
        device,
        fp16=fp16,
    )
    detection_model = build_mosaic_detection_model(
        mosaic_detection_model_path,
        compute_target_id,
        classes=_resolve_detection_classes(
            detect_face_mosaics=detect_face_mosaics,
            mosaic_detection_model_path=mosaic_detection_model_path,
        ),
        conf=0.15,
        fp16=fp16,
    )
    return LoadedModels(
        detection_model=detection_model,
        restoration_model=restoration_model,
        preferred_pad_mode=preferred_pad_mode,
    )
