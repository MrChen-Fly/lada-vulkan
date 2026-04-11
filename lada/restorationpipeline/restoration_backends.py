from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from lada.compute_targets import (
    UnsupportedComputeTargetError,
    get_compute_target,
    normalize_compute_target_id,
)
from lada.extensions.runtime_registry import get_runtime_extension
from .basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from .runtime_options import (
    RestorationRuntimeFeatures,
    RestorationSchedulingOptions,
)
from lada.utils import Image, ImageTensor


@runtime_checkable
class MosaicRestorationModel(Protocol):
    runtime: str
    dtype: torch.dtype

    def restore(
        self,
        video: list[Image | ImageTensor],
        max_frames: int = -1,
    ) -> list[Image | ImageTensor]:
        ...

    def get_runtime_scheduling_options(self) -> RestorationSchedulingOptions:
        ...

    def get_runtime_features(self) -> RestorationRuntimeFeatures:
        ...

    def release_cached_memory(self) -> None:
        ...


def build_mosaic_restoration_model(
    model_name: str,
    model_path: str,
    config_path: str | dict | None,
    compute_target_id: str,
    torch_device: torch.device | None,
    *,
    fp16: bool,
    artifacts_dir: str | Path | None = None,
) -> tuple[MosaicRestorationModel, str]:
    normalized_target_id = normalize_compute_target_id(compute_target_id)
    target = get_compute_target(normalized_target_id, include_experimental=True)
    if target is None:
        raise UnsupportedComputeTargetError(
            f"Unknown compute target '{compute_target_id}'."
        )

    if target.runtime == "torch":
        if torch_device is None:
            raise UnsupportedComputeTargetError(
                f"Compute target '{normalized_target_id}' does not expose a torch device."
            )
        if model_name.startswith("deepmosaics"):
            from lada.models.deepmosaics.models import loadmodel
            from lada.restorationpipeline.deepmosaics_mosaic_restorer import (
                DeepmosaicsMosaicRestorer,
            )

            model = loadmodel.video(torch_device, model_path, fp16)
            return DeepmosaicsMosaicRestorer(model, torch_device), "reflect"

        if model_name.startswith("basicvsrpp"):
            from lada.models.basicvsrpp.inference import load_model

            model = load_model(config_path, model_path, torch_device, fp16)
            return BasicvsrppMosaicRestorer(model, torch_device, fp16), "zero"

        raise NotImplementedError()

    extension = get_runtime_extension(target.runtime)
    if extension is not None and extension.build_restoration_model is not None:
        return extension.build_restoration_model(
            model_name=model_name,
            model_path=model_path,
            config_path=config_path,
            compute_target_id=normalized_target_id,
            torch_device=torch_device,
            fp16=fp16,
            artifacts_dir=artifacts_dir,
        )

    raise UnsupportedComputeTargetError(
        f"Unsupported mosaic restoration runtime '{target.runtime}'."
    )
