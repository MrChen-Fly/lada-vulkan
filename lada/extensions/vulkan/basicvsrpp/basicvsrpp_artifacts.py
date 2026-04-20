from __future__ import annotations

from pathlib import Path

from lada.extensions.runtime_registry import UnsupportedComputeTargetError
from ..ncnn_runtime import (
    ncnn_has_lada_custom_layer,
    ncnn_has_lada_gridsample_layer,
)

from .basicvsrpp_common import (
    NcnnArtifacts,
    _DEFAULT_RUNTIME_FRAME_SHAPE,
    _MODULAR_FRAME_COUNT,
    _delete_ncnn_artifact_group,
    _import_ncnn,
    _prune_extra_ncnn_artifact_files,
    _resolve_ncnn_basicvsrpp_modular_artifacts,
    _validate_ncnn_artifact_group,
)
from .basicvsrpp_export import (
    _build_modular_export_specs,
    _enable_basicvsrpp_export_mode,
    _export_ncnn_module,
)
from .basicvsrpp_runtime_support import (
    resolve_basicvsrpp_runtime_shape,
)


def ensure_ncnn_basicvsrpp_modular_artifacts(
    checkpoint_path: str,
    *,
    config_path: str | dict | None = None,
    frame_count: int = _MODULAR_FRAME_COUNT,
    artifacts_dir: str | Path | None = None,
    frame_shape: int | tuple[int, int] | list[int] = _DEFAULT_RUNTIME_FRAME_SHAPE,
) -> dict[str, NcnnArtifacts]:
    """Build cached ncnn artifacts for one shape-matched modular Vulkan runtime."""
    if frame_count != _MODULAR_FRAME_COUNT:
        raise UnsupportedComputeTargetError(
            f"Modular Vulkan BasicVSR++ currently supports frame_count={_MODULAR_FRAME_COUNT} only."
        )

    runtime_shape = resolve_basicvsrpp_runtime_shape(frame_shape)
    ncnn_module = _import_ncnn()
    if not ncnn_has_lada_custom_layer(ncnn_module):
        raise UnsupportedComputeTargetError(
            "Vulkan BasicVSR++ requires the local ncnn runtime with the Lada deform-conv custom layer."
        )
    if not ncnn_has_lada_gridsample_layer(ncnn_module):
        raise UnsupportedComputeTargetError(
            "Vulkan BasicVSR++ requires the local ncnn runtime with the Lada GridSample custom layer."
        )
    enable_lada_gridsample = True
    artifacts_by_name = _resolve_ncnn_basicvsrpp_modular_artifacts(
        checkpoint_path,
        frame_count=frame_count,
        artifacts_dir=artifacts_dir,
        frame_shape=runtime_shape.frame_shape,
    )
    _prune_extra_ncnn_artifact_files(artifacts_by_name)
    if all(
        artifacts.param_path.exists() and artifacts.bin_path.exists()
        for artifacts in artifacts_by_name.values()
    ):
        try:
            _validate_ncnn_artifact_group(
                artifacts_by_name,
                enable_lada_gridsample=enable_lada_gridsample,
            )
            return artifacts_by_name
        except UnsupportedComputeTargetError:
            _delete_ncnn_artifact_group(artifacts_by_name)

    from lada.models.basicvsrpp.inference import load_model

    model = load_model(config_path, checkpoint_path, "cpu", fp16=False)
    _enable_basicvsrpp_export_mode(model)

    for spec in _build_modular_export_specs(model, runtime_shape=runtime_shape):
        _export_ncnn_module(spec, artifacts_by_name[spec.name])

    _validate_ncnn_artifact_group(
        artifacts_by_name,
        enable_lada_gridsample=enable_lada_gridsample,
    )
    return artifacts_by_name
