from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from lada.compute_targets import UnsupportedComputeTargetError
from lada.extensions.vulkan.basicvsrpp_ncnn_runtime import (
    ensure_ncnn_gpu_instance,
    import_ncnn_module,
    register_lada_custom_layers,
)
from lada.models.basicvsrpp.vulkan_runtime import patch_ncnn_param_for_vulkan_runtime
from lada.extensions.vulkan.basicvsrpp_runtime_support import (
    get_legacy_modular_artifact_dir_name,
    get_modular_artifact_dir_name,
    resolve_basicvsrpp_runtime_shape,
)

_MODULAR_FRAME_COUNT = 5
_EXPORT_FRAME_SIZE = 256
_FEATURE_SIZE = _EXPORT_FRAME_SIZE // 4
_MODULAR_RUNTIME_REVISION = 16
_DEFAULT_RUNTIME_FRAME_SHAPE = (_EXPORT_FRAME_SIZE, _EXPORT_FRAME_SIZE)
_MODULAR_MODULE_NAMES = (
    "quarter_downsample",
    "feat_extract",
    "spynet",
    "spynet_patch",
    "flow_warp",
    "backward_1_deform_align",
    "forward_1_deform_align",
    "backward_2_deform_align",
    "forward_2_deform_align",
    "restore_clip",
    "backward_1_backbone",
    "forward_1_backbone",
    "backward_2_backbone",
    "forward_2_backbone",
    "backward_1_step",
    "forward_1_step",
    "backward_2_step",
    "forward_2_step",
    "output_frame",
)


@dataclass(frozen=True)
class NcnnArtifacts:
    param_path: Path
    bin_path: Path


@dataclass(frozen=True)
class _ModuleExportSpec:
    name: str
    module: torch.nn.Module
    example_inputs: tuple[torch.Tensor, ...]


def _get_default_vulkan_cache_dir() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    else:
        root = Path(tempfile.gettempdir())
    return root / "lada" / "vulkan_cache"


def _get_bundled_vulkan_cache_dir() -> Path | None:
    bundled_cache = os.environ.get("LADA_BUNDLED_VULKAN_CACHE_DIR")
    if not bundled_cache:
        return None
    cache_path = Path(bundled_cache)
    return cache_path if cache_path.exists() else None


def _as_pnnx_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/")


def _import_ncnn():
    try:
        ncnn = import_ncnn_module()
    except ModuleNotFoundError as exc:
        raise UnsupportedComputeTargetError(
            "Vulkan restoration backend requires the optional Python package 'ncnn'."
        ) from exc
    return ncnn


def _import_pnnx():
    try:
        import pnnx
    except ModuleNotFoundError as exc:
        raise UnsupportedComputeTargetError(
            "Vulkan restoration backend requires the optional Python package 'pnnx' "
            "to build ncnn artifacts from BasicVSR++ checkpoints."
        ) from exc
    resolved_exec_path = os.environ.get("LADA_PNNX_EXEC_PATH")
    if resolved_exec_path and Path(resolved_exec_path).exists():
        pnnx.EXEC_PATH = resolved_exec_path
    return pnnx


def _pick_existing_artifacts_root(
    cache_root: Path | None,
    *,
    shape_dir_name: str,
    legacy_dir_name: str | None,
) -> Path | None:
    if cache_root is None:
        return None

    shape_root = cache_root / shape_dir_name
    if shape_root.exists():
        return shape_root

    if legacy_dir_name is not None:
        legacy_root = cache_root / legacy_dir_name
        if legacy_root.exists():
            return legacy_root

    return None


def _resolve_ncnn_basicvsrpp_modular_artifacts(
    model_path: str,
    *,
    frame_count: int = _MODULAR_FRAME_COUNT,
    artifacts_dir: str | Path | None = None,
    frame_shape: int | tuple[int, int] | list[int] = _DEFAULT_RUNTIME_FRAME_SHAPE,
) -> dict[str, NcnnArtifacts]:
    model_file = Path(model_path)
    runtime_shape = resolve_basicvsrpp_runtime_shape(frame_shape)
    shape_dir_name = get_modular_artifact_dir_name(
        model_file,
        frame_count=frame_count,
        frame_shape=runtime_shape.frame_shape,
        revision=_MODULAR_RUNTIME_REVISION,
    )
    legacy_dir_name = (
        get_legacy_modular_artifact_dir_name(
            model_file,
            frame_count=frame_count,
            revision=_MODULAR_RUNTIME_REVISION,
        )
        if runtime_shape.frame_shape == _DEFAULT_RUNTIME_FRAME_SHAPE
        else None
    )

    if artifacts_dir is not None:
        artifacts_root = _pick_existing_artifacts_root(
            Path(artifacts_dir),
            shape_dir_name=shape_dir_name,
            legacy_dir_name=legacy_dir_name,
        ) or (Path(artifacts_dir) / shape_dir_name)
    else:
        bundled_cache_root = _get_bundled_vulkan_cache_dir()
        artifacts_root = _pick_existing_artifacts_root(
            bundled_cache_root,
            shape_dir_name=shape_dir_name,
            legacy_dir_name=legacy_dir_name,
        )
        if artifacts_root is None:
            artifacts_root = _pick_existing_artifacts_root(
                _get_default_vulkan_cache_dir(),
                shape_dir_name=shape_dir_name,
                legacy_dir_name=legacy_dir_name,
            ) or (_get_default_vulkan_cache_dir() / shape_dir_name)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    return {
        module_name: NcnnArtifacts(
            param_path=artifacts_root / f"{module_name}.ncnn.param",
            bin_path=artifacts_root / f"{module_name}.ncnn.bin",
        )
        for module_name in _MODULAR_MODULE_NAMES
    }


def _validate_ncnn_artifacts(
    artifacts: NcnnArtifacts,
    *,
    enable_lada_gridsample: bool | None = None,
) -> None:
    patch_ncnn_param_for_vulkan_runtime(
        artifacts.param_path,
        enable_lada_gridsample=enable_lada_gridsample,
    )
    ncnn = _import_ncnn()
    net = ncnn.Net()
    register_lada_custom_layers(net, ncnn_module=ncnn)
    ensure_ncnn_gpu_instance(ncnn)
    net.opt.use_vulkan_compute = True
    if net.load_param(str(artifacts.param_path)) != 0:
        raise UnsupportedComputeTargetError(
            f"Failed to load generated ncnn param '{artifacts.param_path}'."
        )
    if net.load_model(str(artifacts.bin_path)) != 0:
        raise UnsupportedComputeTargetError(
            f"Failed to load generated ncnn weights '{artifacts.bin_path}'."
        )


def _validate_ncnn_artifact_group(
    artifacts_by_name: dict[str, NcnnArtifacts],
    *,
    enable_lada_gridsample: bool | None = None,
) -> None:
    for artifacts in artifacts_by_name.values():
        _validate_ncnn_artifacts(
            artifacts,
            enable_lada_gridsample=enable_lada_gridsample,
        )


def _delete_ncnn_artifact_group(artifacts_by_name: dict[str, NcnnArtifacts]) -> None:
    for artifacts in artifacts_by_name.values():
        artifacts.param_path.unlink(missing_ok=True)
        artifacts.bin_path.unlink(missing_ok=True)


def _prune_extra_ncnn_artifact_files(artifacts_by_name: dict[str, NcnnArtifacts]) -> None:
    if not artifacts_by_name:
        return

    artifacts_root = next(iter(artifacts_by_name.values())).param_path.parent
    expected_paths = {
        artifacts.param_path.resolve()
        for artifacts in artifacts_by_name.values()
    } | {
        artifacts.bin_path.resolve()
        for artifacts in artifacts_by_name.values()
    }
    for path in artifacts_root.glob("*.ncnn.*"):
        if path.resolve() not in expected_paths:
            path.unlink(missing_ok=True)
