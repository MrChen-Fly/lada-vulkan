from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from lada.compute_targets import UnsupportedComputeTargetError


def normalize_runtime_imgsz(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize one runtime image size into a canonical ``(height, width)`` tuple."""
    if isinstance(imgsz, int):
        return int(imgsz), int(imgsz)
    if len(imgsz) != 2:
        raise ValueError(f"Unsupported IREE detection image size '{imgsz}'.")
    return int(imgsz[0]), int(imgsz[1])


def get_iree_precision_artifact_dir(
    model_path: Path,
    *,
    fp16: bool,
    imgsz: int | tuple[int, int] | list[int],
) -> Path:
    """Return the shape-specific IREE artifact directory."""
    stem_path = model_path.with_suffix("")
    precision = "fp16" if fp16 else "fp32"
    height, width = normalize_runtime_imgsz(imgsz)
    return stem_path.parent / f"{stem_path.name}.{precision}.{height}x{width}_iree_vulkan_model"


def resolve_letterbox_output_shape(
    source_shape: tuple[int, int],
    *,
    target_shape: int | tuple[int, int] | list[int],
    stride: int,
    auto: bool,
) -> tuple[int, int]:
    """Resolve the post-letterbox tensor shape used by Ultralytics rect preprocessing."""
    src_h = max(int(source_shape[0]), 1)
    src_w = max(int(source_shape[1]), 1)
    dst_h, dst_w = normalize_runtime_imgsz(target_shape)
    ratio = min(float(dst_h) / float(src_h), float(dst_w) / float(src_w))
    resized_h = max(1, int(round(float(src_h) * ratio)))
    resized_w = max(1, int(round(float(src_w) * ratio)))
    if not auto:
        return dst_h, dst_w

    pad_h = (dst_h - resized_h) % max(int(stride), 1)
    pad_w = (dst_w - resized_w) % max(int(stride), 1)
    return resized_h + pad_h, resized_w + pad_w


def coerce_names(value: Any) -> dict[int, str]:
    """Normalize model names loaded from metadata."""
    if isinstance(value, Mapping):
        return {int(key): str(name) for key, name in value.items()}
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    raise UnsupportedComputeTargetError("IREE detection metadata is missing class names.")


def build_segmentation_args(
    *,
    base_overrides: Mapping[str, Any] | None,
    fp16: bool,
    **kwargs,
) -> Any:
    """Build Ultralytics prediction args for the IREE detection runtime."""
    custom = {
        "conf": 0.25,
        "batch": 1,
        "save": False,
        "mode": "predict",
        "device": "cpu",
        "half": fp16,
    }
    return get_cfg(DEFAULT_CFG, {**dict(base_overrides or {}), **custom, **kwargs})
