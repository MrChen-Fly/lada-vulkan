from __future__ import annotations

from pathlib import Path


def normalize_runtime_imgsz(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize one runtime image size into a canonical ``(height, width)`` tuple."""
    if isinstance(imgsz, int):
        return int(imgsz), int(imgsz)
    if len(imgsz) != 2:
        raise ValueError(f"Unsupported NCNN detection image size '{imgsz}'.")
    return int(imgsz[0]), int(imgsz[1])


def get_precision_artifact_dir(
    model_path: Path,
    *,
    fp16: bool,
    imgsz: int | tuple[int, int] | list[int],
) -> Path:
    """Return the shape-specific NCNN artifact directory."""
    stem_path = model_path.with_suffix("")
    precision = "fp16" if fp16 else "fp32"
    height, width = normalize_runtime_imgsz(imgsz)
    return stem_path.parent / f"{stem_path.name}.{precision}.{height}x{width}_ncnn_model"


def get_legacy_precision_artifact_dir(model_path: Path, *, fp16: bool) -> Path:
    """Return the legacy NCNN artifact directory without an ``imgsz`` suffix."""
    stem_path = model_path.with_suffix("")
    precision = "fp16" if fp16 else "fp32"
    return stem_path.parent / f"{stem_path.name}.{precision}_ncnn_model"


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
