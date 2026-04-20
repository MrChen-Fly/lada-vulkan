from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        return int(multiple)
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def normalize_basicvsrpp_runtime_shape(
    shape: int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    """Normalize one runtime frame shape into a canonical ``(height, width)`` tuple."""
    if isinstance(shape, int):
        height = int(shape)
        width = int(shape)
    elif len(shape) == 2:
        height = int(shape[0])
        width = int(shape[1])
    else:
        raise ValueError(f"Unsupported BasicVSR++ Vulkan runtime shape '{shape}'.")

    if height <= 0 or width <= 0:
        raise ValueError(
            f"BasicVSR++ Vulkan runtime shape must be positive, got {(height, width)}."
        )
    return height, width


@dataclass(frozen=True)
class BasicvsrppRuntimeShape:
    """Describe the coupled tensor shapes used by one modular BasicVSR++ runtime bundle."""

    frame_shape: tuple[int, int]
    feature_shape: tuple[int, int]
    spynet_patch_shape: tuple[int, int]
    spynet_core_shape: tuple[int, int]


def resolve_basicvsrpp_runtime_shape(
    frame_shape: int | tuple[int, int] | list[int],
) -> BasicvsrppRuntimeShape:
    """Resolve the derived feature/SPyNet shapes for one frame shape."""
    frame_height, frame_width = normalize_basicvsrpp_runtime_shape(frame_shape)
    feature_shape = (
        max(frame_height // 4, 1),
        max(frame_width // 4, 1),
    )
    return BasicvsrppRuntimeShape(
        frame_shape=(frame_height, frame_width),
        feature_shape=feature_shape,
        spynet_patch_shape=feature_shape,
        spynet_core_shape=(
            (192, 320)
            if (frame_height, frame_width) == (256, 256)
            else (
                _round_up_to_multiple(frame_height, 32),
                _round_up_to_multiple(frame_width, 32),
            )
        ),
    )


def get_modular_artifact_dir_name(
    model_path: Path,
    *,
    frame_count: int,
    frame_shape: int | tuple[int, int] | list[int],
    revision: int,
) -> str:
    """Return the shape-specific artifact directory name for one runtime bundle."""
    frame_height, frame_width = normalize_basicvsrpp_runtime_shape(frame_shape)
    return (
        f"{model_path.stem}.vulkan_modular_{int(frame_count)}f."
        f"{frame_height}x{frame_width}_r{int(revision)}"
    )


def get_legacy_modular_artifact_dir_name(
    model_path: Path,
    *,
    frame_count: int,
    revision: int,
) -> str:
    """Return the legacy artifact directory name without a shape suffix."""
    return f"{model_path.stem}.vulkan_modular_{int(frame_count)}f_r{int(revision)}"
