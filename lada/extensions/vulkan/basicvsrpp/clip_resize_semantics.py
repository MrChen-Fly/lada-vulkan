from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lada.utils import Image, ImageTensor, Pad
from lada.utils import image_utils

ClipFrameResizeMode = Literal["opencv", "torch_bilinear"]


def floor_resized_length(
    length: int,
    *,
    target_size: int,
    reference_length: int,
) -> int:
    """Match Python/CUDA resize-plan floor semantics without float precision drift."""
    resolved_reference = max(int(reference_length), 1)
    return max(1, (int(length) * int(target_size)) // resolved_reference)


def _resize_with_torch_bilinear(
    frame: Image | ImageTensor,
    resize_shape: tuple[int, int],
) -> Image | ImageTensor:
    """Resize one HWC frame with Torch bilinear semantics on the original 0..255 scale."""
    if isinstance(frame, torch.Tensor):
        tensor = frame.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        resized = F.interpolate(
            tensor,
            size=resize_shape,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        return resized.squeeze(0).permute(1, 2, 0).contiguous()

    tensor = (
        torch.from_numpy(np.ascontiguousarray(frame))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=torch.float32)
    )
    resized = F.interpolate(
        tensor,
        size=resize_shape,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    return np.ascontiguousarray(resized.squeeze(0).permute(1, 2, 0).cpu().numpy())


def resize_and_pad_clip_frame(
    frame: Image | ImageTensor,
    *,
    resize_shape: tuple[int, int],
    pad_after_resize: Pad,
    size: int,
    pad_mode: str,
    resize_mode: ClipFrameResizeMode,
) -> Image | ImageTensor:
    """Resize and pad one clip frame with the requested interpolation semantics."""
    if resize_mode == "opencv":
        resized = image_utils.resize(
            frame,
            resize_shape,
            interpolation=cv2.INTER_LINEAR,
        )
    elif resize_mode == "torch_bilinear":
        resized = _resize_with_torch_bilinear(frame, resize_shape)
    else:
        raise ValueError(f"Unsupported clip resize mode '{resize_mode}'.")

    if pad_after_resize == (0, 0, 0, 0):
        padded = resized
    elif isinstance(resized, torch.Tensor):
        padded = image_utils.pad_image_tensor_by_pad(
            resized,
            pad_after_resize,
            mode=pad_mode,
        )
    else:
        padded = image_utils.pad_image_by_pad(
            np.ascontiguousarray(resized),
            pad_after_resize,
            mode=pad_mode,
        )

    if padded.shape[:2] != (size, size):
        raise ValueError(
            "Clip resize semantics produced an unexpected padded frame shape "
            f"{tuple(padded.shape[:2])} for target size {size}."
        )
    return padded
