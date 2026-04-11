from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from lada.utils import image_utils, mask_utils, Image, ImageTensor

from .clip_units import Clip
from .frame_restorer_clip_ops import release_cached_memory_after_completed_clips

if TYPE_CHECKING:
    from .frame_restorer import FrameRestorer


_ENABLE_NATIVE_BATCH_BLEND_FAST_PATH = False
_ENABLE_NATIVE_BLEND_FAST_PATH = False
_ENABLE_NATIVE_PADDED_BLEND_FAST_PATH = False


def restore_frame(
    restorer: "FrameRestorer",
    frame: ImageTensor,
    frame_num: int,
    ready_clips: list[Clip],
) -> None:
    """Blend the restored clip content back into one frame."""
    is_cpu_input = frame.device.type == "cpu"
    target_dtype = (
        torch.float32 if is_cpu_input else restorer.mosaic_restoration_model.dtype
    )
    native_blend_patch = None
    native_padded_blend_patch = None
    if is_cpu_input:
        padded_candidate = getattr(
            restorer.mosaic_restoration_model,
            "blend_patch_padded",
            None,
        )
        if _ENABLE_NATIVE_PADDED_BLEND_FAST_PATH and callable(padded_candidate):
            native_padded_blend_patch = padded_candidate
        candidate = getattr(restorer.mosaic_restoration_model, "blend_patch", None)
        # Keep the Python CPU blend path as the default until the native blend
        # bridge matches `mask_utils.create_blend_mask()` on border-touching and
        # multi-mosaic clips. The current fast path can still emit black blocks.
        if _ENABLE_NATIVE_BLEND_FAST_PATH and callable(candidate):
            native_blend_patch = candidate

    def _blend_gpu(
        blend_mask: torch.Tensor,
        clip_img: torch.Tensor,
        orig_clip_box: tuple[int, int, int, int],
    ) -> None:
        top, left, bottom, right = orig_clip_box
        frame_roi = frame[top : bottom + 1, left : right + 1, :]
        roi_f = frame_roi.to(dtype=restorer.mosaic_restoration_model.dtype)
        temp = clip_img.to(
            dtype=restorer.mosaic_restoration_model.dtype,
            device=frame_roi.device,
        )
        temp.sub_(roi_f)
        temp.mul_(blend_mask.unsqueeze(-1))
        temp.add_(roi_f)
        temp.round_().clamp_(0, 255)
        frame_roi[:] = temp

    def _blend_cpu(
        blend_mask: torch.Tensor,
        clip_img: torch.Tensor,
        orig_clip_box: tuple[int, int, int, int],
    ) -> None:
        blend_mask_np = blend_mask.cpu().numpy()
        clip_img_np = clip_img.cpu().numpy()
        top, left, bottom, right = orig_clip_box
        frame_roi = frame[top : bottom + 1, left : right + 1, :].numpy()
        temp_buffer = np.empty_like(frame_roi, dtype=np.float32)
        np.subtract(clip_img_np, frame_roi, out=temp_buffer, dtype=np.float32)
        np.multiply(temp_buffer, blend_mask_np[..., None], out=temp_buffer)
        np.add(temp_buffer, frame_roi, out=temp_buffer)
        frame_roi[:] = temp_buffer.astype(np.uint8)

    blend = _blend_cpu if is_cpu_input else _blend_gpu

    for buffered_clip in ready_clips:
        assert buffered_clip.frame_start == frame_num
        restorer.profiler.add_count("frame_blend_clip_s")
        clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize = (
            buffered_clip.pop()
        )
        if native_padded_blend_patch is not None:
            top, left, bottom, right = orig_clip_box
            frame_roi = frame[top : bottom + 1, left : right + 1, :]
            with restorer.profiler.measure("frame_blend_native_s"):
                blended_roi = native_padded_blend_patch(
                    frame_roi,
                    clip_img,
                    clip_mask,
                    pad_after_resize,
                )
            with restorer.profiler.measure("frame_blend_apply_s"):
                if (
                    blended_roi is not frame_roi
                    and blended_roi.data_ptr() != frame_roi.data_ptr()
                ):
                    frame[top : bottom + 1, left : right + 1, :] = blended_roi
            continue

        with restorer.profiler.measure("frame_blend_unpad_s"):
            clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
            clip_mask = image_utils.unpad_image(clip_mask, pad_after_resize)
        with restorer.profiler.measure("frame_blend_resize_s"):
            clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
            clip_mask = image_utils.resize(
                clip_mask,
                orig_crop_shape[:2],
                interpolation=cv2.INTER_NEAREST,
            )
        if native_blend_patch is not None:
            top, left, bottom, right = orig_clip_box
            frame_roi = frame[top : bottom + 1, left : right + 1, :]
            with restorer.profiler.measure("frame_blend_native_s"):
                blended_roi = native_blend_patch(frame_roi, clip_img, clip_mask)
            with restorer.profiler.measure("frame_blend_apply_s"):
                if (
                    blended_roi is not frame_roi
                    and blended_roi.data_ptr() != frame_roi.data_ptr()
                ):
                    frame[top : bottom + 1, left : right + 1, :] = blended_roi
            continue

        with restorer.profiler.measure("frame_blend_mask_s"):
            blend_mask = mask_utils.create_blend_mask(
                clip_mask.to(device=restorer.device).float()
            ).to(device=clip_img.device, dtype=target_dtype)
        with restorer.profiler.measure("frame_blend_apply_s"):
            blend(blend_mask, clip_img, orig_clip_box)


def maybe_batch_blend_single_clip_run(
    restorer: "FrameRestorer",
    *,
    frame_num: int,
    frame: ImageTensor,
    frame_pts: int,
    num_mosaics_detected: int,
    ready_clips: list[Clip],
    clip_index: dict[int, list[Clip]],
    frame_buffer: dict[int, tuple[int, Image | ImageTensor, int]],
) -> list[tuple[Image | ImageTensor, int]] | None:
    # Keep the single-frame native padded-blend path as the safe default until
    # the batched path matches the reference output on full videos.
    if not _ENABLE_NATIVE_BATCH_BLEND_FAST_PATH:
        return None

    batch_blend_fn = getattr(
        restorer.mosaic_restoration_model,
        "blend_patch_padded_batch",
        None,
    )
    if (
        not callable(batch_blend_fn)
        or frame.device.type != "cpu"
        or num_mosaics_detected != 1
        or len(ready_clips) != 1
    ):
        return None

    clip = ready_clips[0]
    if len(clip) < 2:
        return None

    frames_to_emit: list[tuple[Image | ImageTensor, int]] = [(frame, frame_pts)]
    future_frame_nums: list[int] = []
    next_frame_num = frame_num + 1
    while next_frame_num in frame_buffer and len(clip) > len(future_frame_nums) + 1:
        next_num_mosaics, next_frame, next_pts = frame_buffer[next_frame_num]
        if next_num_mosaics != 1:
            break
        if not isinstance(next_frame, torch.Tensor):
            with restorer.profiler.measure("frame_tensorize_s"):
                next_frame = torch.from_numpy(np.ascontiguousarray(next_frame))
            frame_buffer[next_frame_num] = (next_num_mosaics, next_frame, next_pts)
        frames_to_emit.append((next_frame, next_pts))
        future_frame_nums.append(next_frame_num)
        next_frame_num += 1

    if len(frames_to_emit) < 2:
        return None

    frame_rois: list[ImageTensor] = []
    clip_imgs: list[ImageTensor] = []
    clip_masks: list[ImageTensor] = []
    pad_after_resizes: list[tuple[int, int, int, int]] = []
    for blended_frame, _ in frames_to_emit:
        clip_img, clip_mask, orig_clip_box, _orig_crop_shape, pad_after_resize = (
            clip.pop()
        )
        top, left, bottom, right = orig_clip_box
        frame_rois.append(blended_frame[top : bottom + 1, left : right + 1, :])
        clip_imgs.append(clip_img)
        clip_masks.append(clip_mask)
        pad_after_resizes.append(pad_after_resize)

    with restorer.profiler.measure("frame_blend_native_s"):
        batch_blend_fn(frame_rois, clip_imgs, clip_masks, pad_after_resizes)

    for future_num in future_frame_nums:
        frame_buffer.pop(future_num, None)

    completed_count = 0
    if len(clip) == 0:
        completed_count = 1
    else:
        clip_index.setdefault(clip.frame_start, []).append(clip)
    release_cached_memory_after_completed_clips(restorer, completed_count)
    return frames_to_emit
