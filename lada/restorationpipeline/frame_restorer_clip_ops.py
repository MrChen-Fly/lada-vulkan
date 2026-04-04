from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lada.utils import Image, ImageTensor
from lada.utils import visualization_utils

from .clip_units import (
    Clip,
    ClipDescriptor,
    ClipWorkItem,
    build_clip_resize_plans,
    crop_descriptor_with_profile,
    materialize_clip_masks_with_profile,
)

if TYPE_CHECKING:
    from .frame_restorer import FrameRestorer


def restore_clip_frames(
    restorer: "FrameRestorer",
    images: list[Image | ImageTensor],
) -> list[Image | ImageTensor]:
    if not hasattr(restorer.mosaic_restoration_model, "restore"):
        raise NotImplementedError()
    return restorer.mosaic_restoration_model.restore(images)


def restore_clip(restorer: "FrameRestorer", clip: Clip) -> None:
    """Restore one prepared clip in place."""
    if restorer.mosaic_detection:
        restored_clip_images = visualization_utils.draw_mosaic_detections(clip)
    else:
        restored_clip_images = restore_clip_frames(restorer, clip.frames)
    assert len(restored_clip_images) == len(clip.frames)

    for index, restored_frame in enumerate(restored_clip_images):
        assert clip.frames[index].shape == restored_frame.shape
        clip.frames[index] = restored_frame


def materialize_clip_work_item(
    restorer: "FrameRestorer",
    clip: ClipWorkItem,
) -> Clip:
    if isinstance(clip, Clip):
        return clip
    profile: dict[str, float] = {}
    with restorer.profiler.measure("clip_preprocess_s"):
        materialized = Clip.from_descriptor_with_profile(clip, profile)
    for bucket, duration in profile.items():
        restorer.profiler.add_duration(bucket, duration)
    return materialized


def can_restore_descriptor_directly(restorer: "FrameRestorer") -> bool:
    restore_fn = getattr(
        restorer.mosaic_restoration_model,
        "restore_cropped_clip_frames",
        None,
    )
    return callable(restore_fn) and restorer.runtime_features.supports_descriptor_restore


def prepare_descriptor_for_native_restore(
    restorer: "FrameRestorer",
    clip: ClipDescriptor,
) -> tuple[
    list[Image | ImageTensor],
    list[torch.Tensor],
    list[tuple[int, int, int, int]],
    list[tuple[int, ...]],
    list[tuple[int, int, int, int]],
]:
    profile: dict[str, float] = {}
    cropped_frames, cropped_masks, boxes, crop_shapes = crop_descriptor_with_profile(
        clip,
        profile,
    )
    resize_plans = build_clip_resize_plans(clip, crop_shapes)
    padded_masks, pad_after_resizes = materialize_clip_masks_with_profile(
        cropped_masks,
        resize_plans,
        size=clip.size,
        profile=profile,
    )
    for bucket, duration in profile.items():
        restorer.profiler.add_duration(bucket, duration)
    return cropped_frames, padded_masks, boxes, crop_shapes, pad_after_resizes


def restore_descriptor_work_item(
    restorer: "FrameRestorer",
    clip: ClipDescriptor,
) -> Clip:
    with restorer.profiler.measure("clip_preprocess_s"):
        cropped_frames, padded_masks, boxes, crop_shapes, pad_after_resizes = (
            prepare_descriptor_for_native_restore(restorer, clip)
        )
    with restorer.profiler.measure("clip_restore_s"):
        restored_frames = restorer.mosaic_restoration_model.restore_cropped_clip_frames(
            cropped_frames,
            size=clip.size,
            resize_reference_shape=clip.resize_reference_shape,
            pad_mode=clip.pad_mode,
        )
    return Clip.from_processed_data(
        file_path=clip.file_path,
        frame_start=clip.frame_start,
        size=clip.size,
        pad_mode=clip.pad_mode,
        id=clip.id,
        frames=restored_frames,
        masks=padded_masks,
        boxes=boxes,
        crop_shapes=crop_shapes,
        pad_after_resizes=pad_after_resizes,
    )


def iter_restore_work_units(
    restorer: "FrameRestorer",
    clip: ClipWorkItem,
) -> list[ClipWorkItem]:
    if (
        restorer.stream_restore_chunk_size is None
        or len(clip) <= restorer.stream_restore_chunk_size
    ):
        if isinstance(clip, Clip) or not can_restore_descriptor_directly(restorer):
            return [materialize_clip_work_item(restorer, clip)]
        return [clip]

    if isinstance(clip, Clip):
        return clip.split(restorer.stream_restore_chunk_size)

    segments = clip.split(restorer.stream_restore_chunk_size)
    if can_restore_descriptor_directly(restorer):
        return segments
    return [materialize_clip_work_item(restorer, segment) for segment in segments]


def release_cached_memory_after_completed_clips(
    restorer: "FrameRestorer",
    completed_count: int,
) -> None:
    if completed_count <= 0:
        return
    if hasattr(restorer.mosaic_restoration_model, "release_cached_memory"):
        restorer.mosaic_restoration_model.release_cached_memory()
    if restorer.device.type == "cuda":
        torch.cuda.empty_cache()
    elif restorer.device.type == "mps":
        torch.mps.empty_cache()


def clip_index_contains_all_clips_needed_for_current_restoration(
    current_frame_num: int,
    num_mosaic_detections: int,
    clip_index: dict[int, list[Clip]],
) -> bool:
    num_clips_starting_at_frame = len(clip_index.get(current_frame_num, ()))
    assert num_clips_starting_at_frame <= num_mosaic_detections
    return num_clips_starting_at_frame == num_mosaic_detections


def requeue_ready_clips(
    restorer: "FrameRestorer",
    ready_clips: list[Clip],
    clip_index: dict[int, list[Clip]],
) -> None:
    completed_count = 0
    for clip in ready_clips:
        if len(clip) == 0:
            completed_count += 1
            continue
        clip_index.setdefault(clip.frame_start, []).append(clip)
    release_cached_memory_after_completed_clips(restorer, completed_count)
