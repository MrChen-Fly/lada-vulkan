from __future__ import annotations

from collections.abc import Iterator

from lada.restorationpipeline.mosaic_detector import Clip


def slice_processed_clip(
    clip: Clip,
    *,
    start: int,
    stop: int,
    clip_id=None,
) -> Clip:
    """Build a lightweight Clip view from already-cropped clip tensors."""
    clip_length = len(clip.frames)
    normalized_start = max(int(start), 0)
    normalized_stop = min(int(stop), clip_length)
    if normalized_start >= normalized_stop:
        raise ValueError(
            "Clip slice must contain at least one frame, "
            f"got start={start}, stop={stop}, length={clip_length}."
        )

    sliced_clip = Clip.__new__(Clip)
    sliced_clip.id = clip.id if clip_id is None else clip_id
    sliced_clip.file_path = clip.file_path
    sliced_clip.frame_start = clip.frame_start + normalized_start
    sliced_clip.frame_end = sliced_clip.frame_start + (normalized_stop - normalized_start) - 1
    sliced_clip.size = clip.size
    sliced_clip.pad_mode = clip.pad_mode
    sliced_clip.frames = list(clip.frames[normalized_start:normalized_stop])
    sliced_clip.masks = list(clip.masks[normalized_start:normalized_stop])
    sliced_clip.boxes = list(clip.boxes[normalized_start:normalized_stop])
    sliced_clip.crop_shapes = list(clip.crop_shapes[normalized_start:normalized_stop])
    sliced_clip.pad_after_resizes = list(
        clip.pad_after_resizes[normalized_start:normalized_stop]
    )
    sliced_clip._index = 0
    return sliced_clip


def iter_processed_clip_segments(
    clip: Clip,
    *,
    segment_length: int | None,
) -> Iterator[Clip]:
    """Yield one or more processed clip segments without re-cropping frames."""
    if segment_length is None:
        yield clip
        return

    normalized_segment_length = int(segment_length)
    if normalized_segment_length <= 0 or len(clip.frames) <= normalized_segment_length:
        yield clip
        return

    for start in range(0, len(clip.frames), normalized_segment_length):
        yield slice_processed_clip(
            clip,
            start=start,
            stop=start + normalized_segment_length,
            clip_id=clip.id,
        )
