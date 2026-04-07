from __future__ import annotations

import logging
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING

import numpy as np
import torch

from lada.utils import Image, ImageTensor
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, EofMarker, StopMarker

from .clip_units import Clip, ClipDescriptor
from .frame_restorer_blend import maybe_batch_blend_single_clip_run, restore_frame
from .frame_restorer_clip_ops import (
    can_restore_descriptor_directly,
    clip_index_contains_all_clips_needed_for_current_restoration,
    iter_restore_work_units,
    materialize_clip_work_item,
    requeue_ready_clips,
    restore_clip,
    restore_descriptor_work_item,
)

if TYPE_CHECKING:
    from .frame_restorer import FrameRestorer


logger = logging.getLogger(__name__)

FrameDetectionQueueItem = tuple[int, int, Image | ImageTensor, int]
BufferedFrame = tuple[int, Image | ImageTensor, int]
ClipIndex = dict[int, list[Clip]]
FrameBuffer = dict[int, BufferedFrame]


@dataclass
class FrameRestorationState:
    frame_num: int
    clip_queue_marker: StopMarker | EofMarker | None = None
    frame_queue_marker: StopMarker | EofMarker | None = None
    clip_index: ClipIndex = field(default_factory=dict)
    frame_buffer: FrameBuffer = field(default_factory=dict)


def read_frame_detection_queue_item(
    restorer: "FrameRestorer",
    *,
    block: bool,
    timeout: float | None = None,
) -> FrameDetectionQueueItem | StopMarker | EofMarker | None:
    try:
        with restorer.profiler.measure("frame_detection_queue_get_s"):
            elem = restorer.frame_detection_queue.get(block=block, timeout=timeout)
    except Empty:
        return None

    if restorer.stop_requested or elem is STOP_MARKER:
        logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
        return STOP_MARKER
    if elem is EOF_MARKER:
        return EOF_MARKER
    assert elem is not STOP_MARKER, (
        f"Illegal state: Expected to read detection result from detection queue but received {elem}"
    )
    return elem


def buffer_frame_detection_item(
    item: FrameDetectionQueueItem | StopMarker | EofMarker | None,
    frame_buffer: FrameBuffer,
) -> StopMarker | EofMarker | None:
    if item is None:
        return None
    if item is STOP_MARKER or item is EOF_MARKER:
        return item

    detection_frame_num, num_mosaics_detected, frame, frame_pts = item
    frame_buffer[detection_frame_num] = (num_mosaics_detected, frame, frame_pts)
    return None


def ensure_current_frame_buffered(
    restorer: "FrameRestorer",
    current_frame_num: int,
    frame_buffer: FrameBuffer,
) -> StopMarker | EofMarker | None:
    while current_frame_num not in frame_buffer and not restorer.stop_requested:
        item = read_frame_detection_queue_item(restorer, block=True)
        queue_marker = buffer_frame_detection_item(item, frame_buffer)
        if queue_marker is not None:
            return queue_marker
    return None


def drain_frame_detection_queue(
    restorer: "FrameRestorer",
    frame_buffer: FrameBuffer,
) -> StopMarker | EofMarker | None:
    while (
        len(frame_buffer) < restorer.frame_detection_buffer_limit
        and not restorer.stop_requested
    ):
        item = read_frame_detection_queue_item(restorer, block=False)
        if item is None:
            return None
        queue_marker = buffer_frame_detection_item(item, frame_buffer)
        if queue_marker is not None:
            return queue_marker
    return None


def buffer_restored_clip(
    current_frame_num: int,
    clip: Clip,
    clip_index: ClipIndex,
) -> None:
    assert clip.frame_start >= current_frame_num, "clip queue out of sync!"
    clip_index.setdefault(clip.frame_start, []).append(clip)


def read_next_clip(
    restorer: "FrameRestorer",
    current_frame_num: int,
    clip_index: ClipIndex,
    *,
    block: bool = True,
    timeout: float | None = None,
) -> StopMarker | EofMarker | None:
    try:
        with restorer.profiler.measure("restored_clip_queue_get_s"):
            clip = restorer.restored_clip_queue.get(block=block, timeout=timeout)
    except Empty:
        return None
    if restorer.stop_requested or clip is STOP_MARKER:
        logger.debug("frame restoration worker: restored_clip_queue consumer unblocked")
        return STOP_MARKER
    if clip is EOF_MARKER:
        return EOF_MARKER
    if not isinstance(clip, Clip):
        raise TypeError(f"Expected restored clip, got {type(clip)!r}.")
    buffer_restored_clip(current_frame_num, clip, clip_index)
    return None


def prefetch_ready_clips(
    restorer: "FrameRestorer",
    current_frame_num: int,
    clip_index: ClipIndex,
) -> StopMarker | EofMarker | None:
    queue_marker = None
    prefetched = 0
    while True:
        try:
            with restorer.profiler.measure("restored_clip_queue_prefetch_s"):
                clip = restorer.restored_clip_queue.get(block=False)
        except Empty:
            break

        prefetched += 1
        if restorer.stop_requested or clip is STOP_MARKER:
            logger.debug("frame restoration worker: restored_clip_queue prefetch unblocked")
            return STOP_MARKER
        if clip is EOF_MARKER:
            queue_marker = EOF_MARKER
            break
        if not isinstance(clip, Clip):
            raise TypeError(f"Expected restored clip, got {type(clip)!r}.")
        buffer_restored_clip(current_frame_num, clip, clip_index)

    if prefetched > 0:
        restorer.profiler.add_count("restored_clip_queue_prefetch_clip", prefetched)
    return queue_marker


def read_current_frame(
    restorer: "FrameRestorer",
    state: FrameRestorationState,
) -> BufferedFrame | StopMarker | EofMarker | None:
    if state.frame_num not in state.frame_buffer and state.frame_queue_marker is None:
        state.frame_queue_marker = ensure_current_frame_buffered(
            restorer,
            state.frame_num,
            state.frame_buffer,
        )
    if restorer.stop_requested or state.frame_queue_marker is STOP_MARKER:
        return STOP_MARKER
    if state.frame_num not in state.frame_buffer:
        if state.frame_queue_marker is EOF_MARKER:
            return EOF_MARKER
        return None
    return state.frame_buffer.pop(state.frame_num)


def ensure_ready_clips_for_frame(
    restorer: "FrameRestorer",
    state: FrameRestorationState,
    *,
    num_mosaics_detected: int,
) -> StopMarker | EofMarker | None:
    if state.clip_queue_marker is None:
        prefetched_marker = prefetch_ready_clips(
            restorer,
            state.frame_num,
            state.clip_index,
        )
        if prefetched_marker is not None:
            state.clip_queue_marker = prefetched_marker

    while (
        state.clip_queue_marker is None
        and not clip_index_contains_all_clips_needed_for_current_restoration(
            state.frame_num,
            num_mosaics_detected,
            state.clip_index,
        )
    ):
        state.clip_queue_marker = read_next_clip(
            restorer,
            state.frame_num,
            state.clip_index,
            block=False,
        )
        if state.clip_queue_marker is None:
            if state.frame_queue_marker is None:
                buffered_marker = drain_frame_detection_queue(
                    restorer,
                    state.frame_buffer,
                )
                if buffered_marker is not None:
                    state.frame_queue_marker = buffered_marker
            state.clip_queue_marker = read_next_clip(
                restorer,
                state.frame_num,
                state.clip_index,
                block=True,
                timeout=0.05,
            )
    return state.clip_queue_marker


def refresh_pipeline_prefetch(
    restorer: "FrameRestorer",
    state: FrameRestorationState,
    *,
    next_frame_num: int,
    drain_frames_now: bool,
) -> None:
    if state.clip_queue_marker is None:
        prefetched_marker = prefetch_ready_clips(
            restorer,
            next_frame_num,
            state.clip_index,
        )
        if prefetched_marker is not None:
            state.clip_queue_marker = prefetched_marker
    if drain_frames_now and state.frame_queue_marker is None:
        buffered_marker = drain_frame_detection_queue(restorer, state.frame_buffer)
        if buffered_marker is not None:
            state.frame_queue_marker = buffered_marker


def emit_frame_restoration_output(
    restorer: "FrameRestorer",
    frame: Image | ImageTensor,
    frame_pts: int,
) -> None:
    if isinstance(frame, torch.Tensor):
        # Detach queue consumers from live torch storage before the worker keeps
        # progressing; a raw tensor reference can still be observed as mutated.
        frame = np.ascontiguousarray(frame.detach().cpu().numpy()).copy()
    with restorer.profiler.measure("frame_restoration_queue_put_s"):
        restorer.frame_restoration_queue.put((frame, frame_pts))


def run_clip_restoration_worker(restorer: "FrameRestorer") -> None:
    logger.debug("clip restoration worker: started")
    eof = False
    while not (eof or restorer.stop_requested):
        with restorer.profiler.measure("prepared_clip_queue_get_s"):
            clip = restorer.prepared_clip_queue.get()
        if restorer.stop_requested or clip is STOP_MARKER:
            logger.debug("clip restoration worker: prepared_clip_queue consumer unblocked")
            break
        if clip is EOF_MARKER:
            eof = True
            with restorer.profiler.measure("restored_clip_queue_put_s"):
                restorer.restored_clip_queue.put(EOF_MARKER)
            if restorer.stop_requested:
                logger.debug(
                    "clip restoration worker: restored_clip_queue producer unblocked"
                )
                break
            continue

        if isinstance(clip, ClipDescriptor) and can_restore_descriptor_directly(restorer):
            clip_frame_count = len(clip)
            clip = restore_descriptor_work_item(restorer, clip)
        else:
            if not isinstance(clip, Clip):
                clip = materialize_clip_work_item(restorer, clip)
            clip_frame_count = len(clip.frames)
            with restorer.profiler.measure("clip_restore_s"):
                restore_clip(restorer, clip)
        restorer._clips_restored += 1
        restorer._clip_frames_restored += clip_frame_count
        restorer._merge_restore_model_profile()
        if hasattr(restorer.mosaic_restoration_model, "release_cached_memory"):
            restorer.mosaic_restoration_model.release_cached_memory()
        if restorer.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        with restorer.profiler.measure("restored_clip_queue_put_s"):
            restorer.restored_clip_queue.put(clip)
        if restorer.stop_requested:
            logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
            break

    if eof:
        logger.debug("clip restoration worker: stopped itself, EOF")
    else:
        logger.debug("clip restoration worker: stopped by request")


def run_clip_preprocess_worker(restorer: "FrameRestorer") -> None:
    logger.debug("clip preprocess worker: started")
    eof = False
    while not (eof or restorer.stop_requested):
        with restorer.profiler.measure("mosaic_clip_queue_get_s"):
            clip = restorer.mosaic_clip_queue.get()
        if restorer.stop_requested or clip is STOP_MARKER:
            logger.debug("clip preprocess worker: mosaic_clip_queue consumer unblocked")
            break
        if clip is EOF_MARKER:
            eof = True
            with restorer.profiler.measure("prepared_clip_queue_put_s"):
                restorer.prepared_clip_queue.put(EOF_MARKER)
            if restorer.stop_requested:
                logger.debug(
                    "clip preprocess worker: prepared_clip_queue producer unblocked"
                )
                break
            continue

        for restore_unit in iter_restore_work_units(restorer, clip):
            with restorer.profiler.measure("prepared_clip_queue_put_s"):
                restorer.prepared_clip_queue.put(restore_unit)
            if restorer.stop_requested:
                logger.debug(
                    "clip preprocess worker: prepared_clip_queue producer unblocked"
                )
                break
        if restorer.stop_requested:
            break

    if eof:
        logger.debug("clip preprocess worker: stopped itself, EOF")
    else:
        logger.debug("clip preprocess worker: stopped by request")


def run_frame_restoration_worker(restorer: "FrameRestorer") -> None:
    logger.debug("frame restoration worker: started")
    state = FrameRestorationState(frame_num=restorer.start_frame)

    while not (restorer.eof or restorer.stop_requested):
        frame_item = read_current_frame(restorer, state)
        if restorer.stop_requested or frame_item is STOP_MARKER:
            break
        if frame_item is EOF_MARKER:
            restorer.eof = True
            with restorer.profiler.measure("frame_restoration_queue_put_s"):
                restorer.frame_restoration_queue.put(EOF_MARKER)
            break
        if frame_item is None:
            break

        num_mosaics_detected, frame, frame_pts = frame_item
        if num_mosaics_detected > 0:
            if not isinstance(frame, torch.Tensor):
                with restorer.profiler.measure("frame_tensorize_s"):
                    frame = torch.from_numpy(np.ascontiguousarray(frame))
            if ensure_ready_clips_for_frame(
                restorer,
                state,
                num_mosaics_detected=num_mosaics_detected,
            ) is STOP_MARKER:
                break

            ready_clips = state.clip_index.pop(state.frame_num, [])
            with restorer.profiler.measure("frame_blend_s"):
                batched_frames = maybe_batch_blend_single_clip_run(
                    restorer,
                    frame_num=state.frame_num,
                    frame=frame,
                    frame_pts=frame_pts,
                    num_mosaics_detected=num_mosaics_detected,
                    ready_clips=ready_clips,
                    clip_index=state.clip_index,
                    frame_buffer=state.frame_buffer,
                )
            if batched_frames is not None:
                restorer._frames_blended += len(batched_frames)
                for batched_frame, batched_pts in batched_frames:
                    emit_frame_restoration_output(restorer, batched_frame, batched_pts)
                    if restorer.stop_requested:
                        logger.debug(
                            "frame restoration worker: frame_restoration_queue producer unblocked"
                        )
                        break
                if restorer.stop_requested:
                    break
                state.frame_num += len(batched_frames)
                refresh_pipeline_prefetch(
                    restorer,
                    state,
                    next_frame_num=state.frame_num,
                    drain_frames_now=True,
                )
                continue

            with restorer.profiler.measure("frame_blend_s"):
                restore_frame(restorer, frame, state.frame_num, ready_clips)
            restorer._frames_blended += 1
            emit_frame_restoration_output(restorer, frame, frame_pts)
            if restorer.stop_requested:
                logger.debug(
                    "frame restoration worker: frame_restoration_queue producer unblocked"
                )
                break
            requeue_ready_clips(restorer, ready_clips, state.clip_index)
            refresh_pipeline_prefetch(
                restorer,
                state,
                next_frame_num=state.frame_num + 1,
                drain_frames_now=False,
            )
        else:
            restorer._frames_passthrough += 1
            emit_frame_restoration_output(restorer, frame, frame_pts)
            if restorer.stop_requested:
                logger.debug(
                    "frame restoration worker: frame_restoration_queue producer unblocked"
                )
                break
            refresh_pipeline_prefetch(
                restorer,
                state,
                next_frame_num=state.frame_num + 1,
                drain_frames_now=True,
            )
        state.frame_num += 1

    if restorer.eof:
        logger.debug("frame restoration worker: stopped itself, EOF")
    else:
        logger.debug("frame restoration worker: stopped by request")
