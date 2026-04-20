from __future__ import annotations

import logging

from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER
from lada.utils.threading_utils import PipelineQueue

from .basicvsrpp.runtime_options import resolve_restoration_scheduling_options
from .clip_streaming import iter_processed_clip_segments
from .clip_size_policy import (
    resolve_max_clip_size,
    resolve_restoration_clip_size_options,
)
from .mosaic_detector import VulkanMosaicDetector

logger = logging.getLogger(__name__)


class VulkanFrameRestorer(FrameRestorer):
    """Vulkan-specific frame restorer with adaptive clip sizing for large scenes."""

    def __init__(
        self,
        device,
        video_file,
        max_clip_length,
        mosaic_restoration_model_name,
        mosaic_detection_model,
        mosaic_restoration_model,
        preferred_pad_mode,
        mosaic_detection=False,
    ):
        self.runtime_scheduling = resolve_restoration_scheduling_options(
            mosaic_restoration_model
        )
        self.stream_restore_chunk_size = self.runtime_scheduling.stream_restore_chunk_size
        self.detector_batch_size = self.runtime_scheduling.detector_batch_size
        self.detector_segment_length = (
            self.runtime_scheduling.resolve_detector_segment_length(
                max_clip_length=max_clip_length
            )
        )
        self.detector_clip_sizes = resolve_restoration_clip_size_options(
            mosaic_restoration_model_name
        )
        self.max_detector_clip_size = resolve_max_clip_size(self.detector_clip_sizes)
        super().__init__(
            device,
            video_file,
            max_clip_length,
            mosaic_restoration_model_name,
            mosaic_detection_model,
            mosaic_restoration_model,
            preferred_pad_mode,
            mosaic_detection,
        )
        self._rebuild_clip_queues_and_detector()

    def _rebuild_clip_queues_and_detector(self) -> None:
        detector_clip_length = max(int(self.detector_segment_length), 1)
        max_clips_in_mosaic_clips_queue = max(
            1,
            (512 * 1024 * 1024)
            // (
                detector_clip_length
                * self.max_detector_clip_size
                * self.max_detector_clip_size
                * 4
            ),
        )
        self.mosaic_clip_queue = PipelineQueue(
            name="mosaic_clip_queue",
            maxsize=max_clips_in_mosaic_clips_queue,
        )

        max_clips_in_restored_clips_queue = max(
            1,
            (512 * 1024 * 1024)
            // (
                detector_clip_length
                * self.max_detector_clip_size
                * self.max_detector_clip_size
                * 4
            ),
        )
        self.restored_clip_queue = PipelineQueue(
            name="restored_clip_queue",
            maxsize=max_clips_in_restored_clips_queue,
        )
        self.frame_detection_queue = PipelineQueue(name="frame_detection_queue")

        self.mosaic_detector = VulkanMosaicDetector(
            self.mosaic_detection_model,
            self.video_meta_data,
            frame_detection_queue=self.frame_detection_queue,
            mosaic_clip_queue=self.mosaic_clip_queue,
            device=self.device,
            max_clip_length=detector_clip_length,
            clip_size=self.detector_clip_sizes,
            pad_mode=self.preferred_pad_mode,
            batch_size=self.detector_batch_size,
            error_handler=self._on_worker_thread_error,
        )

    def _iter_restore_work_units(self, clip):
        return iter_processed_clip_segments(
            clip,
            segment_length=self.stream_restore_chunk_size,
        )

    def _clip_restoration_worker(self):
        logger.debug("clip restoration worker: started")
        eof = False
        while not (eof or self.stop_requested):
            clip = self.mosaic_clip_queue.get()
            if self.stop_requested or clip is STOP_MARKER:
                logger.debug("clip restoration worker: mosaic_clip_queue consumer unblocked")
                break
            if clip is EOF_MARKER:
                eof = True
                self.restored_clip_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
                    break
            else:
                for work_clip in self._iter_restore_work_units(clip):
                    self._restore_clip(work_clip)
                    self.restored_clip_queue.put(work_clip)
                    if self.stop_requested:
                        logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
                        break
                if self.stop_requested:
                    break
        if eof:
            logger.debug("clip restoration worker: stopped itself, EOF")
        else:
            logger.debug("clip restoration worker: stopped by request")


def build_vulkan_frame_restorer(
    *,
    device,
    video_file,
    max_clip_length,
    mosaic_restoration_model_name,
    mosaic_detection_model,
    mosaic_restoration_model,
    preferred_pad_mode,
    mosaic_detection=False,
):
    return VulkanFrameRestorer(
        device,
        video_file,
        max_clip_length,
        mosaic_restoration_model_name,
        mosaic_detection_model,
        mosaic_restoration_model,
        preferred_pad_mode,
        mosaic_detection,
    )
