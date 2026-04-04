# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import textwrap
import threading
import time
from copy import deepcopy

import torch

from lada import LOG_LEVEL
from lada.models.yolo.detection_backends import MosaicDetectionModel
from lada.utils import Image, ImageTensor, threading_utils, video_utils
from lada.utils.threading_utils import (
    EOF_MARKER,
    STOP_MARKER,
    ErrorMarker,
    PipelineQueue,
    PipelineThread,
    StopMarker,
)

from .frame_restorer_worker import (
    run_clip_preprocess_worker,
    run_clip_restoration_worker,
    run_frame_restoration_worker,
)
from .mosaic_detector import MosaicDetector
from .runtime_options import (
    resolve_basicvsrpp_vulkan_runtime_features,
    resolve_restoration_scheduling_options,
)
from .runtime_profiling import WallClockProfiler

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


class FrameRestorer:
    def __init__(
        self,
        device,
        video_file,
        max_clip_length,
        mosaic_restoration_model_name,
        mosaic_detection_model: MosaicDetectionModel,
        mosaic_restoration_model,
        preferred_pad_mode,
        mosaic_detection=False,
    ):
        self.device = torch.device(device)
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.video_meta_data = video_utils.get_video_meta_data(video_file)
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = 0
        self.start_frame = 0
        self.mosaic_detection = mosaic_detection
        self.eof = False
        self.stop_requested = False
        self.runtime_scheduling = resolve_restoration_scheduling_options(
            self.mosaic_restoration_model,
        )
        self.runtime_features = resolve_basicvsrpp_vulkan_runtime_features(
            self.mosaic_restoration_model,
        )
        self.stream_restore_chunk_size = self.runtime_scheduling.stream_restore_chunk_size

        max_frames_in_frame_restoration_queue = (
            (512 * 1024 * 1024)
            // (self.video_meta_data.video_width * self.video_meta_data.video_height * 3)
        )
        self.frame_restoration_queue = PipelineQueue(
            name="frame_restoration_queue",
            maxsize=max_frames_in_frame_restoration_queue,
        )

        max_clips_in_mosaic_clips_queue = max(
            1,
            (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4),
        )
        self.mosaic_clip_queue = PipelineQueue(
            name="mosaic_clip_queue",
            maxsize=max_clips_in_mosaic_clips_queue,
        )

        max_clips_in_restored_clips_queue = max(
            1,
            (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4),
        )
        self.restored_clip_queue = PipelineQueue(
            name="restored_clip_queue",
            maxsize=max_clips_in_restored_clips_queue,
        )
        self.prepared_clip_queue = PipelineQueue(
            name="prepared_clip_queue",
            maxsize=max_clips_in_restored_clips_queue,
        )

        self.frame_detection_queue = PipelineQueue(
            name="frame_detection_queue",
            maxsize=max_frames_in_frame_restoration_queue,
        )
        self.detector_segment_length = (
            self.runtime_scheduling.resolve_detector_segment_length(
                max_clip_length=self.max_clip_length,
            )
        )
        self.frame_detection_buffer_limit = (
            self.runtime_scheduling.resolve_frame_detection_buffer_limit(
                max_clip_length=self.max_clip_length,
                queue_maxsize=self.frame_detection_queue.maxsize,
            )
        )

        self.mosaic_detector = MosaicDetector(
            self.mosaic_detection_model,
            self.video_meta_data,
            frame_detection_queue=self.frame_detection_queue,
            mosaic_clip_queue=self.mosaic_clip_queue,
            max_clip_length=self.max_clip_length,
            pad_mode=self.preferred_pad_mode,
            batch_size=self.runtime_scheduling.detector_batch_size,
            segment_length=self.detector_segment_length,
            error_handler=self._on_worker_thread_error,
        )

        self.clip_restoration_thread: PipelineThread | None = None
        self.clip_preprocess_thread: PipelineThread | None = None
        self.frame_restoration_thread: PipelineThread | None = None
        self.start_stop_lock = threading.Lock()
        self.profiler = WallClockProfiler()
        self.last_profile: dict[str, object] = {}
        self._profile_started_at = 0.0
        self._clips_restored = 0
        self._clip_frames_restored = 0
        self._frames_blended = 0
        self._frames_passthrough = 0
        self._restore_model_durations: dict[str, float] = {}
        self._restore_model_counts: dict[str, int] = {}

    def _reset_profile_state(self):
        self.profiler.reset()
        self.last_profile = {}
        self._profile_started_at = time.perf_counter()
        self._clips_restored = 0
        self._clip_frames_restored = 0
        self._frames_blended = 0
        self._frames_passthrough = 0
        self._restore_model_durations = {}
        self._restore_model_counts = {}

    def _merge_restore_model_profile(self):
        get_last_profile = getattr(self.mosaic_restoration_model, "get_last_profile", None)
        if not callable(get_last_profile):
            return

        for key, value in get_last_profile().items():
            if key == "total_s" and isinstance(value, (int, float)):
                self._restore_model_durations[key] = (
                    self._restore_model_durations.get(key, 0.0) + float(value)
                )
                continue
            if key.startswith("vulkan_") and isinstance(value, (int, float)):
                self._restore_model_durations[key] = (
                    self._restore_model_durations.get(key, 0.0) + float(value)
                )
                continue
            if key.endswith("__count"):
                bucket = key[:-7]
                if bucket.startswith("vulkan_") and isinstance(value, int):
                    self._restore_model_counts[bucket] = (
                        self._restore_model_counts.get(bucket, 0) + int(value)
                    )

    def _build_restore_model_profile(self) -> dict[str, float | int]:
        profile: dict[str, float | int] = dict(self._restore_model_durations)
        for bucket, count in self._restore_model_counts.items():
            profile[f"{bucket}__count"] = count
        return profile

    def _queue_stats_snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            "frame_restoration_queue": self.frame_restoration_queue.snapshot_stats(),
            "mosaic_clip_queue": self.mosaic_clip_queue.snapshot_stats(),
            "prepared_clip_queue": self.prepared_clip_queue.snapshot_stats(),
            "restored_clip_queue": self.restored_clip_queue.snapshot_stats(),
            "frame_detection_queue": self.frame_detection_queue.snapshot_stats(),
        }

    def _build_profile_snapshot(self) -> dict[str, object]:
        total_s = max(time.perf_counter() - self._profile_started_at, 0.0)
        profile: dict[str, object] = self.profiler.snapshot(total_s=total_s)
        profile["clips_restored"] = self._clips_restored
        profile["clip_frames_restored"] = self._clip_frames_restored
        profile["frames_blended"] = self._frames_blended
        profile["frames_passthrough"] = self._frames_passthrough
        profile["mosaic_detector"] = self.mosaic_detector.get_last_profile()
        profile["restoration_model"] = self._build_restore_model_profile()
        profile["queues"] = self._queue_stats_snapshot()
        return profile

    def start(self, start_ns=0):
        with self.start_stop_lock:
            assert (
                self.frame_restoration_thread is None
                and self.clip_restoration_thread is None
                and self.clip_preprocess_thread is None
            ), (
                "Illegal State: Tried to start FrameRestorer when it's already "
                "running. You need to stop it first"
            )
            assert self.mosaic_clip_queue.empty()
            assert self.prepared_clip_queue.empty()
            assert self.restored_clip_queue.empty()
            assert self.frame_detection_queue.empty()
            assert self.frame_restoration_queue.empty()

            self.start_ns = start_ns
            self.start_frame = video_utils.offset_ns_to_frame_num(
                self.start_ns,
                self.video_meta_data.video_fps_exact,
            )
            self.stop_requested = False
            self._reset_profile_state()

            self.frame_restoration_thread = PipelineThread(
                name="frame restoration worker",
                target=self._frame_restoration_worker,
                error_handler=self._on_worker_thread_error,
            )
            self.clip_preprocess_thread = PipelineThread(
                name="clip preprocess worker",
                target=self._clip_preprocess_worker,
                error_handler=self._on_worker_thread_error,
            )
            self.clip_restoration_thread = PipelineThread(
                name="clip restoration worker",
                target=self._clip_restoration_worker,
                error_handler=self._on_worker_thread_error,
            )

            self.mosaic_detector.start(start_ns=start_ns)
            self.clip_preprocess_thread.start()
            self.clip_restoration_thread.start()
            self.frame_restoration_thread.start()

    def stop(self):
        logger.debug("FrameRestorer: stopping...")
        started_at = time.time()
        with self.start_stop_lock:
            self.stop_requested = True

            self.mosaic_detector.stop()

            threading_utils.put_queue_stop_marker(self.mosaic_clip_queue)
            threading_utils.empty_out_queue(self.prepared_clip_queue)
            if self.clip_preprocess_thread:
                self.clip_preprocess_thread.join()
                logger.debug("FrameRestorer: joined clip_preprocess_thread")
            self.clip_preprocess_thread = None

            threading_utils.put_queue_stop_marker(self.prepared_clip_queue)
            threading_utils.empty_out_queue(self.restored_clip_queue)
            if self.clip_restoration_thread:
                self.clip_restoration_thread.join()
                logger.debug("FrameRestorer: joined clip_restoration_thread")
            self.clip_restoration_thread = None

            threading_utils.put_queue_stop_marker(self.frame_detection_queue)
            threading_utils.empty_out_queue(self.frame_restoration_queue)
            if self.frame_restoration_thread:
                self.frame_restoration_thread.join()
                logger.debug("FrameRestorer: joined frame_restoration_thread")
            self.frame_restoration_thread = None

            threading_utils.empty_out_queue(self.mosaic_clip_queue)
            threading_utils.empty_out_queue(self.prepared_clip_queue)
            threading_utils.empty_out_queue(self.restored_clip_queue)
            threading_utils.empty_out_queue(self.frame_detection_queue)
            threading_utils.empty_out_queue(self.frame_restoration_queue)

            assert self.mosaic_clip_queue.empty()
            assert self.prepared_clip_queue.empty()
            assert self.restored_clip_queue.empty()
            assert self.frame_detection_queue.empty()
            assert self.frame_restoration_queue.empty()

            logger.debug(
                f"FrameRestorer: stopped, took {time.time() - started_at}"
            )
            self._dump_queue_stats()
            self.last_profile = self._build_profile_snapshot()

    def _on_worker_thread_error(self, error: ErrorMarker):
        self.stop_requested = True
        try:
            self.mosaic_detector.stop_requested = True
        except AttributeError:
            pass

        threading_utils.put_queue_stop_marker(self.mosaic_clip_queue)
        threading_utils.put_queue_stop_marker(self.prepared_clip_queue)
        threading_utils.put_queue_stop_marker(self.restored_clip_queue)
        threading_utils.put_queue_stop_marker(self.frame_detection_queue)
        self.frame_restoration_queue.put(error)

    def _dump_queue_stats(self):
        logger.debug(
            textwrap.dedent(
                f"""\
                FrameRestorer Queue stats
                frame_restoration_queue/wait-time-get: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_wait_time_get"]:.0f}
                frame_restoration_queue/wait-time-put: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_wait_time_put"]:.0f}
                frame_restoration_queue/max-qsize: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_max_size"]}/{self.frame_restoration_queue.maxsize}
                ---
                mosaic_clip_queue/wait-time-get: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_wait_time_get"]:.0f}
                mosaic_clip_queue/wait-time-put: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_wait_time_put"]:.0f}
                mosaic_clip_queue/max-qsize: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_max_size"]}/{self.mosaic_clip_queue.maxsize}
                ---
                prepared_clip_queue/wait-time-get: {self.prepared_clip_queue.stats[f"{self.prepared_clip_queue.name}_wait_time_get"]:.0f}
                prepared_clip_queue/wait-time-put: {self.prepared_clip_queue.stats[f"{self.prepared_clip_queue.name}_wait_time_put"]:.0f}
                prepared_clip_queue/max-qsize: {self.prepared_clip_queue.stats[f"{self.prepared_clip_queue.name}_max_size"]}/{self.prepared_clip_queue.maxsize}
                ---
                frame_detection_queue/wait-time-get: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_wait_time_get"]:.0f}
                frame_detection_queue/wait-time-put: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_wait_time_put"]:.0f}
                frame_detection_queue/max-qsize: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_max_size"]}/{self.frame_detection_queue.maxsize}
                ---
                restored_clip_queue/wait-time-get: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_wait_time_get"]:.0f}
                restored_clip_queue/wait-time-put: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_wait_time_put"]:.0f}
                restored_clip_queue/max-qsize: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_max_size"]}/{self.restored_clip_queue.maxsize}
                ---
                frame_feeder_queue/wait-time-get: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_wait_time_get"]:.0f}
                frame_feeder_queue/wait-time-put: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_wait_time_put"]:.0f}
                frame_feeder_queue/max-qsize: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_max_size"]}/{self.mosaic_detector.frame_feeder_queue.maxsize}"""
            )
        )

    def _clip_restoration_worker(self):
        run_clip_restoration_worker(self)

    def _clip_preprocess_worker(self):
        run_clip_preprocess_worker(self)

    def _frame_restoration_worker(self):
        run_frame_restoration_worker(self)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Image | ImageTensor, int] | ErrorMarker | StopMarker:
        if self.eof and self.frame_restoration_queue.empty():
            raise StopIteration

        while True:
            elem = self.frame_restoration_queue.get()
            if (
                self.stop_requested
                or elem is STOP_MARKER
                or isinstance(elem, ErrorMarker)
            ):
                logger.debug("frame_restoration_queue consumer unblocked")
                return elem
            if elem is EOF_MARKER:
                raise StopIteration
            return elem

    def get_frame_restoration_queue(self) -> PipelineQueue:
        return self.frame_restoration_queue

    def get_last_profile(self) -> dict[str, object]:
        return deepcopy(self.last_profile)
