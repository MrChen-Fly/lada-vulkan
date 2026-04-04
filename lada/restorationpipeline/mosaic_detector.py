# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import time
from typing import Callable

import torch

from lada import LOG_LEVEL
from lada.models.yolo.detection_backends import MosaicDetectionModel
from lada.utils import VideoMetadata, threading_utils
from lada.utils import video_utils
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, PipelineQueue, StopMarker, PipelineThread, ErrorMarker
from lada.utils.ultralytics_utils import (
    DetectionResult,
    convert_yolo_box,
    convert_yolo_mask_image,
)
from .runtime_profiling import WallClockProfiler
from .clip_units import ClipDescriptor, Scene

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class MosaicDetector:
    def __init__(
        self,
        model: MosaicDetectionModel,
        video_metadata: VideoMetadata,
        frame_detection_queue: PipelineQueue,
        mosaic_clip_queue: PipelineQueue,
        error_handler: Callable[[ErrorMarker], None],
        max_clip_length=30,
        clip_size=256,
        pad_mode='reflect',
        batch_size=4,
        segment_length: int | None = None,
    ):
        self.model = model
        self.video_meta_data = video_metadata
        self.max_clip_length = max_clip_length
        assert max_clip_length > 0
        if segment_length is None or segment_length <= 0:
            segment_length = max_clip_length
        self.segment_length = min(max_clip_length, segment_length)
        self.clip_size = clip_size
        self.pad_mode = pad_mode
        self.clip_counter = 0
        self.start_ns = 0
        self.start_frame = 0
        self.frame_detection_queue = frame_detection_queue
        self.mosaic_clip_queue = mosaic_clip_queue
        self.frame_feeder_queue = PipelineQueue(name="frame_feeder_queue", maxsize=8)
        self.inference_queue = PipelineQueue(name="inference_queue", maxsize=8)
        self.error_handler = error_handler
        self.frame_detector_thread: PipelineThread | None = None
        self.frame_feeder_thread: PipelineThread | None = None
        self.inference_thread: PipelineThread | None = None
        self.stop_requested = False
        self.batch_size = batch_size
        self.profiler = WallClockProfiler()
        self.last_profile: dict[str, object] = {}
        self._profile_started_at = 0.0
        self._frames_read = 0
        self._batches_preprocessed = 0
        self._batches_inferred = 0

    def _reset_profile_state(self):
        self.profiler.reset()
        self.last_profile = {}
        self._profile_started_at = time.perf_counter()
        self._frames_read = 0
        self._batches_preprocessed = 0
        self._batches_inferred = 0

    def _merge_model_profile(self):
        consume_profile = getattr(self.model, "consume_profile", None)
        if not callable(consume_profile):
            return

        for key, value in consume_profile().items():
            if key.endswith("__count"):
                self.profiler.add_count(key[:-7], int(value))
            elif isinstance(value, (int, float)):
                self.profiler.add_duration(key, float(value))

    def _queue_stats_snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            "frame_feeder_queue": self.frame_feeder_queue.snapshot_stats(),
            "inference_queue": self.inference_queue.snapshot_stats(),
            "frame_detection_queue": self.frame_detection_queue.snapshot_stats(),
            "mosaic_clip_queue": self.mosaic_clip_queue.snapshot_stats(),
        }

    def _build_profile_snapshot(self) -> dict[str, object]:
        total_s = max(time.perf_counter() - self._profile_started_at, 0.0)
        profile: dict[str, object] = self.profiler.snapshot(total_s=total_s)
        profile["frames_read"] = self._frames_read
        profile["batches_preprocessed"] = self._batches_preprocessed
        profile["batches_inferred"] = self._batches_inferred
        profile["clips_emitted"] = self.clip_counter
        profile["queues"] = self._queue_stats_snapshot()
        return profile

    def start(self, start_ns):
        assert self.frame_feeder_queue.empty()
        assert self.inference_queue.empty()

        self.start_ns = start_ns
        self.start_frame = video_utils.offset_ns_to_frame_num(self.start_ns, self.video_meta_data.video_fps_exact)
        self.stop_requested = False
        self.clip_counter = 0
        self._reset_profile_state()

        self.frame_detector_thread = PipelineThread(name="frame detector worker", target=self._frame_detector_worker, error_handler=self.error_handler)
        self.frame_detector_thread.start()

        self.inference_thread = PipelineThread(name="frame inference worker", target=self._frame_inference_worker, error_handler=self.error_handler)
        self.inference_thread.start()

        self.frame_feeder_thread = PipelineThread(name="frame feeder worker", target=self._frame_feeder_worker, error_handler=self.error_handler)
        self.frame_feeder_thread.start()

    def stop(self):
        logger.debug("MosaicDetector: stopping...")
        start = time.time()
        self.stop_requested = True

        # unblock producer
        threading_utils.empty_out_queue(self.frame_feeder_queue)
        if self.frame_feeder_thread:
            self.frame_feeder_thread.join()
            logger.debug("MosaicDetector: joined frame_feeder_thread")
        self.frame_feeder_thread = None
        
        # unblock consumer
        threading_utils.put_queue_stop_marker(self.frame_feeder_queue)
        # unblock producer
        threading_utils.empty_out_queue(self.inference_queue)
        if self.inference_thread:
            self.inference_thread.join()
            logger.debug("MosaicDetector: joined inference_thread")
        self.inference_thread = None

        # unblock consumer
        threading_utils.put_queue_stop_marker(self.inference_queue)
        # unblock producer
        threading_utils.empty_out_queue(self.mosaic_clip_queue)
        if self.frame_detector_thread:
            self.frame_detector_thread.join()
            logger.debug("MosaicDetector: joined frame_detector_thread")
        self.frame_detector_thread = None

        # garbage collection
        threading_utils.empty_out_queue(self.frame_feeder_queue)
        threading_utils.empty_out_queue(self.inference_queue)

        assert self.frame_feeder_queue.empty()
        assert self.inference_queue.empty()

        logger.debug(f"MosaicDetector: stopped, took: {time.time() - start}")
        self.last_profile = self._build_profile_snapshot()

    def _emit_scene_descriptor(self, scene: Scene) -> StopMarker | None:
        with self.profiler.measure("clip_descriptor_build_s"):
            clip = ClipDescriptor.from_scene(
                scene,
                self.clip_size,
                self.pad_mode,
                self.clip_counter,
            )
        with self.profiler.measure("mosaic_clip_queue_put_s"):
            self.mosaic_clip_queue.put(clip)
        if self.stop_requested:
            logger.debug("frame detector worker: mosaic_clip_queue producer unblocked")
            return STOP_MARKER
        self.clip_counter += 1
        return None

    def _create_clips_for_completed_scenes(self, scenes, frame_num, eof) -> StopMarker | None:
        completed_scenes = []
        for current_scene in scenes:
            is_scene_finished = current_scene.frame_end < frame_num or eof
            is_scene_ready = len(current_scene) >= self.segment_length
            if (is_scene_finished or is_scene_ready) and current_scene not in completed_scenes:
                completed_scenes.append(current_scene)
                other_scenes = [other for other in scenes if other != current_scene]
                for other_scene in other_scenes:
                    if other_scene.frame_start < current_scene.frame_start and other_scene not in completed_scenes:
                        completed_scenes.append(other_scene)

        for completed_scene in sorted(completed_scenes, key=lambda s: s.frame_start):
            is_scene_finished = completed_scene.frame_end is None or completed_scene.frame_end < frame_num or eof
            if is_scene_finished:
                while len(completed_scene) > 0:
                    emit_length = min(len(completed_scene), self.segment_length)
                    emitted_scene = completed_scene.pop_prefix(emit_length)
                    queue_marker = self._emit_scene_descriptor(emitted_scene)
                    if queue_marker is STOP_MARKER:
                        return STOP_MARKER
                scenes.remove(completed_scene)
                continue

            while len(completed_scene) >= self.segment_length:
                emitted_scene = completed_scene.pop_prefix(self.segment_length)
                queue_marker = self._emit_scene_descriptor(emitted_scene)
                if queue_marker is STOP_MARKER:
                    return STOP_MARKER
            if len(completed_scene) == 0 and completed_scene in scenes:
                scenes.remove(completed_scene)
        return None

    def _create_or_append_scenes_based_on_prediction_result(
        self,
        results: DetectionResult,
        scenes: list[Scene],
        frame_num,
    ) -> int:
        updated_scene_ids: set[int] = set()
        for i in range(len(results.boxes)):
            mask = convert_yolo_mask_image(results.masks[i], results.orig_shape)
            box = convert_yolo_box(results.boxes[i], results.orig_shape)

            current_scene = None
            for scene in scenes:
                if scene.belongs(box):
                    if scene.frame_end == frame_num:
                        current_scene = scene
                        current_scene.merge_mask_box(mask, box)
                    else:
                        current_scene = scene
                        current_scene.add_frame(frame_num, results.orig_img, mask, box)
                    break
            if current_scene is None:
                current_scene = Scene(
                    self.video_meta_data.video_file,
                    self.video_meta_data,
                    self.clip_size,
                )
                scenes.append(current_scene)
                current_scene.add_frame(frame_num, results.orig_img, mask, box)
            updated_scene_ids.add(id(current_scene))
        return len(updated_scene_ids)

    def _frame_feeder_worker(self):
        logger.debug("frame feeder: started")
        eof = False
        with video_utils.VideoReader(self.video_meta_data.video_file) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)
            video_frames_generator = video_reader.frames(output_format="numpy")
            frame_num = self.start_frame
            while not (eof or self.stop_requested):
                decode_started_at = time.perf_counter()
                try:
                    frames = []
                    frame_pts = []
                    for i in range(self.batch_size):
                        frame, pts = next(video_frames_generator)
                        frames.append(frame)
                        frame_pts.append(pts)
                except StopIteration:
                    eof = True
                self.profiler.add_duration("frame_decode_s", time.perf_counter() - decode_started_at)
                if len(frames) > 0:
                    self._frames_read += len(frames)
                    with self.profiler.measure("frame_preprocess_s"):
                        frames_batch = self.model.preprocess(frames)
                    self._batches_preprocessed += 1
                    data = (frames_batch, frames, frame_pts, frame_num)
                    with self.profiler.measure("frame_feeder_queue_put_s"):
                        self.frame_feeder_queue.put(data)
                    if self.stop_requested:
                        logger.debug("frame feeder worker: frame_feeder_queue producer unblocked")
                        break
                frame_num += len(frames)
                if eof:
                    self.frame_feeder_queue.put(EOF_MARKER)
                    if self.stop_requested:
                        logger.debug("frame feeder worker: frame_feeder_queue producer unblocked")
                        break
        if eof:
            logger.debug("frame feeder worker: stopped itself, EOF")
        else:
            logger.debug("frame feeder worker: stopped by request")

    def _frame_inference_worker(self):
        logger.debug("frame inference worker: started")
        eof = False
        while not (eof or self.stop_requested):
            with self.profiler.measure("frame_feeder_queue_get_s"):
                frames_data = self.frame_feeder_queue.get()
            if self.stop_requested or frames_data is STOP_MARKER:
                logger.debug("inference worker: frame_feeder_queue consumer unblocked")
                break
            if frames_data is EOF_MARKER:
                eof = True
                with self.profiler.measure("inference_queue_put_s"):
                    self.inference_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("inference worker: inference_queue producer unblocked")
                    break
                break
            frames_batch, frames, frame_pts, frame_num = frames_data

            with self.profiler.measure("frame_inference_s"):
                batch_prediction_results = self.model.inference_and_postprocess(frames_batch, frames)
            self._merge_model_profile()
            self._batches_inferred += 1

            with self.profiler.measure("inference_queue_put_s"):
                self.inference_queue.put((batch_prediction_results, frames, frame_pts, frame_num))
            if self.stop_requested:
                logger.debug("inference worker: inference_queue producer unblocked")
                break
        if eof:
            logger.debug("inference worker: stopped itself, EOF")
        else:
            logger.debug("inference worker: stopped by request")

    def _frame_detector_worker(self):
        logger.debug("frame detector worker: started")
        scenes: list[Scene] = []
        frame_num = self.start_frame
        eof = False
        while not (eof or self.stop_requested):
            with self.profiler.measure("inference_queue_get_s"):
                inference_data = self.inference_queue.get()
            if self.stop_requested or inference_data is STOP_MARKER:
                logger.debug("frame detector worker: inference_queue consumer unblocked")
                break
            eof = inference_data is EOF_MARKER
            if eof:
                with self.profiler.measure("frame_detector_finalize_s"):
                    self._create_clips_for_completed_scenes(scenes, frame_num, eof=True)
                with self.profiler.measure("frame_detection_queue_put_s"):
                    self.frame_detection_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("frame detector worker: frame_detection_queue producer unblocked")
                    break
                with self.profiler.measure("mosaic_clip_queue_put_s"):
                    self.mosaic_clip_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("frame detector worker: mosaic_clip_queue producer unblocked")
                    break
            else:
                batch_prediction_results, orig_frames, frame_pts, _frame_num = inference_data
                assert frame_num == _frame_num, "frame detector worker out of sync with frame reader"
                with self.profiler.measure("frame_detector_postprocess_s"):
                    for i, results in enumerate(batch_prediction_results):
                        with self.profiler.measure("frame_detector_scene_update_s"):
                            num_scenes_containing_frame = self._create_or_append_scenes_based_on_prediction_result(
                                results,
                                scenes,
                                frame_num,
                            )
                        with self.profiler.measure("frame_detection_queue_put_s"):
                            self.frame_detection_queue.put(
                                (frame_num, num_scenes_containing_frame, orig_frames[i], frame_pts[i])
                            )
                        if self.stop_requested:
                            logger.debug("frame detector worker: frame_detection_queue producer unblocked")
                            break
                        with self.profiler.measure("frame_detector_clip_finalize_s"):
                            queue_marker = self._create_clips_for_completed_scenes(scenes, frame_num, eof=False)
                        if queue_marker is STOP_MARKER:
                            break
                        frame_num += 1
                self.model.release_cached_memory()
        if eof:
            logger.debug("frame detector worker: stopped itself, EOF")
        else:
            logger.debug("frame detector worker: stopped by request")

    def get_last_profile(self) -> dict[str, object]:
        return dict(self.last_profile)
