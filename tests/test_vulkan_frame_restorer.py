from __future__ import annotations

from types import SimpleNamespace

import torch

from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.extensions.vulkan.frame_restorer import VulkanFrameRestorer


def test_vulkan_frame_restorer_inherits_shared_cpu_reference_blend_path() -> None:
    assert VulkanFrameRestorer._restore_frame is FrameRestorer._restore_frame


def _build_processed_clip(*, frame_start: int, length: int):
    from lada.restorationpipeline.mosaic_detector import Clip

    clip = Clip.__new__(Clip)
    clip.id = 7
    clip.file_path = "input.webm"
    clip.frame_start = frame_start
    clip.frame_end = frame_start + length - 1
    clip.size = 256
    clip.pad_mode = "zero"
    clip.frames = [torch.zeros((2, 2, 3), dtype=torch.uint8) for _ in range(length)]
    clip.masks = [torch.ones((2, 2, 1), dtype=torch.uint8) for _ in range(length)]
    clip.boxes = [(0, 0, 1, 1) for _ in range(length)]
    clip.crop_shapes = [(2, 2, 3) for _ in range(length)]
    clip.pad_after_resizes = [(0, 0, 0, 0) for _ in range(length)]
    clip._index = 0
    return clip


def test_vulkan_frame_restorer_splits_long_clips_by_runtime_chunk_size() -> None:
    restorer = VulkanFrameRestorer.__new__(VulkanFrameRestorer)
    restorer.stream_restore_head_chunk_size = None
    restorer.stream_restore_chunk_size = 2

    clip = _build_processed_clip(frame_start=10, length=5)

    segments = list(restorer._iter_restore_work_units(clip))

    assert [segment.frame_start for segment in segments] == [10, 12, 14]
    assert [segment.frame_end for segment in segments] == [11, 13, 14]
    assert [len(segment.frames) for segment in segments] == [2, 2, 1]
    assert segments[0].frames is not clip.frames
    assert segments[0].frames[0] is clip.frames[0]


def test_vulkan_frame_restorer_uses_short_head_chunk_before_steady_segments() -> None:
    restorer = VulkanFrameRestorer.__new__(VulkanFrameRestorer)
    restorer.stream_restore_head_chunk_size = 2
    restorer.stream_restore_chunk_size = 4

    clip = _build_processed_clip(frame_start=10, length=10)

    segments = list(restorer._iter_restore_work_units(clip))

    assert [segment.frame_start for segment in segments] == [10, 12, 16]
    assert [segment.frame_end for segment in segments] == [11, 15, 19]
    assert [len(segment.frames) for segment in segments] == [2, 4, 4]
    assert segments[1].frames[0] is clip.frames[2]


def test_vulkan_frame_restorer_ignores_head_chunk_that_is_not_shorter() -> None:
    restorer = VulkanFrameRestorer.__new__(VulkanFrameRestorer)
    restorer.stream_restore_head_chunk_size = 4
    restorer.stream_restore_chunk_size = 4

    clip = _build_processed_clip(frame_start=10, length=9)

    segments = list(restorer._iter_restore_work_units(clip))

    assert [segment.frame_start for segment in segments] == [10, 14, 18]
    assert [segment.frame_end for segment in segments] == [13, 17, 18]
    assert [len(segment.frames) for segment in segments] == [4, 4, 1]


def test_vulkan_frame_restorer_uses_head_chunk_only_for_first_eligible_clip() -> None:
    restorer = VulkanFrameRestorer.__new__(VulkanFrameRestorer)
    restorer.stream_restore_head_chunk_size = 2
    restorer.stream_restore_chunk_size = 4
    restorer._head_chunk_available = True

    first_clip = _build_processed_clip(frame_start=10, length=10)
    second_clip = _build_processed_clip(frame_start=30, length=10)

    first_segments = list(restorer._iter_restore_work_units(first_clip))
    second_segments = list(restorer._iter_restore_work_units(second_clip))

    assert [len(segment.frames) for segment in first_segments] == [2, 4, 4]
    assert [len(segment.frames) for segment in second_segments] == [4, 4, 2]


def test_vulkan_frame_restorer_consumes_runtime_scheduling_options(monkeypatch) -> None:
    from lada.extensions.vulkan.basicvsrpp.runtime_options import (
        RestorationSchedulingOptions,
    )
    import lada.extensions.vulkan.frame_restorer as frame_restorer_module

    captured: dict[str, object] = {}

    def fake_frame_restorer_init(
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
        self.device = torch.device(device)
        self.video_meta_data = SimpleNamespace(
            video_width=1280,
            video_height=720,
            video_file=video_file,
        )
        self.max_clip_length = max_clip_length
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.preferred_pad_mode = preferred_pad_mode
        self.mosaic_detection = mosaic_detection

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(FrameRestorer, "__init__", fake_frame_restorer_init)
    monkeypatch.setattr(frame_restorer_module, "VulkanMosaicDetector", FakeDetector)

    runtime_model = SimpleNamespace(
        get_runtime_scheduling_options=lambda: RestorationSchedulingOptions(
            stream_restore_head_chunk_size=24,
            stream_restore_chunk_size=60,
            detector_batch_size=1,
            detector_segment_length=90,
        )
    )

    restorer = VulkanFrameRestorer(
        device="vulkan:0",
        video_file="input.webm",
        max_clip_length=180,
        mosaic_restoration_model_name="basicvsrpp-v1.2",
        mosaic_detection_model=SimpleNamespace(),
        mosaic_restoration_model=runtime_model,
        preferred_pad_mode="zero",
    )

    assert restorer.stream_restore_head_chunk_size == 24
    assert restorer.stream_restore_chunk_size == 60
    assert restorer.detector_batch_size == 1
    assert restorer.detector_segment_length == 90
    assert captured["batch_size"] == 1
    assert captured["max_clip_length"] == 90
