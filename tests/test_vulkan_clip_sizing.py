from __future__ import annotations

from types import SimpleNamespace

import torch

from lada.restorationpipeline.mosaic_detector import Scene
from lada.utils.threading_utils import PipelineQueue


def _build_scene(
    box: tuple[int, int, int, int],
    *,
    length: int = 1,
) -> Scene:
    frame = torch.zeros((720, 1280, 3), dtype=torch.uint8)
    mask = torch.ones((720, 1280), dtype=torch.uint8)
    scene = Scene("input.webm", SimpleNamespace(video_file="input.webm"))
    for frame_num in range(length):
        scene.add_frame(frame_num, frame, mask, box)
    return scene


def test_vulkan_detector_promotes_large_scenes_to_higher_clip_size() -> None:
    from lada.extensions.vulkan.clip_size_policy import (
        resolve_clip_size_for_edge,
        resolve_restoration_clip_size_options,
    )
    from lada.extensions.vulkan.mosaic_detector import VulkanMosaicDetector

    clip_sizes = resolve_restoration_clip_size_options("basicvsrpp-v1.2")
    assert clip_sizes == (256, 320, 384)
    assert resolve_clip_size_for_edge(250, clip_sizes) == 256
    assert resolve_clip_size_for_edge(320, clip_sizes) == 320
    assert resolve_clip_size_for_edge(626, clip_sizes) == 384

    detector = VulkanMosaicDetector(
        model=SimpleNamespace(),
        video_metadata=SimpleNamespace(video_file="input.webm"),
        frame_detection_queue=PipelineQueue(name="frame_detection_queue"),
        mosaic_clip_queue=PipelineQueue(name="mosaic_clip_queue"),
        error_handler=lambda _error: None,
        clip_size=clip_sizes,
    )

    clip = detector._build_clip(_build_scene((80, 180, 520, 760)), clip_id=0)

    assert clip.size == 384
    assert clip.frames[0].shape[:2] == (384, 384)


def test_vulkan_detector_caps_long_large_scenes_to_memory_safe_clip_size() -> None:
    from lada.extensions.vulkan.clip_size_policy import (
        resolve_restoration_clip_size_options,
    )
    from lada.extensions.vulkan.mosaic_detector import VulkanMosaicDetector

    detector = VulkanMosaicDetector(
        model=SimpleNamespace(),
        video_metadata=SimpleNamespace(video_file="input.webm"),
        frame_detection_queue=PipelineQueue(name="frame_detection_queue"),
        mosaic_clip_queue=PipelineQueue(name="mosaic_clip_queue"),
        error_handler=lambda _error: None,
        max_clip_length=180,
        clip_size=resolve_restoration_clip_size_options("basicvsrpp-v1.2"),
    )

    clip = detector._build_clip(
        _build_scene((80, 180, 520, 760), length=180),
        clip_id=1,
    )

    assert clip.size == 320
    assert clip.frames[0].shape[:2] == (320, 320)
