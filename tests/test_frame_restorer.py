from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lada.restorationpipeline.frame_restorer import FrameRestorer


class _FakeBasicvsrppRuntimeRestorer:
    dtype = torch.float32

    def __init__(self) -> None:
        self.received_images = None

    def restore(self, images):
        self.received_images = images
        return ["restored-frame"]


def test_restore_clip_frames_accepts_runtime_basicvsrpp_restorer() -> None:
    frame_restorer = FrameRestorer.__new__(FrameRestorer)
    runtime_restorer = _FakeBasicvsrppRuntimeRestorer()
    input_images = ["clip-frame"]

    frame_restorer.mosaic_restoration_model_name = "basicvsrpp-v1.2"
    frame_restorer.mosaic_restoration_model = runtime_restorer

    restored_images = frame_restorer._restore_clip_frames(input_images)

    assert restored_images == ["restored-frame"]
    assert runtime_restorer.received_images is input_images


def test_restore_frame_keeps_cpu_blend_mask_off_vulkan_device(monkeypatch) -> None:
    import lada.restorationpipeline.frame_restorer as frame_restorer_module

    frame_restorer = FrameRestorer.__new__(FrameRestorer)
    frame_restorer.device = torch.device("vulkan:0")
    frame_restorer.mosaic_restoration_model = SimpleNamespace(dtype=torch.float32)

    monkeypatch.setattr(frame_restorer_module.image_utils, "unpad_image", lambda image, _pad: image)
    monkeypatch.setattr(
        frame_restorer_module.image_utils,
        "resize",
        lambda image, _shape, interpolation=None: image,
    )
    monkeypatch.setattr(
        frame_restorer_module.mask_utils,
        "create_blend_mask",
        lambda mask: torch.ones_like(mask, dtype=torch.float32),
    )

    clip = SimpleNamespace(
        frame_start=0,
        pop=lambda: (
            torch.zeros((2, 2, 3), dtype=torch.uint8),
            torch.ones((2, 2), dtype=torch.uint8),
            (0, 0, 1, 1),
            (2, 2, 3),
            (0, 0, 0, 0),
        ),
    )
    frame = torch.zeros((2, 2, 3), dtype=torch.uint8)

    frame_restorer._restore_frame(frame, 0, [clip])
