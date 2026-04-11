from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch

from lada.restorationpipeline.basicvsrpp_vulkan_restore_paths import (
    _resize_cropped_frames_for_runtime,
)
from lada.restorationpipeline.clip_units import ClipDescriptor
from lada.restorationpipeline.frame_restorer_blend import restore_frame
from lada.restorationpipeline.frame_restorer_clip_ops import (
    restore_descriptor_work_item,
)
from lada.utils import image_utils, mask_utils


class _FakeProfiler:
    def add_duration(self, _bucket: str, _duration: float) -> None:
        return None

    def add_count(self, _bucket: str) -> None:
        return None

    def measure(self, _bucket: str):
        return nullcontext()


class _FakeDescriptorRestoreModel:
    def __init__(self) -> None:
        self.dtype = torch.float32

    def restore_cropped_clip_frames(
        self,
        cropped_frames: list[np.ndarray],
        *,
        size: int,
        resize_reference_shape: tuple[int, int],
        pad_mode: str,
    ) -> list[torch.Tensor]:
        boosted_frames = [
            np.clip(frame.astype(np.int16) + 32, 0, 255).astype(np.uint8)
            for frame in cropped_frames
        ]
        resized = _resize_cropped_frames_for_runtime(
            boosted_frames,
            size=size,
            resize_reference_shape=resize_reference_shape,
            pad_mode=pad_mode,
        )
        return [
            torch.from_numpy(np.ascontiguousarray(frame))
            for frame in resized
        ]


def _build_descriptor() -> ClipDescriptor:
    frame = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:6, 4:6] = 255
    crop_box = (2, 3, 6, 6)
    return ClipDescriptor(
        file_path="synthetic.webm",
        frame_start=0,
        size=8,
        pad_mode="reflect",
        id="descriptor-blend",
        frames=[frame],
        masks=[mask],
        boxes=[crop_box],
        crop_boxes=[crop_box],
        resize_reference_shape=(7, 6),
    )


def _simulate_expected_frame(
    frame: np.ndarray,
    *,
    clip_img: torch.Tensor,
    clip_mask: torch.Tensor,
    orig_clip_box: tuple[int, int, int, int],
    orig_crop_shape: tuple[int, ...],
    pad_after_resize: tuple[int, int, int, int],
) -> np.ndarray:
    expected = np.ascontiguousarray(frame).copy()
    clip_img_unpadded = image_utils.unpad_image(clip_img, pad_after_resize)
    clip_mask_unpadded = image_utils.unpad_image(clip_mask, pad_after_resize)
    clip_img_resized = image_utils.resize(clip_img_unpadded, orig_crop_shape[:2])
    clip_mask_resized = image_utils.resize(
        clip_mask_unpadded,
        orig_crop_shape[:2],
        interpolation=cv2.INTER_NEAREST,
    )
    blend_mask = mask_utils.create_blend_mask(clip_mask_resized.float()).cpu().numpy()
    clip_img_np = clip_img_resized.cpu().numpy()

    top, left, bottom, right = orig_clip_box
    frame_roi = expected[top : bottom + 1, left : right + 1, :]
    temp = np.empty_like(frame_roi, dtype=np.float32)
    np.subtract(clip_img_np, frame_roi, out=temp, dtype=np.float32)
    np.multiply(temp, blend_mask[..., None], out=temp)
    np.add(temp, frame_roi, out=temp)
    frame_roi[:] = temp.astype(np.uint8)
    return expected


class FrameRestorerDescriptorBlendTests(unittest.TestCase):
    def test_descriptor_restore_blend_matches_reference_with_padding(self) -> None:
        descriptor = _build_descriptor()
        restorer = SimpleNamespace(
            profiler=_FakeProfiler(),
            device=torch.device("cpu"),
            mosaic_restoration_model=_FakeDescriptorRestoreModel(),
        )
        frame_tensor = torch.from_numpy(np.ascontiguousarray(descriptor.frames[0]).copy())
        original_frame = frame_tensor.numpy().copy()

        clip = restore_descriptor_work_item(restorer, descriptor)
        self.assertEqual([(5, 4, 3)], clip.crop_shapes)
        self.assertEqual([(1, 1, 2, 2)], clip.pad_after_resizes)

        clip_img = clip.frames[0].clone()
        clip_mask = clip.masks[0].clone()
        orig_clip_box = clip.boxes[0]
        orig_crop_shape = clip.crop_shapes[0]
        pad_after_resize = clip.pad_after_resizes[0]
        expected_frame = _simulate_expected_frame(
            original_frame,
            clip_img=clip_img,
            clip_mask=clip_mask,
            orig_clip_box=orig_clip_box,
            orig_crop_shape=orig_crop_shape,
            pad_after_resize=pad_after_resize,
        )

        restore_frame(restorer, frame_tensor, frame_num=0, ready_clips=[clip])

        actual_frame = frame_tensor.numpy()
        np.testing.assert_array_equal(expected_frame, actual_frame)
        self.assertFalse(np.array_equal(original_frame, actual_frame))
        top, left, bottom, right = orig_clip_box
        original_roi = original_frame[top : bottom + 1, left : right + 1, :]
        actual_roi = actual_frame[top : bottom + 1, left : right + 1, :]
        self.assertFalse(np.array_equal(original_roi, actual_roi))
        self.assertGreater(int(actual_roi.max()), 0)


if __name__ == "__main__":
    unittest.main()
