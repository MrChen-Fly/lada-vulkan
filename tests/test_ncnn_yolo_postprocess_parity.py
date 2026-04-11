from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path

from lada.models.yolo.ncnn_vulkan import NcnnVulkanYoloSegmentationModel
from lada.models.basicvsrpp.ncnn_vulkan import import_ncnn_module
from lada.utils.ultralytics_utils import scale_and_unpad_image


def _cuda_style_process_mask(
    proto: torch.Tensor,
    mask_coeffs: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    input_shape: tuple[int, int],
) -> torch.Tensor:
    c, mask_h, mask_w = proto.shape
    masks = (mask_coeffs @ proto.float().view(c, -1)).view(-1, mask_h, mask_w)

    ratios = torch.tensor(
        [[mask_w / input_shape[1], mask_h / input_shape[0], mask_w / input_shape[1], mask_h / input_shape[0]]],
        dtype=boxes_xyxy.dtype,
    )
    scaled_boxes = boxes_xyxy * ratios
    x1, y1, x2, y2 = torch.chunk(scaled_boxes[:, :, None], 4, 1)
    row_coords = torch.arange(mask_w, dtype=boxes_xyxy.dtype)[None, None, :]
    col_coords = torch.arange(mask_h, dtype=boxes_xyxy.dtype)[None, :, None]
    masks = masks * ((row_coords >= x1) * (row_coords < x2) * (col_coords >= y1) * (col_coords < y2))
    masks = F.interpolate(masks[None], input_shape, mode="bilinear", align_corners=False)[0]
    return masks.gt_(0.0).byte()


def _scale_mask_to_orig(mask: torch.Tensor, orig_shape: tuple[int, int]) -> np.ndarray:
    mask_img = scale_and_unpad_image((mask.to(torch.uint8) * 255).unsqueeze(-1), orig_shape)
    mask_img = torch.where(mask_img > 127, 255, 0).to(torch.uint8)
    return np.ascontiguousarray(mask_img.squeeze(-1).cpu().numpy())


def _reference_finalize_from_selected(
    *,
    boxes: np.ndarray,
    selected: np.ndarray,
    proto: np.ndarray,
    input_shape: tuple[int, int],
    orig_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if selected.size == 0:
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, orig_shape[0], orig_shape[1]), dtype=np.uint8),
        )

    proto_t = torch.from_numpy(np.ascontiguousarray(proto, dtype=np.float32))
    boxes_t = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))
    selected_t = torch.from_numpy(np.ascontiguousarray(selected, dtype=np.float32))

    masks = _cuda_style_process_mask(proto_t, selected_t[:, 6:], selected_t[:, :4], input_shape)
    keep = masks.sum((-2, -1)) > 0
    kept_boxes = boxes_t[keep]
    kept_masks = masks[keep]

    if kept_masks.numel() == 0:
        return (
            kept_boxes.cpu().numpy().astype(np.float32, copy=False),
            np.zeros((0, orig_shape[0], orig_shape[1]), dtype=np.uint8),
        )

    orig_masks = np.stack(
        [_scale_mask_to_orig(mask, orig_shape) for mask in kept_masks],
        axis=0,
    )
    return (
        kept_boxes.cpu().numpy().astype(np.float32, copy=False),
        np.ascontiguousarray(orig_masks, dtype=np.uint8),
    )


class NcnnYoloPostprocessParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ncnn = import_ncnn_module()
        if not getattr(cls.ncnn, "has_lada_yolo_seg_postprocess", False):
            raise unittest.SkipTest("Local ncnn runtime does not expose YOLO postprocess bindings.")

    def test_selected_finalize_uses_cuda_style_fractional_crop(self) -> None:
        input_shape = (8, 8)
        orig_shape = (8, 8)
        proto = np.zeros((1, 4, 4), dtype=np.float32)
        proto[0, 0, 0] = 1.0

        selected = np.array([[0.9, 0.9, 1.9, 1.9, 0.95, 0.0, 1.0]], dtype=np.float32)
        boxes = np.array([[0.9, 0.9, 1.9, 1.9, 0.95, 0.0]], dtype=np.float32)

        native = self.ncnn.postprocess_yolo_segmentation_from_selected(
            boxes,
            selected,
            proto,
            list(input_shape),
            list(orig_shape),
        )
        native_boxes = np.asarray(native["boxes"], dtype=np.float32)
        native_masks = np.asarray(native["masks"], dtype=np.uint8)

        ref_boxes, ref_masks = _reference_finalize_from_selected(
            boxes=boxes,
            selected=selected,
            proto=proto,
            input_shape=input_shape,
            orig_shape=orig_shape,
        )

        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=2e-3)
        np.testing.assert_array_equal(native_masks, ref_masks)
        self.assertEqual(native_boxes.shape[0], 0)

    def test_selected_finalize_keeps_input_space_nonempty_masks(self) -> None:
        input_shape = (8, 8)
        orig_shape = (8, 2)
        proto = np.zeros((1, 4, 4), dtype=np.float32)
        proto[0, :, 0] = 1.0

        selected = np.array([[-0.1, 0.1, 1.9, 7.9, 0.95, 0.0, 1.0]], dtype=np.float32)
        boxes = np.array([[0.0, 0.1, 0.0, 7.9, 0.95, 0.0]], dtype=np.float32)

        native = self.ncnn.postprocess_yolo_segmentation_from_selected(
            boxes,
            selected,
            proto,
            list(input_shape),
            list(orig_shape),
        )
        native_boxes = np.asarray(native["boxes"], dtype=np.float32)
        native_masks = np.asarray(native["masks"], dtype=np.uint8)

        ref_boxes, ref_masks = _reference_finalize_from_selected(
            boxes=boxes,
            selected=selected,
            proto=proto,
            input_shape=input_shape,
            orig_shape=orig_shape,
        )

        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=2e-3)
        np.testing.assert_array_equal(native_masks, ref_masks)
        self.assertEqual(native_boxes.shape[0], 1)
        self.assertEqual(int(native_masks[0].sum()), 0)


class NcnnVulkanYoloPostprocessParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ncnn = import_ncnn_module()
        if not getattr(cls.ncnn, "has_lada_vulkan_net_runner", False):
            raise unittest.SkipTest("Local ncnn runtime does not expose the Vulkan runner.")
        if not hasattr(cls.ncnn, "finalize_yolo_segmentation_masks_gpu"):
            raise unittest.SkipTest("Local ncnn runtime does not expose the Vulkan YOLO finalize entry.")

        cls.model_path = Path("model_weights/lada_mosaic_detection_model_v4_fast.pt")
        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Missing YOLO model artifact: {cls.model_path}")

        cls.model = NcnnVulkanYoloSegmentationModel(
            str(cls.model_path),
            imgsz=640,
            fp16=True,
            device_index=0,
        )

    def _run_vulkan_finalize(
        self,
        *,
        boxes: np.ndarray,
        selected: np.ndarray,
        proto: np.ndarray,
        input_shape: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        boxes_tensor = self.model.gpu_runner.upload_cpu_mat(self.ncnn.Mat(np.ascontiguousarray(boxes, dtype=np.float32)))
        selected_tensor = self.model.gpu_runner.upload_cpu_mat(
            self.ncnn.Mat(np.ascontiguousarray(selected, dtype=np.float32))
        )
        proto_tensor = self.model.gpu_runner.upload_cpu_mat(self.ncnn.Mat(np.ascontiguousarray(proto, dtype=np.float32)))
        native = self.ncnn.finalize_yolo_segmentation_masks_gpu(
            boxes_tensor,
            selected_tensor,
            proto_tensor,
            int(boxes.shape[0]),
            list(input_shape),
            list(orig_shape),
        )
        return (
            np.asarray(native["boxes"], dtype=np.float32),
            np.asarray(native["masks"], dtype=np.uint8),
        )

    def test_vulkan_finalize_uses_cuda_style_fractional_crop(self) -> None:
        input_shape = (8, 8)
        orig_shape = (8, 8)
        proto = np.zeros((1, 4, 4), dtype=np.float32)
        proto[0, 0, 0] = 1.0

        selected = np.array([[0.9, 0.9, 1.9, 1.9, 0.95, 0.0, 1.0]], dtype=np.float32)
        boxes = np.array([[0.9, 0.9, 1.9, 1.9, 0.95, 0.0]], dtype=np.float32)
        native_boxes, native_masks = self._run_vulkan_finalize(
            boxes=boxes,
            selected=selected,
            proto=proto,
            input_shape=input_shape,
            orig_shape=orig_shape,
        )
        ref_boxes, ref_masks = _reference_finalize_from_selected(
            boxes=boxes,
            selected=selected,
            proto=proto,
            input_shape=input_shape,
            orig_shape=orig_shape,
        )

        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=2e-3)
        np.testing.assert_array_equal(native_masks, ref_masks)

    def test_vulkan_finalize_keeps_input_space_nonempty_masks(self) -> None:
        input_shape = (8, 8)
        orig_shape = (8, 2)
        proto = np.zeros((1, 4, 4), dtype=np.float32)
        proto[0, :, 0] = 1.0

        selected = np.array([[-0.1, 0.1, 1.9, 7.9, 0.95, 0.0, 1.0]], dtype=np.float32)
        boxes = np.array([[0.0, 0.1, 0.0, 7.9, 0.95, 0.0]], dtype=np.float32)
        native_boxes, native_masks = self._run_vulkan_finalize(
            boxes=boxes,
            selected=selected,
            proto=proto,
            input_shape=input_shape,
            orig_shape=orig_shape,
        )
        ref_boxes, ref_masks = _reference_finalize_from_selected(
            boxes=boxes,
            selected=selected,
            proto=proto,
            input_shape=input_shape,
            orig_shape=orig_shape,
        )

        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=2e-3)
        np.testing.assert_array_equal(native_masks, ref_masks)


if __name__ == "__main__":
    unittest.main()
