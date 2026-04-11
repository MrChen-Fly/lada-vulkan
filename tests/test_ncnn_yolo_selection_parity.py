from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import nms, ops

from lada.models.basicvsrpp.ncnn_vulkan import import_ncnn_module
from lada.models.yolo.ncnn_vulkan import NcnnVulkanYoloSegmentationModel


def _build_prediction(
    *,
    xywh_boxes: list[tuple[float, float, float, float]],
    class_scores: list[tuple[float, ...]],
    mask_coeffs: list[tuple[float, ...]],
) -> np.ndarray:
    num_boxes = len(xywh_boxes)
    num_classes = len(class_scores[0])
    mask_dim = len(mask_coeffs[0])
    pred = np.zeros((4 + num_classes + mask_dim, num_boxes), dtype=np.float32)
    for box_index, (xywh, scores, coeffs) in enumerate(zip(xywh_boxes, class_scores, mask_coeffs)):
        pred[0:4, box_index] = np.asarray(xywh, dtype=np.float32)
        pred[4 : 4 + num_classes, box_index] = np.asarray(scores, dtype=np.float32)
        pred[4 + num_classes :, box_index] = np.asarray(coeffs, dtype=np.float32)
    return pred


def _reference_selected(
    *,
    pred: np.ndarray,
    input_shape: tuple[int, int],
    orig_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    num_classes: int,
    agnostic_nms: bool,
    classes: list[int] | None = None,
    max_nms: int = 30000,
) -> tuple[np.ndarray, np.ndarray]:
    pred_t = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32)).unsqueeze(0)
    selected = nms.non_max_suppression(
        pred_t,
        conf_threshold,
        iou_threshold,
        classes=classes,
        agnostic=agnostic_nms,
        max_det=max_det,
        nc=num_classes,
        max_nms=max_nms,
        end2end=False,
    )[0].cpu()
    boxes = selected[:, :6].clone()
    if boxes.numel():
        boxes[:, :4] = ops.scale_boxes(input_shape, boxes[:, :4], orig_shape)
    return (
        boxes.numpy().astype(np.float32, copy=False),
        selected.numpy().astype(np.float32, copy=False),
    )


def _reference_postprocess(
    *,
    pred: np.ndarray,
    proto: np.ndarray,
    input_shape: tuple[int, int],
    orig_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    num_classes: int,
    agnostic_nms: bool,
    classes: list[int] | None = None,
    max_nms: int = 30000,
) -> tuple[np.ndarray, np.ndarray]:
    pred_t = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32)).unsqueeze(0)
    proto_t = torch.from_numpy(np.ascontiguousarray(proto, dtype=np.float32))
    selected = nms.non_max_suppression(
        pred_t,
        conf_threshold,
        iou_threshold,
        classes=classes,
        agnostic=agnostic_nms,
        max_det=max_det,
        nc=num_classes,
        max_nms=max_nms,
        end2end=False,
    )[0].cpu()
    if not len(selected):
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, orig_shape[0], orig_shape[1]), dtype=np.uint8),
        )

    masks = _cuda_style_process_mask(
        proto_t,
        selected[:, 6:],
        selected[:, :4],
        input_shape,
    )
    boxes = selected[:, :6].clone()
    boxes[:, :4] = ops.scale_boxes(input_shape, boxes[:, :4], orig_shape)
    keep = masks.sum((-2, -1)) > 0
    boxes = boxes[keep]
    masks = masks[keep].numpy().astype(np.uint8, copy=False)
    return (
        boxes.numpy().astype(np.float32, copy=False),
        np.ascontiguousarray(masks, dtype=np.uint8),
    )


def _cuda_style_process_mask(
    proto: torch.Tensor,
    mask_coeffs: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    input_shape: tuple[int, int],
) -> torch.Tensor:
    channels, mask_h, mask_w = proto.shape
    masks = (mask_coeffs @ proto.float().view(channels, -1)).view(-1, mask_h, mask_w)
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
    return masks.gt_(0.0).byte() * 255


class NcnnYoloSelectionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ncnn = import_ncnn_module()
        if not getattr(cls.ncnn, "has_lada_yolo_seg_postprocess", False):
            raise unittest.SkipTest("Local ncnn runtime does not expose YOLO postprocess bindings.")

        cls.input_shape = (8, 8)
        cls.orig_shape = (8, 8)
        cls.conf_threshold = 0.25
        cls.iou_threshold = 0.45
        cls.max_det = 10
        cls.max_nms = 30000
        cls.num_classes = 2
        cls.proto = np.ones((1, 4, 4), dtype=np.float32)
        cls.pred = _build_prediction(
            xywh_boxes=[
                (4.0, 4.0, 4.0, 4.0),
                (4.2, 4.0, 4.0, 4.0),
                (4.0, 4.0, 4.0, 4.0),
                (1.0, 1.0, 1.0, 1.0),
            ],
            class_scores=[
                (0.95, 0.10),
                (0.80, 0.05),
                (0.05, 0.90),
                (0.20, 0.20),
            ],
            mask_coeffs=[
                (1.0,),
                (1.0,),
                (1.0,),
                (1.0,),
            ],
        )

    def test_cpu_postprocess_matches_ultralytics_nms_when_class_aware(self) -> None:
        native = self.ncnn.postprocess_yolo_segmentation(
            self.pred,
            self.proto,
            list(self.input_shape),
            list(self.orig_shape),
            float(self.conf_threshold),
            float(self.iou_threshold),
            int(self.max_det),
            int(self.num_classes),
            False,
            [],
            int(self.max_nms),
        )
        native_boxes = np.asarray(native["boxes"], dtype=np.float32)
        native_masks = np.asarray(native["masks"], dtype=np.uint8)
        ref_boxes, ref_masks = _reference_postprocess(
            pred=self.pred,
            proto=self.proto,
            input_shape=self.input_shape,
            orig_shape=self.orig_shape,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            num_classes=self.num_classes,
            agnostic_nms=False,
            max_nms=self.max_nms,
        )

        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(native_masks, ref_masks)
        self.assertEqual(native_boxes.shape[0], 2)

    def test_cpu_postprocess_matches_ultralytics_nms_when_agnostic(self) -> None:
        native = self.ncnn.postprocess_yolo_segmentation(
            self.pred,
            self.proto,
            list(self.input_shape),
            list(self.orig_shape),
            float(self.conf_threshold),
            float(self.iou_threshold),
            int(self.max_det),
            int(self.num_classes),
            True,
            [],
            int(self.max_nms),
        )
        native_boxes = np.asarray(native["boxes"], dtype=np.float32)
        native_masks = np.asarray(native["masks"], dtype=np.uint8)
        ref_boxes, ref_masks = _reference_postprocess(
            pred=self.pred,
            proto=self.proto,
            input_shape=self.input_shape,
            orig_shape=self.orig_shape,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            num_classes=self.num_classes,
            agnostic_nms=True,
            max_nms=self.max_nms,
        )

        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(native_masks, ref_masks)
        self.assertEqual(native_boxes.shape[0], 1)

    def test_cpu_postprocess_applies_class_filter_before_agnostic_nms(self) -> None:
        native = self.ncnn.postprocess_yolo_segmentation(
            self.pred,
            self.proto,
            list(self.input_shape),
            list(self.orig_shape),
            float(self.conf_threshold),
            float(self.iou_threshold),
            int(self.max_det),
            int(self.num_classes),
            True,
            [1],
            int(self.max_nms),
        )
        native_boxes = np.asarray(native["boxes"], dtype=np.float32)
        native_masks = np.asarray(native["masks"], dtype=np.uint8)
        ref_boxes, ref_masks = _reference_postprocess(
            pred=self.pred,
            proto=self.proto,
            input_shape=self.input_shape,
            orig_shape=self.orig_shape,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            num_classes=self.num_classes,
            agnostic_nms=True,
            classes=[1],
            max_nms=self.max_nms,
        )

        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(native_masks, ref_masks)
        self.assertEqual(native_boxes.shape[0], 1)
        self.assertAlmostEqual(float(native_boxes[0, 5]), 1.0, places=6)

    def test_cpu_postprocess_truncates_candidates_before_nms_with_max_nms(self) -> None:
        pred = _build_prediction(
            xywh_boxes=[
                (1.5, 1.5, 2.0, 2.0),
                (4.0, 1.5, 2.0, 2.0),
                (6.5, 1.5, 2.0, 2.0),
            ],
            class_scores=[
                (0.95, 0.05),
                (0.90, 0.05),
                (0.85, 0.05),
            ],
            mask_coeffs=[
                (1.0,),
                (1.0,),
                (1.0,),
            ],
        )
        native = self.ncnn.postprocess_yolo_segmentation(
            pred,
            self.proto,
            list(self.input_shape),
            list(self.orig_shape),
            float(self.conf_threshold),
            float(self.iou_threshold),
            int(self.max_det),
            int(self.num_classes),
            False,
            [],
            2,
        )
        native_boxes = np.asarray(native["boxes"], dtype=np.float32)
        native_masks = np.asarray(native["masks"], dtype=np.uint8)
        ref_boxes, ref_masks = _reference_postprocess(
            pred=pred,
            proto=self.proto,
            input_shape=self.input_shape,
            orig_shape=self.orig_shape,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            num_classes=self.num_classes,
            agnostic_nms=False,
            max_nms=2,
        )

        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(native_masks, ref_masks)
        self.assertEqual(native_boxes.shape[0], 2)


class NcnnVulkanYoloSelectionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ncnn = import_ncnn_module()
        if not getattr(cls.ncnn, "has_lada_vulkan_net_runner", False):
            raise unittest.SkipTest("Local ncnn runtime does not expose the Vulkan runner.")

        cls.model_path = Path("model_weights/lada_mosaic_detection_model_v4_fast.pt")
        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Missing YOLO model artifact: {cls.model_path}")

        cls.model = NcnnVulkanYoloSegmentationModel(
            str(cls.model_path),
            imgsz=640,
            fp16=True,
            device_index=0,
        )

        cls.input_shape = (8, 8)
        cls.orig_shape = (8, 8)
        cls.conf_threshold = 0.25
        cls.iou_threshold = 0.45
        cls.max_det = 10
        cls.max_nms = 30000
        cls.num_classes = 2
        cls.proto = np.ones((1, 4, 4), dtype=np.float32)
        cls.pred = _build_prediction(
            xywh_boxes=[
                (4.0, 4.0, 4.0, 4.0),
                (4.2, 4.0, 4.0, 4.0),
                (4.0, 4.0, 4.0, 4.0),
                (1.0, 1.0, 1.0, 1.0),
            ],
            class_scores=[
                (0.95, 0.10),
                (0.80, 0.05),
                (0.05, 0.90),
                (0.20, 0.20),
            ],
            mask_coeffs=[
                (1.0,),
                (1.0,),
                (1.0,),
                (1.0,),
            ],
        )

    def _run_selection_subnet(
        self,
        *,
        agnostic_nms: bool,
        classes: list[int] | None = None,
        max_nms: int | None = None,
        pred: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        active_pred = self.pred if pred is None else pred
        active_max_nms = self.max_nms if max_nms is None else int(max_nms)
        class_flags = np.ones((self.num_classes,), dtype=np.float32) if not classes else np.zeros((self.num_classes,), dtype=np.float32)
        if classes:
            for class_index in classes:
                if 0 <= class_index < self.num_classes:
                    class_flags[class_index] = 1.0
        config = np.asarray(
            [
                self.conf_threshold,
                self.iou_threshold,
                float(self.max_det),
                float(active_max_nms),
                float(self.num_classes),
                1.0 if agnostic_nms else 0.0,
                float(self.input_shape[0]),
                float(self.input_shape[1]),
                float(self.orig_shape[0]),
                float(self.orig_shape[1]),
                *class_flags.tolist(),
            ],
            dtype=np.float32,
        )
        outputs = self.model.postprocess_runner.run_many_to_cpu(
            {
                "pred": self.ncnn.Mat(np.ascontiguousarray(active_pred, dtype=np.float32)),
                "proto": self.ncnn.Mat(np.ascontiguousarray(self.proto, dtype=np.float32)),
                "config": self.ncnn.Mat(config),
            },
            ["boxes", "selected", "count"],
        )
        count = int(np.asarray(outputs["count"], dtype=np.float32).reshape(-1)[0])
        boxes = np.asarray(outputs["boxes"], dtype=np.float32)[:count]
        selected = np.asarray(outputs["selected"], dtype=np.float32)[:count]
        return boxes, selected, count

    def test_vulkan_selection_subnet_matches_ultralytics_nms_when_class_aware(self) -> None:
        native_boxes, native_selected, native_count = self._run_selection_subnet(agnostic_nms=False)
        ref_boxes, ref_selected = _reference_selected(
            pred=self.pred,
            input_shape=self.input_shape,
            orig_shape=self.orig_shape,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            num_classes=self.num_classes,
            agnostic_nms=False,
            classes=None,
            max_nms=self.max_nms,
        )

        self.assertEqual(native_count, ref_boxes.shape[0])
        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=2e-3)
        np.testing.assert_allclose(native_selected, ref_selected, rtol=0.0, atol=2e-3)

    def test_vulkan_selection_subnet_matches_ultralytics_nms_when_agnostic(self) -> None:
        native_boxes, native_selected, native_count = self._run_selection_subnet(agnostic_nms=True)
        ref_boxes, ref_selected = _reference_selected(
            pred=self.pred,
            input_shape=self.input_shape,
            orig_shape=self.orig_shape,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            num_classes=self.num_classes,
            agnostic_nms=True,
            classes=None,
            max_nms=self.max_nms,
        )

        self.assertEqual(native_count, ref_boxes.shape[0])
        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=2e-3)
        np.testing.assert_allclose(native_selected, ref_selected, rtol=0.0, atol=2e-3)

    def test_vulkan_selection_subnet_applies_class_filter_before_agnostic_nms(self) -> None:
        native_boxes, native_selected, native_count = self._run_selection_subnet(
            agnostic_nms=True,
            classes=[1],
        )
        ref_boxes, ref_selected = _reference_selected(
            pred=self.pred,
            input_shape=self.input_shape,
            orig_shape=self.orig_shape,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            num_classes=self.num_classes,
            agnostic_nms=True,
            classes=[1],
            max_nms=self.max_nms,
        )

        self.assertEqual(native_count, ref_boxes.shape[0])
        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=2e-3)
        np.testing.assert_allclose(native_selected, ref_selected, rtol=0.0, atol=2e-3)

    def test_vulkan_selection_subnet_truncates_candidates_before_nms_with_max_nms(self) -> None:
        pred = _build_prediction(
            xywh_boxes=[
                (1.5, 1.5, 2.0, 2.0),
                (4.0, 1.5, 2.0, 2.0),
                (6.5, 1.5, 2.0, 2.0),
            ],
            class_scores=[
                (0.95, 0.05),
                (0.90, 0.05),
                (0.85, 0.05),
            ],
            mask_coeffs=[
                (1.0,),
                (1.0,),
                (1.0,),
            ],
        )
        native_boxes, native_selected, native_count = self._run_selection_subnet(
            agnostic_nms=False,
            max_nms=2,
            pred=pred,
        )
        ref_boxes, ref_selected = _reference_selected(
            pred=pred,
            input_shape=self.input_shape,
            orig_shape=self.orig_shape,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            num_classes=self.num_classes,
            agnostic_nms=False,
            classes=None,
            max_nms=2,
        )

        self.assertEqual(native_count, ref_boxes.shape[0])
        np.testing.assert_allclose(native_boxes, ref_boxes, rtol=0.0, atol=2e-3)
        np.testing.assert_allclose(native_selected, ref_selected, rtol=0.0, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
