from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from lada.models.yolo.ncnn_vulkan import NcnnVulkanYoloSegmentationModel


class NcnnVulkanYoloBatchTests(unittest.TestCase):
    def test_native_fused_subnet_batch_normalizes_inputs_before_dispatch(self) -> None:
        model = object.__new__(NcnnVulkanYoloSegmentationModel)
        captured: dict[str, object] = {}

        def fake_native_batch(*args):
            captured["args"] = args
            return [
                {
                    "boxes": [[1.0, 2.0, 3.0, 4.0, 0.9, 0.0]],
                    "masks": [[[1, 0], [0, 1]]],
                },
                {
                    "boxes": [[5.0, 6.0, 7.0, 8.0, 0.8, 0.0]],
                    "masks": [[[0, 1], [1, 0]]],
                },
            ]

        model.gpu_runner = SimpleNamespace(run_yolo_segmentation_subnet_batch=fake_native_batch)
        model.postprocess_runner = object()
        model.input_name = "images"
        model.output_names = ("out0", "out1")
        model.active_input_shape = (640, 640)
        model.args = SimpleNamespace(
            conf=0.25,
            iou=0.7,
            max_det=300,
            agnostic_nms=False,
            classes=None,
        )
        model.names = ["mosaic"]
        model.max_nms = 30000
        conversions: list[object] = []

        def fake_to_ncnn_input_mat(value: object) -> object:
            conversions.append(value)
            return f"mat:{value}"

        model._to_ncnn_input_mat = fake_to_ncnn_input_mat  # type: ignore[method-assign]

        processed_batch = model._run_native_fused_subnet_batch(
            ["frame-0", "frame-1"],
            [(10, 20), (11, 21)],
        )

        self.assertEqual(conversions, ["frame-0", "frame-1"])
        self.assertEqual(captured["args"][0], ["mat:frame-0", "mat:frame-1"])
        self.assertEqual(captured["args"][1], "images")
        self.assertEqual(captured["args"][4], [640, 640])
        self.assertEqual(captured["args"][5], [[10, 20], [11, 21]])
        self.assertEqual(processed_batch[0]["boxes"].dtype, np.float32)
        self.assertEqual(processed_batch[0]["masks"].dtype, np.uint8)
        self.assertEqual(processed_batch[1]["boxes"][0, 0], 5.0)

    def test_fused_subnet_batch_prefers_native_batch_binding_when_available(self) -> None:
        model = object.__new__(NcnnVulkanYoloSegmentationModel)
        model.gpu_runner = SimpleNamespace(run_yolo_segmentation_subnet_batch=object())
        model._run_native_fused_subnet_batch = lambda batch, shapes: [  # type: ignore[method-assign]
            {
                "boxes": np.full((1, 6), len(batch), dtype=np.float32),
                "masks": np.full((1, 2, 2), len(shapes), dtype=np.uint8),
            }
        ]
        model._run_fused_subnet = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("single-frame fallback should not run when native batch binding is available")
        )

        processed_batch = model._run_fused_subnet_batch(["frame-0"], [(10, 20)])

        self.assertEqual(processed_batch[0]["boxes"][0, 0], 1.0)
        self.assertEqual(processed_batch[0]["masks"][0, 0, 0], 1)

    def test_fused_subnet_batch_falls_back_to_single_frame_semantics_without_native_binding(self) -> None:
        model = object.__new__(NcnnVulkanYoloSegmentationModel)
        model.gpu_runner = SimpleNamespace()

        prepared_batch = ["frame-0", "frame-1", "frame-2"]
        orig_shapes = [(10, 20), (11, 21), (12, 22)]
        calls: list[tuple[object, tuple[int, int]]] = []

        def fake_run_fused_subnet(input_frame: object, orig_shape: tuple[int, int]) -> dict[str, np.ndarray]:
            calls.append((input_frame, orig_shape))
            marker = len(calls)
            return {
                "boxes": np.full((1, 6), marker, dtype=np.float32),
                "masks": np.full((1, 2, 2), marker, dtype=np.uint8),
            }

        model._run_fused_subnet = fake_run_fused_subnet  # type: ignore[method-assign]

        processed_batch = model._run_fused_subnet_batch(prepared_batch, orig_shapes)

        self.assertEqual(calls, list(zip(prepared_batch, orig_shapes)))
        self.assertEqual(processed_batch[0]["boxes"][0, 0], 1.0)
        self.assertEqual(processed_batch[1]["boxes"][0, 0], 2.0)
        self.assertEqual(processed_batch[2]["boxes"][0, 0], 3.0)

    def test_fused_subnet_batch_requires_matching_shape_count(self) -> None:
        model = object.__new__(NcnnVulkanYoloSegmentationModel)
        model._run_fused_subnet = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "one original shape per input frame"):
            model._run_fused_subnet_batch(["frame-0"], [])


if __name__ == "__main__":
    unittest.main()
