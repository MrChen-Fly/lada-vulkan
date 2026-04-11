from __future__ import annotations

from pathlib import Path
import unittest

from lada.models.yolo.ncnn_vulkan_runtime_support import (
    get_precision_artifact_dir,
    get_legacy_precision_artifact_dir,
    resolve_letterbox_output_shape,
)


class NcnnYoloRuntimeSupportTests(unittest.TestCase):
    def test_rect_letterbox_shape_matches_16_by_9_video(self) -> None:
        self.assertEqual(
            resolve_letterbox_output_shape(
                (720, 1280),
                target_shape=(640, 640),
                stride=32,
                auto=True,
            ),
            (384, 640),
        )

    def test_fixed_letterbox_shape_keeps_square_target(self) -> None:
        self.assertEqual(
            resolve_letterbox_output_shape(
                (720, 1280),
                target_shape=(640, 640),
                stride=32,
                auto=False,
            ),
            (640, 640),
        )

    def test_shape_specific_artifact_dir_is_distinct_from_legacy_dir(self) -> None:
        model_path = Path("model_weights/lada_mosaic_detection_model_v4_fast.pt")
        self.assertEqual(
            get_precision_artifact_dir(model_path, fp16=False, imgsz=(384, 640)).name,
            "lada_mosaic_detection_model_v4_fast.fp32.384x640_ncnn_model",
        )
        self.assertEqual(
            get_legacy_precision_artifact_dir(model_path, fp16=False).name,
            "lada_mosaic_detection_model_v4_fast.fp32_ncnn_model",
        )


if __name__ == "__main__":
    unittest.main()
