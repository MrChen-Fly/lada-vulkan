import unittest

import numpy as np
from ultralytics.engine.results import Results

from lada.models.yolo.runtime_results import build_native_yolo_result
from lada.utils.ultralytics_utils import (
    convert_yolo_box,
    convert_yolo_boxes,
)


class NativeYoloResultTests(unittest.TestCase):
    def test_native_boxes_preserve_float_coordinates_but_public_box_conversion_stays_integral(self):
        orig_img = np.zeros((10, 20, 3), dtype=np.uint8)
        boxes = np.array([[1.75, 2.25, 8.9, 9.6, 0.5, 4.0]], dtype=np.float32)
        masks = np.zeros((1, 10, 20), dtype=np.uint8)

        result = build_native_yolo_result(
            orig_img=orig_img,
            names={4: "mosaic"},
            boxes=boxes,
            masks=masks,
        )

        self.assertIsInstance(result, Results)
        self.assertTrue(np.allclose(result.boxes.data[0, :4].cpu().numpy(), boxes[0, :4]))
        self.assertEqual((2, 1, 9, 8), convert_yolo_box(result.boxes[0], orig_img.shape))
        self.assertEqual([(2, 1, 9, 8)], convert_yolo_boxes(result.boxes[0], orig_img.shape))


if __name__ == "__main__":
    unittest.main()
