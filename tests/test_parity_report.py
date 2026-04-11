from __future__ import annotations

import unittest

import numpy as np
import torch

from lada.parity_report import quantize_image_output, quantize_unit_interval_output


class ParityReportTest(unittest.TestCase):
    def test_quantize_unit_interval_output_squeezes_batch_dim(self) -> None:
        value = torch.tensor(
            [[[[0.0, 0.5], [1.0, 1.2]], [[0.25, -0.1], [0.75, 0.9]]]],
            dtype=torch.float32,
        )

        quantized = quantize_unit_interval_output(value)

        self.assertEqual((2, 2, 2), quantized.shape)
        self.assertEqual(np.uint8, quantized.dtype)
        self.assertTrue(
            np.array_equal(
                quantized,
                np.array(
                    [[[0, 128], [255, 255]], [[64, 0], [191, 230]]],
                    dtype=np.uint8,
                ),
            )
        )

    def test_quantize_image_output_preserves_uint8_image_domain(self) -> None:
        value = np.array(
            [[[0, 10, 255], [32, 64, 96]]],
            dtype=np.uint8,
        )

        quantized = quantize_image_output(value)

        self.assertEqual(np.uint8, quantized.dtype)
        self.assertTrue(np.array_equal(value, quantized))

    def test_quantize_image_output_accepts_float_image_domain(self) -> None:
        value = np.array(
            [[[0.49, 127.6, 300.0], [-5.0, 128.4, 254.5]]],
            dtype=np.float32,
        )

        quantized = quantize_image_output(value)

        self.assertEqual(np.uint8, quantized.dtype)
        self.assertTrue(
            np.array_equal(
                quantized,
                np.array(
                    [[[0, 128, 255], [0, 128, 254]]],
                    dtype=np.uint8,
                ),
            )
        )

    def test_quantize_image_output_scales_unit_interval_images(self) -> None:
        value = torch.tensor(
            [[[[0.0, 0.5], [1.0, 1.2]], [[0.25, -0.1], [0.75, 0.9]]]],
            dtype=torch.float32,
        )

        quantized = quantize_image_output(value)

        self.assertEqual((2, 2, 2), quantized.shape)
        self.assertEqual(np.uint8, quantized.dtype)
        self.assertTrue(
            np.array_equal(
                quantized,
                np.array(
                    [[[0, 128], [255, 255]], [[64, 0], [191, 230]]],
                    dtype=np.uint8,
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
