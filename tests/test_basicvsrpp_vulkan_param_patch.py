from __future__ import annotations

from pathlib import Path
import unittest

from lada.models.basicvsrpp.vulkan_param_patch import _patch_spynet_output_tail


class BasicvsrppVulkanParamPatchTests(unittest.TestCase):
    def test_patch_spynet_output_tail_rewrites_resize_back_suffix(self) -> None:
        layer_lines = [
            "BinaryOp                 add_49                   2 1 196 221 222 0=0",
            "Interp                   upsample_72              1 1 222 223 0=2 3=16 4=16 6=0",
            "Split                    splitncnn_17             1 2 223 224 225",
            "Crop                     select_0                 1 1 224 226 -23310=1,1 -23311=1,0 -23309=1,0",
            "Squeeze                  squeeze_107              1 1 226 227 -23303=1,0",
            "BinaryOp                 mul_50                   1 1 227 228 0=2 1=1 2=0.5",
            "Split                    splitncnn_18             1 2 228 229 230",
            "Reshape                  reshape_103              1 1 230 231 0=16 1=16 2=1",
            "CopyTo                   slice_copy_0             2 1 225 231 232 -23311=1,0 -23309=1,0",
            "Reshape                  reshape_104              1 1 229 233 0=16 1=16 2=1",
            "CopyTo                   slice_copy_1             2 1 232 233 234 -23311=1,0 -23309=1,0",
            "Split                    splitncnn_19             1 2 234 235 236",
            "Crop                     select_1                 1 1 235 237 -23310=1,2 -23311=1,0 -23309=1,1",
            "Squeeze                  squeeze_108              1 1 237 238 -23303=1,0",
            "BinaryOp                 mul_51                   1 1 238 239 0=2 1=1 2=0.5",
            "Split                    splitncnn_20             1 2 239 240 241",
            "Reshape                  reshape_105              1 1 241 242 0=16 1=16 2=1",
            "CopyTo                   slice_copy_2             2 1 236 242 243 -23311=1,0 -23309=1,1",
            "Reshape                  reshape_106              1 1 240 244 0=16 1=16 2=1",
            "CopyTo                   slice_copy_3             2 1 243 244 out0 -23311=1,0 -23309=1,1",
        ]

        patched_lines, changed = _patch_spynet_output_tail(
            Path("spynet_patch.ncnn.param"),
            layer_lines,
        )

        self.assertTrue(changed)
        self.assertEqual(layer_lines[:2], patched_lines[:2])
        self.assertIn("Split", patched_lines[2])
        self.assertIn("spynet_output_split", patched_lines[2])
        self.assertIn("spynet_output_split_x", patched_lines[2])
        self.assertIn("spynet_output_split_y", patched_lines[2])
        self.assertIn("Concat", patched_lines[-1])
        self.assertIn("spynet_output_concat", patched_lines[-1])
        self.assertIn("spynet_output_x_scaled", patched_lines[-1])
        self.assertIn("spynet_output_y_scaled", patched_lines[-1])
        self.assertIn("out0", patched_lines[-1])
        self.assertFalse(any("CopyTo" in line for line in patched_lines))
        self.assertFalse(any("Squeeze" in line for line in patched_lines))

    def test_patch_spynet_output_tail_skips_already_vulkan_safe_tail(self) -> None:
        layer_lines = [
            "BinaryOp                 add_49                   2 1 194 219 out0 0=0",
        ]

        patched_lines, changed = _patch_spynet_output_tail(
            Path("spynet_patch.ncnn.param"),
            layer_lines,
        )

        self.assertFalse(changed)
        self.assertEqual(layer_lines, patched_lines)


if __name__ == "__main__":
    unittest.main()
