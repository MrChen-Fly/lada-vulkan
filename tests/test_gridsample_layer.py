import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lada.models.basicvsrpp.ncnn_vulkan import (
    NcnnVulkanModuleRunner,
    import_ncnn_module,
    ncnn_has_lada_gridsample_layer,
)


_GRIDSAMPLE_PARAM = """7767517
3 3
Input                    in0                      0 1 in0
Input                    in1                      0 1 in1
lada.GridSample          gridsample               2 1 in0 in1 out0 0=1 1=2 2=1 3=0
"""


class GridSampleLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ncnn = import_ncnn_module()
        if not ncnn_has_lada_gridsample_layer(cls.ncnn):
            raise unittest.SkipTest("Local ncnn runtime does not expose lada.GridSample.")

    def _build_runner(self, *, use_vulkan: bool) -> NcnnVulkanModuleRunner:
        temp_dir = tempfile.TemporaryDirectory(prefix="lada-gridsample-test-")
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        param_path = root / "gridsample.param"
        bin_path = root / "gridsample.bin"
        param_path.write_text(_GRIDSAMPLE_PARAM, encoding="utf-8")
        bin_path.write_bytes(b"")
        return NcnnVulkanModuleRunner(
            param_path,
            bin_path,
            fp16=False,
            use_vulkan=use_vulkan,
        )

    @staticmethod
    def _build_inputs(height: int = 4, width: int = 5) -> tuple[np.ndarray, np.ndarray]:
        image = np.linspace(-1.0, 1.0, num=3 * height * width, dtype=np.float32).reshape(3, height, width)
        grid = np.zeros((height, width, 2), dtype=np.float32)
        xs = np.linspace(-1.2, 1.2, num=width, dtype=np.float32)
        ys = np.linspace(-1.1, 1.1, num=height, dtype=np.float32)
        for y, grid_y in enumerate(ys):
            for x, grid_x in enumerate(xs):
                grid[y, x, 0] = grid_x
                grid[y, x, 1] = grid_y
        return image, grid

    @staticmethod
    def _torch_reference(image: np.ndarray, grid: np.ndarray) -> np.ndarray:
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        grid_tensor = torch.from_numpy(grid).unsqueeze(0)
        return (
            F.grid_sample(
                image_tensor,
                grid_tensor,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .numpy()
        )

    def test_cpu_gridsample_matches_torch(self):
        image, grid = self._build_inputs()
        runner = self._build_runner(use_vulkan=False)
        output = runner.run({"in0": image, "in1": grid})
        reference = self._torch_reference(image, grid)

        self.assertFalse(np.isnan(output).any(), "NCNN CPU GridSample produced NaN values.")
        np.testing.assert_allclose(output, reference, rtol=1e-4, atol=1e-4)

    def test_vulkan_gridsample_matches_torch(self):
        image, grid = self._build_inputs(height=8, width=8)
        runner = self._build_runner(use_vulkan=True)
        output = runner.run_gpu_download({"in0": image, "in1": grid})
        reference = self._torch_reference(image, grid)

        self.assertFalse(np.isnan(output).any(), "NCNN Vulkan GridSample produced NaN values.")
        np.testing.assert_allclose(output, reference, rtol=1e-3, atol=1e-3)

    @unittest.expectedFailure
    def test_vulkan_gridsample_unaligned_grid_upload_known_issue(self):
        image, grid = self._build_inputs()
        runner = self._build_runner(use_vulkan=True)
        output = runner.run_gpu_download({"in0": image, "in1": grid})
        reference = self._torch_reference(image, grid)

        self.assertFalse(np.isnan(output).any(), "NCNN Vulkan GridSample produced NaN values.")
        np.testing.assert_allclose(output, reference, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
