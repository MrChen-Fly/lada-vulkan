from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from lada.models.basicvsrpp.inference import load_model
from lada.models.basicvsrpp.vulkan_runtime import get_basicvsrpp_generator
from lada.restorationpipeline.basicvsrpp_vulkan_restorer import (
    NcnnVulkanBasicvsrppMosaicRestorer,
)


def _build_smooth_pair() -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(0.0, 1.0, 64, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, :]
    ref = np.stack(
        [
            x * 0.7 + y * 0.2,
            x * 0.3 + y * 0.6,
            x * 0.1 + y * 0.8,
        ],
        axis=0,
    ).astype(np.float32)
    supp = np.stack(
        [
            np.roll(ref[0], shift=1, axis=1),
            np.roll(ref[1], shift=1, axis=0),
            np.roll(ref[2], shift=1, axis=1),
        ],
        axis=0,
    ).astype(np.float32)
    return np.ascontiguousarray(ref), np.ascontiguousarray(supp)


class BasicvsrppVulkanSpynetPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_path = Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")
        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Missing restoration model artifact: {cls.model_path}")

        cls.restorer = NcnnVulkanBasicvsrppMosaicRestorer(
            str(cls.model_path),
            config_path=None,
            fp16=True,
            artifacts_dir=None,
        )
        if "spynet_patch" not in cls.restorer.runners:
            raise unittest.SkipTest("Shape-specific SPyNet patch runner is unavailable.")

        torch_model = load_model(
            config=None,
            checkpoint_path=str(cls.model_path),
            device=torch.device("cpu"),
            fp16=False,
        )
        cls.generator = get_basicvsrpp_generator(torch_model).eval()

    @classmethod
    def tearDownClass(cls) -> None:
        restorer = getattr(cls, "restorer", None)
        if restorer is not None:
            restorer.release_cached_memory()

    def test_fp16_vulkan_spynet_patch_matches_torch(self) -> None:
        ref, supp = _build_smooth_pair()
        flow = np.asarray(
            self.restorer.run_spynet(ref, supp, prefer_gpu_download=True),
            dtype=np.float32,
        )

        with torch.no_grad():
            torch_flow = (
                self.generator.spynet(
                    torch.from_numpy(ref).unsqueeze(0),
                    torch.from_numpy(supp).unsqueeze(0),
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

        diff = np.abs(flow - np.asarray(torch_flow, dtype=np.float32))
        self.assertLess(float(diff.mean()), 1e-3)
        self.assertLess(float(diff.max()), 1e-2)


if __name__ == "__main__":
    unittest.main()
