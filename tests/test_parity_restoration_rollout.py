from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from lada.parity_restoration_rollout import run_candidate_module


class _FakeRestorer(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def _run_profiled_module(
        self,
        module_name: str,
        inputs: dict[str, object],
        *,
        bucket: str | None,
        prefer_gpu_download: bool = False,
    ) -> np.ndarray:
        self.calls.append(
            {
                "module_name": module_name,
                "inputs": inputs,
                "bucket": bucket,
                "prefer_gpu_download": prefer_gpu_download,
            }
        )
        return np.ones((1, 2, 2), dtype=np.float32)


class ParityRestorationRolloutTest(unittest.TestCase):
    def test_run_candidate_module_uses_gpu_download_for_output_frame(self) -> None:
        restorer = _FakeRestorer()
        input_tensor = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

        output = run_candidate_module(
            restorer,
            "output_frame",
            {"in0": input_tensor},
        )

        self.assertEqual((1, 2, 2), output.shape)
        self.assertEqual(1, len(restorer.calls))
        self.assertEqual("output_frame", restorer.calls[0]["module_name"])
        self.assertIsNone(restorer.calls[0]["bucket"])
        self.assertTrue(restorer.calls[0]["prefer_gpu_download"])
        self.assertIsInstance(restorer.calls[0]["inputs"]["in0"], np.ndarray)


if __name__ == "__main__":
    unittest.main()
