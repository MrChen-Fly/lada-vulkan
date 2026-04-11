from __future__ import annotations

from unittest import mock
import unittest

from lada.restorationpipeline.basicvsrpp_vulkan_restorer import (
    NcnnVulkanBasicvsrppMosaicRestorer,
)


class BasicvsrppVulkanRestorerTests(unittest.TestCase):
    def test_runtime_scheduling_uses_safe_detector_batch_size(self) -> None:
        with mock.patch.object(
            NcnnVulkanBasicvsrppMosaicRestorer,
            "_initialize_modular_runtime",
            autospec=True,
            return_value=None,
        ):
            restorer = NcnnVulkanBasicvsrppMosaicRestorer("dummy.pth", fp16=False)

        self.assertEqual(1, restorer.get_runtime_scheduling_options().detector_batch_size)


if __name__ == "__main__":
    unittest.main()
