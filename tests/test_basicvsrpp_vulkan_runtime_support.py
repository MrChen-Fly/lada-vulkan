from __future__ import annotations

from pathlib import Path
import unittest

from lada.restorationpipeline.basicvsrpp_vulkan_runtime_support import (
    get_legacy_modular_artifact_dir_name,
    get_modular_artifact_dir_name,
    resolve_basicvsrpp_runtime_shape,
)


class BasicvsrppVulkanRuntimeSupportTests(unittest.TestCase):
    def test_runtime_shape_resolves_feature_and_spynet_shapes(self) -> None:
        runtime_shape = resolve_basicvsrpp_runtime_shape((320, 448))
        self.assertEqual((320, 448), runtime_shape.frame_shape)
        self.assertEqual((80, 112), runtime_shape.feature_shape)
        self.assertEqual((80, 112), runtime_shape.spynet_patch_shape)
        self.assertEqual((320, 448), runtime_shape.spynet_core_shape)

    def test_shape_specific_artifact_dir_name_is_distinct_from_legacy_dir(self) -> None:
        model_path = Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")
        self.assertEqual(
            get_modular_artifact_dir_name(
                model_path,
                frame_count=5,
                frame_shape=(320, 448),
                revision=16,
            ),
            "lada_mosaic_restoration_model_generic_v1.2.vulkan_modular_5f.320x448_r16",
        )
        self.assertEqual(
            get_legacy_modular_artifact_dir_name(
                model_path,
                frame_count=5,
                revision=16,
            ),
            "lada_mosaic_restoration_model_generic_v1.2.vulkan_modular_5f_r16",
        )


if __name__ == "__main__":
    unittest.main()
