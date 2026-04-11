import unittest

from lada.extensions.vulkan import basicvsrpp_ncnn_runtime
from lada.extensions.vulkan import basicvsrpp_artifacts
from lada.extensions.vulkan import basicvsrpp_blend
from lada.extensions.vulkan import basicvsrpp_common
from lada.extensions.vulkan import basicvsrpp_cpu_extractor
from lada.extensions.vulkan import basicvsrpp_export
from lada.extensions.vulkan import basicvsrpp_io
from lada.extensions.vulkan import basicvsrpp_recurrent_runtime
from lada.extensions.vulkan import basicvsrpp_restore_paths
from lada.extensions.vulkan import basicvsrpp_restorer
from lada.extensions.vulkan import basicvsrpp_runtime_bootstrap
from lada.extensions.vulkan import basicvsrpp_runtime_support
from lada.extensions.vulkan import yolo_ncnn_runtime
from lada.extensions.vulkan import yolo_runtime_support
from lada.restorationpipeline import basicvsrpp_vulkan_artifacts as artifacts_shim
from lada.restorationpipeline import basicvsrpp_vulkan_blend as blend_shim
from lada.restorationpipeline import basicvsrpp_vulkan_common as common_shim
from lada.restorationpipeline import basicvsrpp_vulkan_cpu_extractor as cpu_extractor_shim
from lada.restorationpipeline import basicvsrpp_vulkan_export as export_shim
from lada.restorationpipeline import basicvsrpp_vulkan_io as io_shim
from lada.restorationpipeline import basicvsrpp_vulkan_recurrent_runtime as recurrent_runtime_shim
from lada.restorationpipeline import basicvsrpp_vulkan_restore_paths as restore_paths_shim
from lada.restorationpipeline import basicvsrpp_vulkan_restorer as restorer_shim
from lada.restorationpipeline import basicvsrpp_vulkan_runtime_bootstrap as bootstrap_shim
from lada.restorationpipeline import basicvsrpp_vulkan_runtime_support as support_shim
from lada.models.basicvsrpp import ncnn_vulkan as basicvsrpp_shim
from lada.models.yolo import ncnn_vulkan as yolo_shim
from lada.models.yolo import ncnn_vulkan_runtime_support as yolo_support_shim


class VulkanRuntimeShimTests(unittest.TestCase):
    def test_basicvsrpp_runtime_shim_reexports_extension_symbols(self):
        self.assertIs(
            basicvsrpp_shim.import_ncnn_module,
            basicvsrpp_ncnn_runtime.import_ncnn_module,
        )
        self.assertIs(
            basicvsrpp_shim.NcnnVulkanModuleRunner,
            basicvsrpp_ncnn_runtime.NcnnVulkanModuleRunner,
        )

    def test_yolo_runtime_shim_reexports_extension_symbols(self):
        self.assertIs(
            yolo_shim.NcnnVulkanYoloSegmentationModel,
            yolo_ncnn_runtime.NcnnVulkanYoloSegmentationModel,
        )

    def test_yolo_runtime_support_shim_reexports_extension_symbols(self):
        self.assertIs(
            yolo_support_shim.normalize_runtime_imgsz,
            yolo_runtime_support.normalize_runtime_imgsz,
        )
        self.assertIs(
            yolo_support_shim.resolve_letterbox_output_shape,
            yolo_runtime_support.resolve_letterbox_output_shape,
        )

    def test_restoration_common_shim_reexports_extension_symbols(self):
        self.assertIs(common_shim.NcnnArtifacts, basicvsrpp_common.NcnnArtifacts)
        self.assertEqual(
            common_shim._MODULAR_FRAME_COUNT,
            basicvsrpp_common._MODULAR_FRAME_COUNT,
        )

    def test_restoration_runtime_support_shim_reexports_extension_symbols(self):
        self.assertIs(
            support_shim.resolve_basicvsrpp_runtime_shape,
            basicvsrpp_runtime_support.resolve_basicvsrpp_runtime_shape,
        )
        self.assertIs(
            support_shim.BasicvsrppRuntimeShape,
            basicvsrpp_runtime_support.BasicvsrppRuntimeShape,
        )

    def test_restoration_runtime_bootstrap_shim_reexports_extension_symbols(self):
        self.assertIs(
            bootstrap_shim.initialize_modular_runtime,
            basicvsrpp_runtime_bootstrap.initialize_modular_runtime,
        )

    def test_restoration_artifacts_shim_reexports_extension_symbols(self):
        self.assertIs(
            artifacts_shim.ensure_ncnn_basicvsrpp_modular_artifacts,
            basicvsrpp_artifacts.ensure_ncnn_basicvsrpp_modular_artifacts,
        )

    def test_restoration_blend_shim_reexports_extension_symbols(self):
        self.assertIs(blend_shim.blend_patch, basicvsrpp_blend.blend_patch)
        self.assertIs(
            blend_shim.blend_patch_padded_batch,
            basicvsrpp_blend.blend_patch_padded_batch,
        )

    def test_restoration_cpu_extractor_shim_reexports_extension_symbols(self):
        self.assertIs(
            cpu_extractor_shim.runtime_value_to_numpy,
            basicvsrpp_cpu_extractor.runtime_value_to_numpy,
        )

    def test_restoration_io_shim_reexports_extension_symbols(self):
        self.assertIs(io_shim._frame_to_chw_float32, basicvsrpp_io._frame_to_chw_float32)
        self.assertIs(
            io_shim._build_output_frame_inputs,
            basicvsrpp_io._build_output_frame_inputs,
        )

    def test_restoration_export_shim_reexports_extension_symbols(self):
        self.assertIs(
            export_shim._build_window_indices,
            basicvsrpp_export._build_window_indices,
        )
        self.assertIs(
            export_shim._build_modular_export_specs,
            basicvsrpp_export._build_modular_export_specs,
        )

    def test_restoration_recurrent_runtime_shim_reexports_extension_symbols(self):
        self.assertIs(
            recurrent_runtime_shim.run_flow_warp,
            basicvsrpp_recurrent_runtime.run_flow_warp,
        )
        self.assertIs(
            recurrent_runtime_shim.run_propagate_step,
            basicvsrpp_recurrent_runtime.run_propagate_step,
        )

    def test_restoration_restore_paths_shim_reexports_extension_symbols(self):
        self.assertIs(restore_paths_shim.restore, basicvsrpp_restore_paths.restore)
        self.assertIs(
            restore_paths_shim.restore_cropped_clip_frames,
            basicvsrpp_restore_paths.restore_cropped_clip_frames,
        )

    def test_restoration_restorer_shim_reexports_extension_symbols(self):
        self.assertIs(
            restorer_shim.NcnnVulkanBasicvsrppMosaicRestorer,
            basicvsrpp_restorer.NcnnVulkanBasicvsrppMosaicRestorer,
        )


if __name__ == "__main__":
    unittest.main()
