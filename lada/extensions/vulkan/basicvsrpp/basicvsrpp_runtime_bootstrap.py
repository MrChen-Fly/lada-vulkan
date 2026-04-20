from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..ncnn_runtime import (
    NcnnVulkanBasicvsrppClipRunner,
    NcnnVulkanModuleRunner,
    ncnn_has_lada_basicvsrpp_clip_runner,
    summarize_ncnn_vulkan_audits,
)

from .basicvsrpp_artifacts import (
    ensure_ncnn_basicvsrpp_modular_artifacts,
)
from .basicvsrpp_runtime_support import (
    resolve_basicvsrpp_runtime_shape,
)
from .runtime_options import RestorationRuntimeFeatures

if TYPE_CHECKING:
    from .basicvsrpp_restorer import (
        NcnnVulkanBasicvsrppMosaicRestorer,
    )


_RECURRENT_RUNTIME_MODULES = (
    "flow_warp",
    "backward_1_deform_align",
    "forward_1_deform_align",
    "backward_2_deform_align",
    "forward_2_deform_align",
    "backward_1_backbone",
    "forward_1_backbone",
    "backward_2_backbone",
    "forward_2_backbone",
    "output_frame",
)
_RECURRENT_CLIP_RUNNER_MODULES = (
    "quarter_downsample",
    "feat_extract",
    "spynet",
    "backward_1_backbone",
    "forward_1_backbone",
    "backward_2_backbone",
    "forward_2_backbone",
    "backward_1_step",
    "forward_1_step",
    "backward_2_step",
    "forward_2_step",
    "output_frame",
)
_SPYNET_FP32_MODULES = frozenset({"spynet", "spynet_patch"})


def _module_fp16_enabled(restorer: "NcnnVulkanBasicvsrppMosaicRestorer", module_name: str) -> bool:
    """Keep SPyNet in fp32 so Vulkan flow inference stays aligned with Torch/CUDA."""
    return bool(restorer.fp16 and module_name not in _SPYNET_FP32_MODULES)


def initialize_modular_runtime(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    checkpoint_path: str,
    *,
    config_path: str | dict | None,
    artifacts_dir: str | Path | None,
    frame_shape: int | tuple[int, int] | list[int],
) -> None:
    """Build all modular ncnn runners and runtime feature flags."""
    restorer.runtime_shape = resolve_basicvsrpp_runtime_shape(frame_shape)
    restorer.artifacts = ensure_ncnn_basicvsrpp_modular_artifacts(
        checkpoint_path,
        config_path=config_path,
        frame_count=restorer.frame_count,
        artifacts_dir=artifacts_dir,
        frame_shape=restorer.runtime_shape.frame_shape,
    )
    restorer.runners = {
        module_name: NcnnVulkanModuleRunner(
            artifacts.param_path,
            artifacts.bin_path,
            fp16=_module_fp16_enabled(restorer, module_name),
            use_vulkan=True,
            num_threads=restorer.num_threads,
        )
        for module_name, artifacts in restorer.artifacts.items()
    }
    restorer.ncnn = next(iter(restorer.runners.values())).ncnn
    restorer.vulkan_audit = summarize_ncnn_vulkan_audits(
        {
            module_name: runner.layer_audit
            for module_name, runner in restorer.runners.items()
        }
    )
    use_vulkan_blend_patch = bool(
        getattr(restorer.ncnn, "has_lada_blend_patch_vulkan", False)
        and callable(getattr(restorer.ncnn, "blend_patch_gpu", None))
    )
    use_vulkan_blend_patch_inplace = bool(
        getattr(restorer.ncnn, "has_lada_blend_patch_vulkan_inplace", False)
        and callable(getattr(restorer.ncnn, "blend_patch_gpu_inplace", None))
    )
    use_vulkan_blend_patch_preprocess = bool(
        getattr(restorer.ncnn, "has_lada_blend_patch_vulkan_preprocess", False)
        and callable(getattr(restorer.ncnn, "blend_patch_gpu_preprocess", None))
    )
    use_vulkan_blend_patch_preprocess_inplace = bool(
        getattr(restorer.ncnn, "has_lada_blend_patch_vulkan_preprocess_inplace", False)
        and callable(getattr(restorer.ncnn, "blend_patch_gpu_preprocess_inplace", None))
    )
    use_vulkan_blend_patch_preprocess_batch_inplace = bool(
        getattr(
            restorer.ncnn,
            "has_lada_blend_patch_vulkan_preprocess_inplace_batch",
            False,
        )
        and callable(
            getattr(restorer.ncnn, "blend_patch_gpu_preprocess_inplace_batch", None)
        )
    )
    use_gpu_blob_bridge = all(
        runner.gpu_runner is not None for runner in restorer.runners.values()
    )
    restorer.frame_preprocess_runner = restorer.runners.get("quarter_downsample")
    use_vulkan_frame_preprocess = bool(
        use_gpu_blob_bridge
        and restorer.frame_preprocess_runner is not None
        and restorer.frame_preprocess_runner.gpu_runner is not None
        and hasattr(restorer.frame_preprocess_runner.gpu_runner, "preprocess_bgr_u8")
    )
    use_vulkan_frame_preprocess_batch = bool(
        use_vulkan_frame_preprocess
        and restorer.frame_preprocess_runner is not None
        and hasattr(restorer.frame_preprocess_runner, "preprocess_bgr_u8_frames")
        and hasattr(restorer.frame_preprocess_runner.gpu_runner, "preprocess_bgr_u8_batch")
    )
    use_fused_restore_clip = "restore_clip" in restorer.runners
    use_recurrent_modular_runtime = all(
        module_name in restorer.runners for module_name in _RECURRENT_RUNTIME_MODULES
    )
    restorer.native_clip_runner = None
    if all(
        module_name in restorer.artifacts
        for module_name in _RECURRENT_CLIP_RUNNER_MODULES
    ) and ncnn_has_lada_basicvsrpp_clip_runner(restorer.ncnn):
        clip_runner_artifacts = {
            module_name: restorer.artifacts[module_name]
            for module_name in _RECURRENT_CLIP_RUNNER_MODULES
        }
        if "spynet_patch" in restorer.artifacts:
            clip_runner_artifacts["spynet_patch"] = restorer.artifacts["spynet_patch"]
        restorer.native_clip_runner = NcnnVulkanBasicvsrppClipRunner(
            clip_runner_artifacts,
            fp16=restorer.fp16,
            num_threads=restorer.num_threads,
            spynet_patch_shape=restorer.runtime_shape.spynet_patch_shape,
            spynet_core_shape=restorer.runtime_shape.spynet_core_shape,
        )
    native_clip_runner_available = restorer.native_clip_runner is not None
    supports_descriptor_restore = bool(
        native_clip_runner_available
        and restorer.native_clip_runner is not None
        and restorer.native_clip_runner.supports_resized_bgr_u8_input
    )
    # Keep the native clip runner available for explicit experiments such as
    # descriptor-native restore, but do not route the default restore() path
    # through it until its outputs match the verified CPU/modular semantics.
    use_native_clip_runner = False
    restorer.runtime_features = RestorationRuntimeFeatures(
        use_native_blob_bridge=use_gpu_blob_bridge,
        use_native_frame_preprocess=use_vulkan_frame_preprocess,
        use_native_frame_preprocess_batch=use_vulkan_frame_preprocess_batch,
        use_native_fused_restore_clip=use_fused_restore_clip,
        use_native_recurrent_runtime=use_recurrent_modular_runtime,
        use_native_clip_runner=use_native_clip_runner,
        supports_descriptor_restore=supports_descriptor_restore,
        use_native_blend_patch=use_vulkan_blend_patch,
        use_native_blend_patch_inplace=use_vulkan_blend_patch_inplace,
        use_native_padded_blend_patch=use_vulkan_blend_patch_preprocess,
        use_native_padded_blend_patch_inplace=use_vulkan_blend_patch_preprocess_inplace,
        use_native_padded_blend_patch_batch_inplace=use_vulkan_blend_patch_preprocess_batch_inplace,
    )
