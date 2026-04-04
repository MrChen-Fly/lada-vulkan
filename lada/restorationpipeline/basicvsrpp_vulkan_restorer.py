from __future__ import annotations

import gc
from pathlib import Path

import torch

from lada.compute_targets import UnsupportedComputeTargetError
from lada.models.basicvsrpp.ncnn_vulkan import (
    NcnnVulkanBasicvsrppClipRunner,
    NcnnVulkanModuleRunner,
    ncnn_has_lada_basicvsrpp_clip_runner,
    summarize_ncnn_vulkan_audits,
)
from lada.utils import Image, ImageTensor

from .basicvsrpp_vulkan_artifacts import ensure_ncnn_basicvsrpp_modular_artifacts
from .basicvsrpp_vulkan_blend import (
    blend_patch as run_blend_patch,
    blend_patch_padded as run_blend_patch_padded,
    blend_patch_padded_batch as run_blend_patch_padded_batch,
)
from .basicvsrpp_vulkan_common import _MODULAR_FRAME_COUNT
from .basicvsrpp_vulkan_restore_paths import (
    restore as run_restore,
    restore_cropped_clip_frames as run_restore_cropped_clip_frames,
)
from .runtime_options import (
    BasicvsrppVulkanRuntimeFeatures,
    RestorationSchedulingOptions,
)
from .runtime_profiling import WallClockProfiler

_RECURRENT_RUNTIME_MODULES = (
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
_RECURRENT_CLIP_RUNNER_MODULES = (
    "quarter_downsample",
    "feat_extract",
    "spynet",
    *_RECURRENT_RUNTIME_MODULES,
)
_STREAM_RESTORE_CHUNK_SIZE = 60


class NcnnVulkanBasicvsrppMosaicRestorer:
    """BasicVSR++ Vulkan restorer with fixed-size Vulkan subgraphs."""

    runtime = "vulkan"
    torch_device = None

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str | dict | None = None,
        *,
        fp16: bool = False,
        frame_count: int = _MODULAR_FRAME_COUNT,
        artifacts_dir: str | Path | None = None,
        num_threads: int = 1,
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.frame_count = frame_count
        self.dtype = torch.float16 if fp16 else torch.float32
        self.fp16 = fp16
        self.artifacts_dir = artifacts_dir
        self.num_threads = max(int(num_threads), 1)
        self.profiler = WallClockProfiler()
        self.last_profile: dict[str, float | int] = {}
        self._native_clip_profile_snapshot: dict[str, float | int] = {}
        self.vulkan_audit: dict[str, object] = {}
        self.runtime_features = BasicvsrppVulkanRuntimeFeatures()
        self.runtime_scheduling = RestorationSchedulingOptions(
            stream_restore_chunk_size=_STREAM_RESTORE_CHUNK_SIZE,
            detector_batch_size=6,
        )
        self.native_clip_runner: NcnnVulkanBasicvsrppClipRunner | None = None
        self.frame_preprocess_runner: NcnnVulkanModuleRunner | None = None

        if self.frame_count != _MODULAR_FRAME_COUNT:
            raise UnsupportedComputeTargetError(
                "Vulkan BasicVSR++ uses the modular 5-frame export graph. "
                f"Expected frame_count={_MODULAR_FRAME_COUNT}, got {self.frame_count}."
            )

        self._initialize_modular_runtime(
            checkpoint_path,
            config_path=config_path,
            artifacts_dir=artifacts_dir,
        )

    def _initialize_modular_runtime(
        self,
        checkpoint_path: str,
        *,
        config_path: str | dict | None,
        artifacts_dir: str | Path | None,
    ) -> None:
        self.artifacts = ensure_ncnn_basicvsrpp_modular_artifacts(
            checkpoint_path,
            config_path=config_path,
            frame_count=self.frame_count,
            artifacts_dir=artifacts_dir,
        )
        self.runners = {
            module_name: NcnnVulkanModuleRunner(
                artifacts.param_path,
                artifacts.bin_path,
                fp16=self.fp16,
                use_vulkan=True,
                num_threads=self.num_threads,
            )
            for module_name, artifacts in self.artifacts.items()
        }
        self.ncnn = next(iter(self.runners.values())).ncnn
        self.vulkan_audit = summarize_ncnn_vulkan_audits(
            {
                module_name: runner.layer_audit
                for module_name, runner in self.runners.items()
            }
        )
        use_vulkan_blend_patch = bool(
            getattr(self.ncnn, "has_lada_blend_patch_vulkan", False)
            and callable(getattr(self.ncnn, "blend_patch_gpu", None))
        )
        use_vulkan_blend_patch_inplace = bool(
            getattr(self.ncnn, "has_lada_blend_patch_vulkan_inplace", False)
            and callable(getattr(self.ncnn, "blend_patch_gpu_inplace", None))
        )
        use_vulkan_blend_patch_preprocess = bool(
            getattr(self.ncnn, "has_lada_blend_patch_vulkan_preprocess", False)
            and callable(getattr(self.ncnn, "blend_patch_gpu_preprocess", None))
        )
        use_vulkan_blend_patch_preprocess_inplace = bool(
            getattr(self.ncnn, "has_lada_blend_patch_vulkan_preprocess_inplace", False)
            and callable(getattr(self.ncnn, "blend_patch_gpu_preprocess_inplace", None))
        )
        use_vulkan_blend_patch_preprocess_batch_inplace = bool(
            getattr(
                self.ncnn,
                "has_lada_blend_patch_vulkan_preprocess_inplace_batch",
                False,
            )
            and callable(
                getattr(self.ncnn, "blend_patch_gpu_preprocess_inplace_batch", None)
            )
        )
        use_gpu_blob_bridge = all(
            runner.gpu_runner is not None for runner in self.runners.values()
        )
        self.frame_preprocess_runner = self.runners.get("quarter_downsample")
        use_vulkan_frame_preprocess = bool(
            use_gpu_blob_bridge
            and self.frame_preprocess_runner is not None
            and self.frame_preprocess_runner.gpu_runner is not None
            and hasattr(self.frame_preprocess_runner.gpu_runner, "preprocess_bgr_u8")
        )
        use_vulkan_frame_preprocess_batch = bool(
            use_vulkan_frame_preprocess
            and self.frame_preprocess_runner is not None
            and hasattr(self.frame_preprocess_runner, "preprocess_bgr_u8_frames")
            and hasattr(self.frame_preprocess_runner.gpu_runner, "preprocess_bgr_u8_batch")
        )
        use_fused_restore_clip = "restore_clip" in self.runners
        use_recurrent_modular_runtime = all(
            module_name in self.runners for module_name in _RECURRENT_RUNTIME_MODULES
        )
        self.native_clip_runner = None
        if all(
            module_name in self.artifacts
            for module_name in _RECURRENT_CLIP_RUNNER_MODULES
        ) and ncnn_has_lada_basicvsrpp_clip_runner(self.ncnn):
            self.native_clip_runner = NcnnVulkanBasicvsrppClipRunner(
                {
                    module_name: self.artifacts[module_name]
                    for module_name in _RECURRENT_CLIP_RUNNER_MODULES
                },
                fp16=self.fp16,
                num_threads=self.num_threads,
            )
        use_native_clip_runner = self.native_clip_runner is not None
        supports_descriptor_restore = bool(
            use_native_clip_runner
            and self.native_clip_runner is not None
            and self.native_clip_runner.supports_resized_bgr_u8_input
        )
        self.runtime_features = BasicvsrppVulkanRuntimeFeatures(
            use_gpu_blob_bridge=use_gpu_blob_bridge,
            use_vulkan_frame_preprocess=use_vulkan_frame_preprocess,
            use_vulkan_frame_preprocess_batch=use_vulkan_frame_preprocess_batch,
            use_fused_restore_clip=use_fused_restore_clip,
            use_recurrent_modular_runtime=use_recurrent_modular_runtime,
            use_native_clip_runner=use_native_clip_runner,
            supports_descriptor_restore=supports_descriptor_restore,
            use_vulkan_blend_patch=use_vulkan_blend_patch,
            use_vulkan_blend_patch_inplace=use_vulkan_blend_patch_inplace,
            use_vulkan_blend_patch_preprocess=use_vulkan_blend_patch_preprocess,
            use_vulkan_blend_patch_preprocess_inplace=use_vulkan_blend_patch_preprocess_inplace,
            use_vulkan_blend_patch_preprocess_batch_inplace=use_vulkan_blend_patch_preprocess_batch_inplace,
        )

    def _reload_modular_runtime(self) -> None:
        if self.frame_count != _MODULAR_FRAME_COUNT:
            return

        self.runners = None
        gc.collect()
        self._initialize_modular_runtime(
            self.checkpoint_path,
            config_path=self.config_path,
            artifacts_dir=self.artifacts_dir,
        )

    def _run_profiled_module(
        self,
        module_name: str,
        inputs: dict[str, object],
        *,
        bucket: str,
        prefer_gpu: bool = False,
        prefer_gpu_download: bool = False,
    ) -> object:
        with self.profiler.measure(bucket):
            if prefer_gpu_download and self.runners[module_name].gpu_runner is not None:
                return self.runners[module_name].run_gpu_download(inputs)
            if prefer_gpu and self.runners[module_name].gpu_runner is not None:
                return self.runners[module_name].run_gpu(inputs)
            return self.runners[module_name].run(inputs)

    def restore_cropped_clip_frames(
        self,
        clip_frames: list[Image | ImageTensor],
        *,
        size: int,
        resize_reference_shape: tuple[int, int],
        pad_mode: str,
    ) -> list[ImageTensor]:
        return run_restore_cropped_clip_frames(
            self,
            clip_frames,
            size=size,
            resize_reference_shape=resize_reference_shape,
            pad_mode=pad_mode,
        )

    def restore(
        self,
        video: list[Image | ImageTensor],
        max_frames: int = -1,
    ) -> list[ImageTensor]:
        return run_restore(self, video, max_frames=max_frames)

    def get_last_profile(self) -> dict[str, float | int]:
        return dict(self.last_profile)

    def get_runtime_scheduling_options(self) -> RestorationSchedulingOptions:
        """Return queue scheduling knobs consumed by the outer restoration pipeline."""
        return self.runtime_scheduling

    def get_runtime_features(self) -> BasicvsrppVulkanRuntimeFeatures:
        """Return the optional native fast paths exposed by this runtime."""
        return self.runtime_features

    def blend_patch(
        self,
        frame_roi: ImageTensor,
        clip_img: ImageTensor,
        clip_mask: ImageTensor,
    ) -> ImageTensor:
        return run_blend_patch(self, frame_roi, clip_img, clip_mask)

    def blend_patch_padded(
        self,
        frame_roi: ImageTensor,
        clip_img: ImageTensor,
        clip_mask: ImageTensor,
        pad_after_resize: tuple[int, int, int, int],
    ) -> ImageTensor:
        return run_blend_patch_padded(
            self,
            frame_roi,
            clip_img,
            clip_mask,
            pad_after_resize,
        )

    def blend_patch_padded_batch(
        self,
        frame_rois: list[ImageTensor],
        clip_imgs: list[ImageTensor],
        clip_masks: list[ImageTensor],
        pad_after_resizes: list[tuple[int, int, int, int]],
    ) -> list[ImageTensor] | None:
        return run_blend_patch_padded_batch(
            self,
            frame_rois,
            clip_imgs,
            clip_masks,
            pad_after_resizes,
        )

    def get_vulkan_audit(self) -> dict[str, object]:
        return dict(self.vulkan_audit)

    def release_cached_memory(self) -> None:
        return None
