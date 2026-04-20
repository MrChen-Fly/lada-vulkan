from __future__ import annotations

import gc
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lada.extensions.runtime_registry import UnsupportedComputeTargetError
from ..ncnn_runtime import (
    NcnnVulkanBasicvsrppClipRunner,
    NcnnVulkanModuleRunner,
)
from lada.utils import Image, ImageTensor

from .basicvsrpp_blend import (
    blend_patch as run_blend_patch,
    blend_patch_padded as run_blend_patch_padded,
    blend_patch_padded_batch as run_blend_patch_padded_batch,
)
from .basicvsrpp_common import (
    _DEFAULT_RUNTIME_FRAME_SHAPE,
    _MODULAR_FRAME_COUNT,
)
from .basicvsrpp_cpu_extractor import runtime_value_to_numpy
from .basicvsrpp_recurrent_runtime import (
    run_deform_align as run_modular_deform_align,
    run_flow_warp as run_modular_flow_warp,
    run_propagate_step as run_modular_propagate_step,
)
from .basicvsrpp_runtime_bootstrap import (
    initialize_modular_runtime,
)
from .basicvsrpp_runtime_support import (
    BasicvsrppRuntimeShape,
    resolve_basicvsrpp_runtime_shape,
)
from .basicvsrpp_restore_paths import (
    restore as run_restore,
    restore_cropped_clip_frames as run_restore_cropped_clip_frames,
)
from .runtime_options import (
    RestorationRuntimeFeatures,
    RestorationSchedulingOptions,
)
from .runtime_profiling import WallClockProfiler

_STREAM_RESTORE_CHUNK_SIZE = 60
_DEFAULT_RUNTIME_SHAPE = resolve_basicvsrpp_runtime_shape(_DEFAULT_RUNTIME_FRAME_SHAPE)


def _get_restorer_runtime_shape(restorer: object) -> BasicvsrppRuntimeShape:
    return getattr(restorer, "runtime_shape", _DEFAULT_RUNTIME_SHAPE)


class NcnnVulkanBasicvsrppMosaicRestorer:
    """BasicVSR++ Vulkan restorer with fixed-size Vulkan subgraphs."""

    runtime = "vulkan"
    descriptor_resize_mode = "torch_bilinear"
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
        self.runtime_shape = _DEFAULT_RUNTIME_SHAPE
        self.profiler = WallClockProfiler()
        self.last_profile: dict[str, float | int] = {}
        self._native_clip_profile_snapshot: dict[str, float | int] = {}
        self.vulkan_audit: dict[str, object] = {}
        self.runtime_features = RestorationRuntimeFeatures()
        self.runtime_scheduling = RestorationSchedulingOptions(
            stream_restore_chunk_size=_STREAM_RESTORE_CHUNK_SIZE,
            # Keep the detector on the existing 4-frame batching path so
            # low-latency clip restore can improve first-frame latency
            # without giving back the overall throughput gains.
            detector_batch_size=4,
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
            frame_shape=self.runtime_shape.frame_shape,
        )

    def _initialize_modular_runtime(
        self,
        checkpoint_path: str,
        *,
        config_path: str | dict | None,
        artifacts_dir: str | Path | None,
        frame_shape: int | tuple[int, int] | list[int],
    ) -> None:
        initialize_modular_runtime(
            self,
            checkpoint_path,
            config_path=config_path,
            artifacts_dir=artifacts_dir,
            frame_shape=frame_shape,
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
            frame_shape=self.runtime_shape.frame_shape,
        )

    def _resolve_runtime_input_shape(
        self,
        frames: list[Image | ImageTensor],
    ) -> tuple[int, int]:
        """Resolve one uniform runtime frame shape from a clip batch."""
        shapes = {
            (int(frame.shape[0]), int(frame.shape[1]))
            for frame in frames
        }
        if not shapes:
            return _get_restorer_runtime_shape(self).frame_shape
        if len(shapes) != 1:
            raise RuntimeError(
                f"Vulkan BasicVSR++ expects one uniform frame shape per clip, got {sorted(shapes)}."
            )
        return next(iter(shapes))

    def _ensure_runtime_for_input_shape(
        self,
        input_shape: int | tuple[int, int] | list[int],
    ) -> None:
        """Switch to a shape-matched artifact bundle when the clip frame shape changes."""
        target_shape = resolve_basicvsrpp_runtime_shape(input_shape)
        current_shape = _get_restorer_runtime_shape(self)
        if target_shape.frame_shape == current_shape.frame_shape:
            return

        self.runners = None
        self.native_clip_runner = None
        self.frame_preprocess_runner = None
        gc.collect()
        self._initialize_modular_runtime(
            self.checkpoint_path,
            config_path=self.config_path,
            artifacts_dir=self.artifacts_dir,
            frame_shape=target_shape.frame_shape,
        )

    def _run_profiled_module(
        self,
        module_name: str,
        inputs: dict[str, object],
        *,
        bucket: str | None,
        prefer_gpu: bool = False,
        prefer_gpu_download: bool = False,
    ) -> object:
        with self.profiler.measure(bucket) if bucket is not None else nullcontext():
            if (
                prefer_gpu_download
                and self._should_use_gpu_download(module_name)
            ):
                return self.runners[module_name].run_gpu_download(inputs)
            if prefer_gpu_download and not self._should_use_gpu_download(module_name):
                normalized_inputs = {
                    input_name: runtime_value_to_numpy(self, value)
                    for input_name, value in inputs.items()
                }
                return self.runners[module_name].run(normalized_inputs)
            if prefer_gpu and self.runners[module_name].gpu_runner is not None:
                return self.runners[module_name].run_gpu(inputs)
            return self.runners[module_name].run(inputs)

    def _should_use_gpu_download(self, module_name: str) -> bool:
        """Return whether one module can execute on Vulkan and download to CPU."""
        runner = self.runners[module_name]
        return runner.gpu_runner is not None

    def run_spynet(
        self,
        ref: object,
        supp: object,
        *,
        bucket: str | None = None,
        prefer_gpu_download: bool = False,
    ) -> np.ndarray:
        with self.profiler.measure(bucket) if bucket is not None else nullcontext():
            ref_np = runtime_value_to_numpy(self, ref)
            supp_np = runtime_value_to_numpy(self, supp)
            if ref_np.shape != supp_np.shape:
                raise RuntimeError(
                    f"SPyNet runtime expects matched input shapes, got {ref_np.shape} vs {supp_np.shape}."
                )
            if ref_np.ndim != 3:
                raise RuntimeError(f"SPyNet runtime expects CHW tensors, got {ref_np.shape}.")

            height, width = int(ref_np.shape[1]), int(ref_np.shape[2])
            runtime_shape = _get_restorer_runtime_shape(self)
            runner_name = "spynet"
            runtime_height, runtime_width = runtime_shape.spynet_core_shape
            if (
                (height, width) == runtime_shape.spynet_patch_shape
                and "spynet_patch" in self.runners
            ):
                runner_name = "spynet_patch"
                runtime_height, runtime_width = runtime_shape.spynet_patch_shape

            ref_tensor = torch.from_numpy(ref_np).unsqueeze(0)
            supp_tensor = torch.from_numpy(supp_np).unsqueeze(0)
            if runtime_height != height or runtime_width != width:
                ref_tensor = F.interpolate(
                    ref_tensor,
                    size=(runtime_height, runtime_width),
                    mode="bilinear",
                    align_corners=False,
                )
                supp_tensor = F.interpolate(
                    supp_tensor,
                    size=(runtime_height, runtime_width),
                    mode="bilinear",
                    align_corners=False,
                )

            spynet_inputs = {
                "in0": np.ascontiguousarray(ref_tensor.squeeze(0).numpy(), dtype=np.float32),
                "in1": np.ascontiguousarray(supp_tensor.squeeze(0).numpy(), dtype=np.float32),
            }
            runner = self.runners[runner_name]
            use_gpu_download = prefer_gpu_download and self._should_use_gpu_download(runner_name)
            if use_gpu_download:
                flow = np.ascontiguousarray(runner.run_gpu_download(spynet_inputs), dtype=np.float32)
            else:
                flow = np.ascontiguousarray(runner.run(spynet_inputs), dtype=np.float32)

            if flow.ndim != 3:
                raise RuntimeError(
                    f"SPyNet runtime produced an unexpected output shape {tuple(flow.shape)}."
                )
            flow_height, flow_width = int(flow.shape[1]), int(flow.shape[2])
            if flow_height == height and flow_width == width:
                return flow

            flow_tensor = torch.from_numpy(flow).unsqueeze(0)
            flow_tensor = F.interpolate(
                flow_tensor,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            flow_tensor[:, 0, :, :] *= float(width) / float(flow_width)
            flow_tensor[:, 1, :, :] *= float(height) / float(flow_height)
            return np.ascontiguousarray(flow_tensor.squeeze(0).numpy(), dtype=np.float32)

    def run_flow_warp(
        self,
        feature: object,
        flow: object,
        *,
        bucket: str | None = None,
        prefer_gpu_download: bool = False,
    ) -> np.ndarray:
        return run_modular_flow_warp(
            self,
            feature,
            flow,
            bucket=bucket,
            prefer_gpu_download=prefer_gpu_download,
        )

    def run_deform_align(
        self,
        module_name: str,
        feature_pair: object,
        extra_feat: object,
        flow_n1: object,
        flow_n2: object,
        *,
        bucket: str | None = None,
        prefer_gpu_download: bool = False,
    ) -> np.ndarray:
        return run_modular_deform_align(
            self,
            module_name,
            feature_pair,
            extra_feat,
            flow_n1,
            flow_n2,
            bucket=bucket,
            prefer_gpu_download=prefer_gpu_download,
        )

    def run_propagate_step(
        self,
        module_name: str,
        step_inputs: dict[str, object],
        *,
        bucket: str | None = None,
        prefer_gpu: bool = False,
        prefer_gpu_download: bool = False,
    ) -> object:
        return run_modular_propagate_step(
            self,
            module_name,
            step_inputs,
            bucket=bucket,
            prefer_gpu=prefer_gpu,
            prefer_gpu_download=prefer_gpu_download,
        )

    def restore_cropped_clip_frames(
        self,
        clip_frames: list[Image | ImageTensor],
        *,
        size: int,
        resize_reference_shape: tuple[int, int],
        pad_mode: str,
    ) -> list[ImageTensor]:
        self._ensure_runtime_for_input_shape((int(size), int(size)))
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
        if video:
            self._ensure_runtime_for_input_shape(
                self._resolve_runtime_input_shape(video),
            )
        return run_restore(self, video, max_frames=max_frames)

    def get_last_profile(self) -> dict[str, float | int]:
        return dict(self.last_profile)

    def get_runtime_scheduling_options(self) -> RestorationSchedulingOptions:
        """Return queue scheduling knobs consumed by the outer restoration pipeline."""
        return self.runtime_scheduling

    def get_runtime_features(self) -> RestorationRuntimeFeatures:
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
