from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from lada.utils import Image, ImageTensor

from .basicvsrpp_io import (
    _array_to_uint8_frame,
    _build_output_frame_inputs,
    _build_replicated_clip_window,
    _build_restore_clip_inputs,
    _estimate_feature_area_for_frame,
    _frame_to_chw_float32,
    _is_gpu_bridge_retryable_error,
    _split_clip_restore_output,
)
from .clip_resize_semantics import (
    floor_resized_length,
    resize_and_pad_clip_frame,
)
from .basicvsrpp_recurrent_runtime import (
    finalize_last_profile,
    merge_native_clip_profile,
    run_branch_recurrent,
)

if TYPE_CHECKING:
    from .basicvsrpp_restorer import (
        NcnnVulkanBasicvsrppMosaicRestorer,
    )


_BRANCH_NAMES = ("backward_1", "forward_1", "backward_2", "forward_2")
_RECURRENT_GPU_BRIDGE_MAX_FEATURE_AREA = 128 * 128


def _clone_runtime_values_if_supported(values: list[object]) -> list[object]:
    """Clone GPU runtime blobs when the local ncnn binding exposes a safe device-side clone."""
    cloned_values: list[object] = []
    for value in values:
        clone = getattr(value, "clone", None)
        if callable(clone):
            cloned_values.append(clone())
            continue
        cloned_values.append(value)
    return cloned_values


def _resize_cropped_frames_for_runtime(
    clip_frames: list[Image | ImageTensor],
    *,
    size: int,
    resize_reference_shape: tuple[int, int],
    pad_mode: str,
) -> list[Image | ImageTensor]:
    """Resize cropped clip frames with Torch/CUDA-aligned bilinear semantics."""
    reference_width = max(int(resize_reference_shape[0]), 1)
    reference_height = max(int(resize_reference_shape[1]), 1)

    prepared_frames: list[Image | ImageTensor] = []
    for frame in clip_frames:
        resize_shape = (
            floor_resized_length(
                frame.shape[0],
                target_size=size,
                reference_length=reference_height,
            ),
            floor_resized_length(
                frame.shape[1],
                target_size=size,
                reference_length=reference_width,
            ),
        )
        pad_h = size - resize_shape[0]
        pad_w = size - resize_shape[1]
        pad_after_resize = (
            (pad_h + 1) // 2,
            pad_h // 2,
            (pad_w + 1) // 2,
            pad_w // 2,
        )
        padded_frame = resize_and_pad_clip_frame(
            frame,
            resize_shape=resize_shape,
            pad_after_resize=pad_after_resize,
            size=size,
            pad_mode=pad_mode,
            resize_mode="torch_bilinear",
        )
        prepared_frames.append(padded_frame)
    return prepared_frames


def preprocess_clip_frames_to_runtime_inputs(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
    *,
    prefer_gpu_bridge: bool,
) -> list[object]:
    if (
        prefer_gpu_bridge
        and restorer.runtime_features.use_native_frame_preprocess
        and restorer.frame_preprocess_runner is not None
        and clip_frames
    ):
        input_shape = (int(clip_frames[0].shape[0]), int(clip_frames[0].shape[1]))
        with restorer.profiler.measure("vulkan_frame_preprocess_s"):
            if (
                restorer.runtime_features.use_native_frame_preprocess_batch
                and hasattr(restorer.frame_preprocess_runner, "preprocess_bgr_u8_frames")
            ):
                return _clone_runtime_values_if_supported(
                    list(
                        restorer.frame_preprocess_runner.preprocess_bgr_u8_frames(
                            list(clip_frames),
                            input_shape=input_shape,
                        )
                    )
                )
            return _clone_runtime_values_if_supported(
                [
                    restorer.frame_preprocess_runner.preprocess_bgr_u8_frame(
                        frame,
                        input_shape=input_shape,
                    )
                    for frame in clip_frames
                ]
            )

    restorer.profiler.add_duration("vulkan_frame_preprocess_s", 0.0)
    with restorer.profiler.measure("cpu_frame_preprocess_s"):
        return [_frame_to_chw_float32(frame) for frame in clip_frames]


def prepare_clip_features(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
    *,
    prefer_gpu_bridge: bool | None = None,
) -> tuple[list[object], list[object], list[object], list[object]]:
    if restorer.runners is None:
        raise RuntimeError("Modular Vulkan runners are not initialized.")
    use_gpu_bridge = restorer.runtime_features.use_native_blob_bridge
    if prefer_gpu_bridge is not None:
        use_gpu_bridge = prefer_gpu_bridge

    lqs = preprocess_clip_frames_to_runtime_inputs(
        restorer,
        clip_frames,
        prefer_gpu_bridge=use_gpu_bridge,
    )
    lqs_downsampled = [
        restorer._run_profiled_module(
            "quarter_downsample",
            {"in0": current_lq},
            bucket="vulkan_quarter_downsample_s",
            prefer_gpu=use_gpu_bridge,
            prefer_gpu_download=not use_gpu_bridge,
        )
        for current_lq in lqs
    ]
    spatial_feats = [
        restorer._run_profiled_module(
            "feat_extract",
            {"in0": current_lq},
            bucket="vulkan_feat_extract_s",
            prefer_gpu=use_gpu_bridge,
            prefer_gpu_download=not use_gpu_bridge,
        )
        for current_lq in lqs
    ]
    flows_backward = [
        restorer.run_spynet(
            lqs_downsampled[index],
            lqs_downsampled[index + 1],
            bucket="vulkan_spynet_s",
            prefer_gpu_download=True,
        )
        for index in range(len(lqs_downsampled) - 1)
    ]
    flows_forward = [
        restorer.run_spynet(
            lqs_downsampled[index + 1],
            lqs_downsampled[index],
            bucket="vulkan_spynet_s",
            prefer_gpu_download=True,
        )
        for index in range(len(lqs_downsampled) - 1)
    ]
    return lqs, spatial_feats, flows_backward, flows_forward


def restore_clip_with_fused_module(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    lqs: list[object],
    spatial_feats: list[object],
    flows_backward: list[object],
    flows_forward: list[object],
) -> list[ImageTensor]:
    fused_inputs = _build_restore_clip_inputs(
        lqs,
        spatial_feats,
        flows_backward,
        flows_forward,
    )
    output = restorer._run_profiled_module(
        "restore_clip",
        fused_inputs,
        bucket="vulkan_restore_clip_s",
        prefer_gpu_download=True,
    )
    with restorer.profiler.measure("cpu_output_postprocess_s"):
        return _split_clip_restore_output(output, frame_count=len(lqs))


def restore_clip(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
) -> list[ImageTensor]:
    lqs, spatial_feats, flows_backward, flows_forward = prepare_clip_features(
        restorer,
        clip_frames,
    )
    return restore_clip_with_fused_module(
        restorer,
        lqs,
        spatial_feats,
        flows_backward,
        flows_forward,
    )


def restore_clip_recurrent_impl(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
    *,
    use_gpu_bridge: bool,
) -> list[ImageTensor]:
    lqs, spatial_feats, flows_backward, flows_forward = prepare_clip_features(
        restorer,
        clip_frames,
        prefer_gpu_bridge=use_gpu_bridge,
    )
    branch_feats: dict[str, list[object]] = {}
    for module_name in _BRANCH_NAMES:
        flows = flows_backward if module_name.startswith("backward") else flows_forward
        branch_feats[module_name] = run_branch_recurrent(
            restorer,
            module_name,
            spatial_feats,
            branch_feats,
            flows,
            use_gpu_bridge=use_gpu_bridge,
        )

    outputs: list[ImageTensor] = []
    for frame_index in range(len(lqs)):
        output = restorer._run_profiled_module(
            "output_frame",
            _build_output_frame_inputs(
                lqs,
                spatial_feats,
                branch_feats,
                frame_index,
            ),
            bucket="vulkan_output_frame_s",
            prefer_gpu_download=True,
        )
        with restorer.profiler.measure("cpu_output_postprocess_s"):
            outputs.append(_array_to_uint8_frame(output))
    return outputs


def restore_clip_recurrent_native(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
) -> list[ImageTensor]:
    if restorer.native_clip_runner is None:
        raise RuntimeError("Native BasicVSR++ clip runner is not initialized.")

    with restorer.profiler.measure("cpu_frame_preprocess_s"):
        lqs = [_frame_to_chw_float32(frame) for frame in clip_frames]
    restorer.profiler.add_duration("vulkan_frame_preprocess_s", 0.0)
    with restorer.profiler.measure("vulkan_recurrent_clip_native_s"):
        outputs = restorer.native_clip_runner.restore(lqs)
    merge_native_clip_profile(restorer)
    with restorer.profiler.measure("cpu_output_postprocess_s"):
        return [_array_to_uint8_frame(output) for output in outputs]


def restore_clip_recurrent_native_resized(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
    *,
    target_size: int,
    resize_reference_shape: tuple[int, int],
    pad_mode: str,
) -> list[ImageTensor]:
    """Run the native clip runner's resized uint8 entry without CPU materialization."""
    if restorer.native_clip_runner is None:
        raise RuntimeError("Native BasicVSR++ clip runner is not initialized.")
    if not restorer.native_clip_runner.supports_resized_bgr_u8_input:
        raise RuntimeError("Native BasicVSR++ clip runner does not support resized uint8 input.")

    runtime_input_frames = list(clip_frames)
    center_output_index = 0
    if len(runtime_input_frames) == 1:
        runtime_input_frames = _build_replicated_clip_window(
            runtime_input_frames,
            center_index=0,
            frame_count=restorer.frame_count,
        )
        center_output_index = restorer.frame_count // 2

    with restorer.profiler.measure("vulkan_recurrent_clip_native_s"):
        outputs = restorer.native_clip_runner.restore_bgr_u8_resized(
            runtime_input_frames,
            target_size=target_size,
            resize_reference_shape=resize_reference_shape,
            pad_mode=pad_mode,
        )
    merge_native_clip_profile(restorer)
    with restorer.profiler.measure("cpu_output_postprocess_s"):
        result = [_array_to_uint8_frame(output) for output in outputs]

    if len(clip_frames) == 1:
        return [result[center_output_index]]
    return result


def restore_clip_recurrent(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
) -> list[ImageTensor]:
    use_gpu_bridge = restorer.runtime_features.use_native_blob_bridge and all(
        _estimate_feature_area_for_frame(frame)
        <= _RECURRENT_GPU_BRIDGE_MAX_FEATURE_AREA
        for frame in clip_frames
    )
    if use_gpu_bridge and restorer.runtime_features.use_native_clip_runner:
        try:
            return restore_clip_recurrent_native(restorer, clip_frames)
        except RuntimeError as exc:
            if not _is_gpu_bridge_retryable_error(exc):
                raise
            restorer._reload_modular_runtime()
            return restore_clip_recurrent_impl(
                restorer,
                clip_frames,
                use_gpu_bridge=False,
            )

    if use_gpu_bridge and restorer.runtime_features.use_native_frame_preprocess:
        try:
            return restore_clip_recurrent_impl(
                restorer,
                clip_frames,
                use_gpu_bridge=True,
            )
        except RuntimeError as exc:
            if not _is_gpu_bridge_retryable_error(exc):
                raise
            restorer._reload_modular_runtime()
            return restore_clip_recurrent_impl(
                restorer,
                clip_frames,
                use_gpu_bridge=False,
            )

    if not use_gpu_bridge:
        return restore_clip_recurrent_impl(
            restorer,
            clip_frames,
            use_gpu_bridge=False,
        )

    try:
        return restore_clip_recurrent_impl(
            restorer,
            clip_frames,
            use_gpu_bridge=True,
        )
    except RuntimeError as exc:
        if not _is_gpu_bridge_retryable_error(exc):
            raise
        restorer._reload_modular_runtime()
        return restore_clip_recurrent_impl(
            restorer,
            clip_frames,
            use_gpu_bridge=False,
        )


def restore_clip_sliding_window(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    frames: list[Image | ImageTensor],
) -> list[ImageTensor]:
    center_output_index = restorer.frame_count // 2
    restored_frames: list[ImageTensor] = []
    for center_index in range(len(frames)):
        window = _build_replicated_clip_window(
            frames,
            center_index=center_index,
            frame_count=restorer.frame_count,
        )
        restored_window = restore_clip(restorer, window)
        restored_frames.append(restored_window[center_output_index])
    return restored_frames


def restore_cropped_clip_frames(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
    *,
    size: int,
    resize_reference_shape: tuple[int, int],
    pad_mode: str,
) -> list[ImageTensor]:
    if (
        not restorer.runtime_features.supports_descriptor_restore
        or restorer.native_clip_runner is None
    ):
        raise RuntimeError("Descriptor-native Vulkan restore is unavailable.")

    restorer.profiler.reset()
    restorer._native_clip_profile_snapshot = {}
    started_at = perf_counter()
    if not clip_frames:
        finalize_last_profile(restorer, total_s=perf_counter() - started_at)
        return []

    result = restore_clip_recurrent_native_resized(
        restorer,
        clip_frames,
        target_size=size,
        resize_reference_shape=resize_reference_shape,
        pad_mode=pad_mode,
    )
    finalize_last_profile(restorer, total_s=perf_counter() - started_at)
    return result


def restore(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    video: list[Image | ImageTensor],
    max_frames: int = -1,
) -> list[ImageTensor]:
    restorer.profiler.reset()
    restorer._native_clip_profile_snapshot = {}
    started_at = perf_counter()
    frames = video if max_frames < 0 else video[:max_frames]
    if not frames:
        finalize_last_profile(restorer, total_s=perf_counter() - started_at)
        return []

    if len(frames) > 1 and restorer.runtime_features.use_native_recurrent_runtime:
        result = restore_clip_recurrent(restorer, frames)
    elif (
        len(frames) == restorer.frame_count
        and restorer.runtime_features.use_native_fused_restore_clip
    ):
        result = restore_clip(restorer, frames)
    else:
        result = restore_clip_sliding_window(restorer, frames)
    finalize_last_profile(restorer, total_s=perf_counter() - started_at)
    return result
