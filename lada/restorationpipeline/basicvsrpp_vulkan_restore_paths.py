from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from lada.models.basicvsrpp.ncnn_vulkan import is_ncnn_vulkan_tensor
from lada.utils import Image, ImageTensor

from .basicvsrpp_vulkan_io import (
    _array_to_uint8_frame,
    _build_backbone_inputs,
    _build_output_frame_inputs,
    _build_replicated_clip_window,
    _build_restore_clip_inputs,
    _build_step_inputs,
    _estimate_feature_area_for_frame,
    _frame_to_chw_float32,
    _is_gpu_bridge_retryable_error,
    _split_clip_restore_output,
    _zeros_like_runtime_value,
)

if TYPE_CHECKING:
    from .basicvsrpp_vulkan_restorer import NcnnVulkanBasicvsrppMosaicRestorer


_BRANCH_NAMES = ("backward_1", "forward_1", "backward_2", "forward_2")
_RECURRENT_GPU_BRIDGE_MAX_FEATURE_AREA = 128 * 128


def preprocess_clip_frames_to_runtime_inputs(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
    *,
    prefer_gpu_bridge: bool,
) -> list[object]:
    if (
        prefer_gpu_bridge
        and restorer.runtime_features.use_vulkan_frame_preprocess
        and restorer.frame_preprocess_runner is not None
    ):
        restorer.profiler.add_duration("cpu_frame_preprocess_s", 0.0)
        with restorer.profiler.measure("vulkan_frame_preprocess_s"):
            if restorer.runtime_features.use_vulkan_frame_preprocess_batch:
                restorer.profiler.add_count("vulkan_frame_preprocess_batch")
                return restorer.frame_preprocess_runner.preprocess_bgr_u8_frames(
                    clip_frames
                )
            restorer.profiler.add_count("vulkan_frame_preprocess_single")
            return [
                restorer.frame_preprocess_runner.preprocess_bgr_u8_frame(frame)
                for frame in clip_frames
            ]

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
    use_gpu_bridge = restorer.runtime_features.use_gpu_blob_bridge
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
        )
        for current_lq in lqs
    ]
    spatial_feats = [
        restorer._run_profiled_module(
            "feat_extract",
            {"in0": current_lq},
            bucket="vulkan_feat_extract_s",
            prefer_gpu=use_gpu_bridge,
        )
        for current_lq in lqs
    ]
    flows_backward = [
        restorer._run_profiled_module(
            "spynet",
            {"in0": lqs_downsampled[index], "in1": lqs_downsampled[index + 1]},
            bucket="vulkan_spynet_s",
            prefer_gpu=use_gpu_bridge,
        )
        for index in range(len(lqs_downsampled) - 1)
    ]
    flows_forward = [
        restorer._run_profiled_module(
            "spynet",
            {"in0": lqs_downsampled[index + 1], "in1": lqs_downsampled[index]},
            bucket="vulkan_spynet_s",
            prefer_gpu=use_gpu_bridge,
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


def run_branch_recurrent(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    module_name: str,
    spatial_feats: list[object],
    branch_feats: dict[str, list[object]],
    flows: list[object],
    *,
    use_gpu_bridge: bool,
) -> list[object]:
    use_gpu_bridge = use_gpu_bridge and is_ncnn_vulkan_tensor(
        restorer.ncnn,
        spatial_feats[0],
    )
    frame_indices = list(range(len(spatial_feats)))
    if module_name.startswith("backward"):
        frame_indices.reverse()

    feat_prop = _zeros_like_runtime_value(restorer.ncnn, spatial_feats[0])
    zero_feat = _zeros_like_runtime_value(restorer.ncnn, spatial_feats[0])
    zero_flow = _zeros_like_runtime_value(restorer.ncnn, flows[0])
    outputs: list[object] = []
    previous_raw_flow = zero_flow

    for step_index, frame_index in enumerate(frame_indices):
        feat_current = spatial_feats[frame_index]
        if step_index == 0:
            feat_prop = restorer._run_profiled_module(
                f"{module_name}_backbone",
                _build_backbone_inputs(
                    module_name,
                    feat_current,
                    feat_prop,
                    branch_feats,
                    frame_index,
                ),
                bucket="vulkan_branch_backbone_s",
                prefer_gpu=use_gpu_bridge,
            )
            outputs.append(feat_prop)
            continue

        adjacent_index = frame_indices[step_index - 1]
        raw_flow_n1 = flows[min(frame_index, adjacent_index)]
        feat_n2 = outputs[-2] if step_index > 1 else zero_feat
        prev_flow_n2 = previous_raw_flow if step_index > 1 else zero_flow
        feat_prop = restorer._run_profiled_module(
            f"{module_name}_step",
            _build_step_inputs(
                module_name,
                feat_prop,
                feat_current,
                branch_feats,
                frame_index,
                feat_n2,
                raw_flow_n1,
                prev_flow_n2,
            ),
            bucket="vulkan_branch_step_s",
            prefer_gpu=use_gpu_bridge,
        )
        outputs.append(feat_prop)
        previous_raw_flow = raw_flow_n1

    if module_name.startswith("backward"):
        outputs.reverse()
    return outputs


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
            prefer_gpu_download=use_gpu_bridge,
        )
        with restorer.profiler.measure("cpu_output_postprocess_s"):
            outputs.append(_array_to_uint8_frame(output))
    return outputs


def merge_native_clip_profile(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
) -> None:
    if restorer.native_clip_runner is None:
        return
    get_last_profile = getattr(restorer.native_clip_runner, "get_last_profile", None)
    if not callable(get_last_profile):
        return
    restorer._native_clip_profile_snapshot = {}
    for key, value in get_last_profile().items():
        prefixed_key = f"vulkan_native_{key}"
        if key.endswith("_count"):
            restorer._native_clip_profile_snapshot[prefixed_key] = int(value)
            continue
        restorer.profiler.add_duration(prefixed_key, float(value))


def restore_clip_recurrent_native(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
) -> list[ImageTensor]:
    if restorer.native_clip_runner is None:
        raise RuntimeError("Native BasicVSR++ clip runner is not initialized.")

    if restorer.native_clip_runner.supports_bgr_u8_input:
        restorer.profiler.add_duration("cpu_frame_preprocess_s", 0.0)
        restorer.profiler.add_duration("vulkan_frame_preprocess_s", 0.0)
        with restorer.profiler.measure("vulkan_recurrent_clip_native_s"):
            outputs = restorer.native_clip_runner.restore_bgr_u8(clip_frames)
        merge_native_clip_profile(restorer)
        with restorer.profiler.measure("cpu_output_postprocess_s"):
            return [_array_to_uint8_frame(output) for output in outputs]

    with restorer.profiler.measure("cpu_frame_preprocess_s"):
        lqs = [_frame_to_chw_float32(frame) for frame in clip_frames]
    with restorer.profiler.measure("vulkan_recurrent_clip_native_s"):
        outputs = restorer.native_clip_runner.restore(lqs)
    merge_native_clip_profile(restorer)
    with restorer.profiler.measure("cpu_output_postprocess_s"):
        return [_array_to_uint8_frame(output) for output in outputs]


def finalize_last_profile(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    *,
    total_s: float,
) -> None:
    restorer.last_profile = restorer.profiler.snapshot(total_s=total_s)
    restorer.last_profile.update(restorer._native_clip_profile_snapshot)


def restore_clip_recurrent(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    clip_frames: list[Image | ImageTensor],
) -> list[ImageTensor]:
    use_gpu_bridge = restorer.runtime_features.use_gpu_blob_bridge and all(
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

    if use_gpu_bridge and restorer.runtime_features.use_vulkan_frame_preprocess:
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
    with restorer.profiler.measure("vulkan_recurrent_clip_native_s"):
        outputs = restorer.native_clip_runner.restore_bgr_u8_resized(
            clip_frames,
            target_size=size,
            resize_reference_shape=resize_reference_shape,
            pad_mode=pad_mode,
        )
    merge_native_clip_profile(restorer)
    with restorer.profiler.measure("cpu_output_postprocess_s"):
        result = [_array_to_uint8_frame(output) for output in outputs]
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

    if len(frames) == restorer.frame_count:
        result = restore_clip(restorer, frames)
    elif len(frames) > 1 and restorer.runtime_features.use_recurrent_modular_runtime:
        result = restore_clip_recurrent(restorer, frames)
    else:
        result = restore_clip_sliding_window(restorer, frames)
    finalize_last_profile(restorer, total_s=perf_counter() - started_at)
    return result
