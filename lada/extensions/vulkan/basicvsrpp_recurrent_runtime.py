from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch

from lada.extensions.vulkan.basicvsrpp_ncnn_runtime import is_ncnn_vulkan_tensor
from lada.models.basicvsrpp.mmagic.flow_warp import flow_warp as torch_flow_warp

from lada.extensions.vulkan.basicvsrpp_cpu_extractor import (
    run_cpu_extractor_module,
    runtime_value_to_numpy,
)
from lada.extensions.vulkan.basicvsrpp_io import (
    _build_backbone_inputs,
    _build_step_inputs,
    _zeros_like_runtime_value,
)

if TYPE_CHECKING:
    from lada.extensions.vulkan.basicvsrpp_restorer import (
        NcnnVulkanBasicvsrppMosaicRestorer,
    )


_DEFORM_ALIGN_RUNNERS = {
    "backward_1": "backward_1_deform_align",
    "forward_1": "forward_1_deform_align",
    "backward_2": "backward_2_deform_align",
    "forward_2": "forward_2_deform_align",
}


def run_flow_warp(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    feature: object,
    flow: object,
    *,
    bucket: str | None = None,
    prefer_gpu_download: bool = False,
) -> np.ndarray:
    """Run the standalone ``flow_warp`` module and return a CPU tensor."""
    del prefer_gpu_download
    feature_np = runtime_value_to_numpy(restorer, feature)
    flow_np = runtime_value_to_numpy(restorer, flow)
    warped = run_cpu_extractor_module(
        restorer,
        "flow_warp",
        {
            "in0": feature_np,
            "in1": flow_np,
        },
        bucket=bucket,
    )
    if warped.shape == feature_np.shape:
        return warped

    with restorer.profiler.measure("cpu_flow_warp_fallback_s"):
        feature_tensor = torch.from_numpy(feature_np).unsqueeze(0)
        flow_tensor = torch.from_numpy(flow_np).unsqueeze(0)
        fallback = torch_flow_warp(
            feature_tensor,
            flow_tensor.permute(0, 2, 3, 1),
            interpolation="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    return np.ascontiguousarray(fallback.squeeze(0).numpy(), dtype=np.float32)


def run_deform_align(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    module_name: str,
    feature_pair: object,
    extra_feat: object,
    flow_n1: object,
    flow_n2: object,
    *,
    bucket: str | None = None,
    prefer_gpu_download: bool = False,
) -> np.ndarray:
    """Run one branch deform-align module and return a CPU tensor."""
    del prefer_gpu_download
    runner_name = _DEFORM_ALIGN_RUNNERS.get(module_name)
    if runner_name is None:
        raise RuntimeError(f"Unsupported deform-align branch '{module_name}'.")
    return run_cpu_extractor_module(
        restorer,
        runner_name,
        {
            "in0": runtime_value_to_numpy(restorer, feature_pair),
            "in1": runtime_value_to_numpy(restorer, extra_feat),
            "in2": runtime_value_to_numpy(restorer, flow_n1),
            "in3": runtime_value_to_numpy(restorer, flow_n2),
        },
        bucket=bucket,
    )


def _parse_step_inputs(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    module_name: str,
    step_inputs: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    feat_prop = runtime_value_to_numpy(restorer, step_inputs["in0"])
    feat_current = runtime_value_to_numpy(restorer, step_inputs["in1"])

    if module_name == "backward_1":
        feat_n2 = runtime_value_to_numpy(restorer, step_inputs["in2"])
        flow_n1 = runtime_value_to_numpy(restorer, step_inputs["in3"])
        prev_flow_n2 = runtime_value_to_numpy(restorer, step_inputs["in4"])
        return feat_prop, feat_current, {}, feat_n2, flow_n1, prev_flow_n2

    if module_name == "forward_1":
        branch_feats = {
            "backward_1": [runtime_value_to_numpy(restorer, step_inputs["in2"])],
        }
        feat_n2 = runtime_value_to_numpy(restorer, step_inputs["in3"])
        flow_n1 = runtime_value_to_numpy(restorer, step_inputs["in4"])
        prev_flow_n2 = runtime_value_to_numpy(restorer, step_inputs["in5"])
        return feat_prop, feat_current, branch_feats, feat_n2, flow_n1, prev_flow_n2

    if module_name == "backward_2":
        branch_feats = {
            "backward_1": [runtime_value_to_numpy(restorer, step_inputs["in2"])],
            "forward_1": [runtime_value_to_numpy(restorer, step_inputs["in3"])],
        }
        feat_n2 = runtime_value_to_numpy(restorer, step_inputs["in4"])
        flow_n1 = runtime_value_to_numpy(restorer, step_inputs["in5"])
        prev_flow_n2 = runtime_value_to_numpy(restorer, step_inputs["in6"])
        return feat_prop, feat_current, branch_feats, feat_n2, flow_n1, prev_flow_n2

    if module_name == "forward_2":
        branch_feats = {
            "backward_1": [runtime_value_to_numpy(restorer, step_inputs["in2"])],
            "forward_1": [runtime_value_to_numpy(restorer, step_inputs["in3"])],
            "backward_2": [runtime_value_to_numpy(restorer, step_inputs["in4"])],
        }
        feat_n2 = runtime_value_to_numpy(restorer, step_inputs["in5"])
        flow_n1 = runtime_value_to_numpy(restorer, step_inputs["in6"])
        prev_flow_n2 = runtime_value_to_numpy(restorer, step_inputs["in7"])
        return feat_prop, feat_current, branch_feats, feat_n2, flow_n1, prev_flow_n2

    raise RuntimeError(f"Unsupported propagation branch '{module_name}'.")


def run_propagate_step(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    module_name: str,
    step_inputs: dict[str, object],
    *,
    bucket: str | None = None,
    prefer_gpu: bool = False,
    prefer_gpu_download: bool = False,
) -> object:
    """Run one exported BasicVSR++ propagation-step subgraph."""
    return restorer._run_profiled_module(
        f"{module_name}_step",
        step_inputs,
        bucket=bucket,
        prefer_gpu=prefer_gpu,
        prefer_gpu_download=prefer_gpu_download,
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
    """Run one recurrent branch through the modular step helper."""
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
                prefer_gpu_download=not use_gpu_bridge,
            )
            outputs.append(feat_prop)
            continue

        adjacent_index = frame_indices[step_index - 1]
        raw_flow_n1 = flows[min(frame_index, adjacent_index)]
        feat_n2 = outputs[-2] if step_index > 1 else zero_feat
        prev_flow_n2 = previous_raw_flow if step_index > 1 else zero_flow
        feat_prop = restorer.run_propagate_step(
            module_name,
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
            prefer_gpu_download=not use_gpu_bridge,
        )
        outputs.append(feat_prop)
        previous_raw_flow = raw_flow_n1

    if module_name.startswith("backward"):
        outputs.reverse()
    return outputs


def merge_native_clip_profile(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
) -> None:
    """Merge profile counters from the native clip runner into the Python snapshot."""
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


def finalize_last_profile(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    *,
    total_s: float,
) -> None:
    """Freeze the current profiling snapshot for external inspection."""
    restorer.last_profile = restorer.profiler.snapshot(total_s=total_s)
    restorer.last_profile.update(restorer._native_clip_profile_snapshot)
