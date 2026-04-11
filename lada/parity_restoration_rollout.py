from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lada.extensions.vulkan.basicvsrpp_io import _build_output_frame_inputs
from lada.extensions.vulkan.basicvsrpp_recurrent_runtime import (
    run_branch_recurrent,
)
from lada.extensions.vulkan.basicvsrpp_restorer import (
    NcnnVulkanBasicvsrppMosaicRestorer,
)
from lada.parity_report import build_probe, quantize_unit_interval_output, to_numpy_array


def run_candidate_module(
    restorer: NcnnVulkanBasicvsrppMosaicRestorer,
    module_name: str,
    inputs: dict[str, Any],
) -> np.ndarray:
    if module_name == "spynet":
        return restorer.run_spynet(inputs["in0"], inputs["in1"], prefer_gpu_download=True)
    if module_name.endswith("_step"):
        return restorer.run_propagate_step(
            module_name.removesuffix("_step"),
            inputs,
            prefer_gpu_download=True,
        )
    return restorer._run_profiled_module(
        module_name,
        {name: to_numpy_array(value) for name, value in inputs.items()},
        bucket=None,
        prefer_gpu_download=True,
    )


def append_rollout_probes(
    probes: list[dict[str, Any]],
    *,
    branch_names: tuple[str, ...],
    probe_name: callable,
    variant_name: str,
    candidate_restorer: NcnnVulkanBasicvsrppMosaicRestorer,
    spatial_feats: list[object],
    flows_backward: list[object],
    flows_forward: list[object],
    reference_branch_feats: dict[str, list[torch.Tensor]],
    reference_output_frames: list[torch.Tensor],
    output_lqs: list[object],
    output_spatial_feats: list[object],
) -> dict[str, list[object]]:
    rollout_branch_feats: dict[str, list[object]] = {}
    for module_name in branch_names:
        flows = flows_backward if module_name.startswith("backward") else flows_forward
        rollout_outputs = run_branch_recurrent(
            candidate_restorer,
            module_name,
            spatial_feats,
            rollout_branch_feats,
            flows,
            use_gpu_bridge=False,
        )
        rollout_branch_feats[module_name] = rollout_outputs
        for frame_index, (reference_output, candidate_output) in enumerate(
            zip(reference_branch_feats[module_name], rollout_outputs, strict=True)
        ):
            probes.append(
                build_probe(
                    probe_name(f"{variant_name}/{module_name}/frame_{frame_index}"),
                    reference_output,
                    candidate_output,
                )
            )

    for frame_index, reference_output in enumerate(reference_output_frames):
        candidate_output = run_candidate_module(
            candidate_restorer,
            "output_frame",
            _build_output_frame_inputs(
                output_lqs,
                output_spatial_feats,
                rollout_branch_feats,
                frame_index,
            ),
        )
        probes.append(
            build_probe(
                probe_name(f"{variant_name}/output_frame/frame_{frame_index}"),
                reference_output,
                candidate_output,
            )
        )
        probes.append(
            build_probe(
                probe_name(f"{variant_name}/output_frame_quantized/frame_{frame_index}"),
                quantize_unit_interval_output(reference_output),
                quantize_unit_interval_output(candidate_output),
            )
        )
    return rollout_branch_feats
