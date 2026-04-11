from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lada.models.basicvsrpp.vulkan_runtime import (
    BasicVsrppBackboneModule,
    BasicVsrppFeatExtractModule,
    BasicVsrppOutputFrameModule,
    BasicVsrppQuarterDownsampleModule,
    BasicVsrppSpynetFlowModule,
)
from lada.models.basicvsrpp.vulkan_runtime_propagate_with_flow_chains import (
    BasicVsrppBackward1PropagateWithFlowChain,
    BasicVsrppBackward2PropagateWithFlowChain,
    BasicVsrppForward1PropagateWithFlowChain,
    BasicVsrppForward2PropagateWithFlowChain,
)
from lada.parity_report import build_probe, quantize_unit_interval_output
from lada.parity_restoration_rollout import (
    append_rollout_probes,
    run_candidate_module,
)
from lada.extensions.vulkan.basicvsrpp_io import (
    _build_backbone_inputs,
    _build_output_frame_inputs,
    _build_step_inputs,
    _frame_to_chw_float32,
)
from lada.extensions.vulkan.basicvsrpp_restorer import (
    NcnnVulkanBasicvsrppMosaicRestorer,
)

_BRANCH_NAMES = ("backward_1", "forward_1", "backward_2", "forward_2")


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return np.ascontiguousarray(value).copy()


def _frame_to_reference_tensor(frame: Any, device: torch.device) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach().clone()
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(frame).copy())
    return (
        tensor.permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def build_reference_basicvsrpp_modules(
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    return {
        "quarter_downsample": BasicVsrppQuarterDownsampleModule().to(device).eval(),
        "feat_extract": BasicVsrppFeatExtractModule.from_model(model).to(device).eval(),
        "spynet": BasicVsrppSpynetFlowModule.from_model(model).to(device).eval(),
        "backward_1_backbone": BasicVsrppBackboneModule.from_model(model, "backward_1").to(device).eval(),
        "forward_1_backbone": BasicVsrppBackboneModule.from_model(model, "forward_1").to(device).eval(),
        "backward_2_backbone": BasicVsrppBackboneModule.from_model(model, "backward_2").to(device).eval(),
        "forward_2_backbone": BasicVsrppBackboneModule.from_model(model, "forward_2").to(device).eval(),
        "backward_1_step": BasicVsrppBackward1PropagateWithFlowChain.from_model(model).to(device).eval(),
        "forward_1_step": BasicVsrppForward1PropagateWithFlowChain.from_model(model).to(device).eval(),
        "backward_2_step": BasicVsrppBackward2PropagateWithFlowChain.from_model(model).to(device).eval(),
        "forward_2_step": BasicVsrppForward2PropagateWithFlowChain.from_model(model).to(device).eval(),
        "output_frame": BasicVsrppOutputFrameModule.from_model(model).to(device).eval(),
    }
def _probe_name(prefix: str | None, name: str) -> str:
    if not prefix:
        return name
    return f"{prefix}/{name}"


def build_restoration_core_probes(
    frames: list[Any],
    *,
    reference_device: torch.device,
    reference_modules: dict[str, torch.nn.Module],
    reference_restorer: Any,
    candidate_restorer: NcnnVulkanBasicvsrppMosaicRestorer,
    probe_prefix: str | None = None,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with torch.inference_mode():
        reference_lqs = [_frame_to_reference_tensor(frame, reference_device) for frame in frames]
        candidate_lqs = [_frame_to_chw_float32(_clone_value(frame)) for frame in frames]
        quarter = []
        spatial = []
        candidate_spatial_feats: list[np.ndarray] = []
        for index, (reference_lq, candidate_lq) in enumerate(zip(reference_lqs, candidate_lqs, strict=True)):
            reference_quarter = reference_modules["quarter_downsample"](reference_lq)
            candidate_quarter = run_candidate_module(
                candidate_restorer,
                "quarter_downsample",
                {"in0": candidate_lq},
            )
            probes.append(
                build_probe(
                    _probe_name(probe_prefix, f"quarter_downsample/frame_{index}"),
                    reference_quarter,
                    candidate_quarter,
                )
            )
            quarter.append(reference_quarter)
            reference_spatial = reference_modules["feat_extract"](reference_lq)
            candidate_spatial = run_candidate_module(
                candidate_restorer,
                "feat_extract",
                {"in0": candidate_lq},
            )
            probes.append(
                build_probe(
                    _probe_name(probe_prefix, f"feat_extract/frame_{index}"),
                    reference_spatial,
                    candidate_spatial,
                )
            )
            spatial.append(reference_spatial)
            candidate_spatial_feats.append(candidate_spatial)

        flows_backward = []
        flows_forward = []
        candidate_flows_backward: list[np.ndarray] = []
        candidate_flows_forward: list[np.ndarray] = []
        for index in range(len(quarter) - 1):
            reference_backward = reference_modules["spynet"](quarter[index], quarter[index + 1])
            candidate_backward = run_candidate_module(
                candidate_restorer,
                "spynet",
                {"in0": quarter[index], "in1": quarter[index + 1]},
            )
            probes.append(
                build_probe(
                    _probe_name(probe_prefix, f"spynet/backward_{index}"),
                    reference_backward,
                    candidate_backward,
                )
            )
            flows_backward.append(reference_backward)
            candidate_flows_backward.append(candidate_backward)
            reference_forward = reference_modules["spynet"](quarter[index + 1], quarter[index])
            candidate_forward = run_candidate_module(
                candidate_restorer,
                "spynet",
                {"in0": quarter[index + 1], "in1": quarter[index]},
            )
            probes.append(
                build_probe(
                    _probe_name(probe_prefix, f"spynet/forward_{index}"),
                    reference_forward,
                    candidate_forward,
                )
            )
            flows_forward.append(reference_forward)
            candidate_flows_forward.append(candidate_forward)

        branch_feats: dict[str, list[torch.Tensor]] = {}
        for module_name in _BRANCH_NAMES:
            frame_indices = list(range(len(spatial)))
            if module_name.startswith("backward"):
                frame_indices.reverse()
            feat_prop = torch.zeros_like(spatial[0])
            zero_feat = torch.zeros_like(spatial[0])
            zero_flow = torch.zeros_like(flows_backward[0])
            outputs: list[torch.Tensor] = []
            previous_raw_flow = zero_flow
            for step_index, frame_index in enumerate(frame_indices):
                feat_current = spatial[frame_index]
                if step_index == 0:
                    backbone_inputs = _build_backbone_inputs(
                        module_name,
                        feat_current,
                        feat_prop,
                        branch_feats,
                        frame_index,
                    )
                    reference_output = reference_modules[f"{module_name}_backbone"](
                        *backbone_inputs.values()
                    )
                    candidate_output = run_candidate_module(
                        candidate_restorer,
                        f"{module_name}_backbone",
                        backbone_inputs,
                    )
                    probes.append(
                        build_probe(
                            _probe_name(
                                probe_prefix,
                                f"{module_name}_backbone/frame_{frame_index}",
                            ),
                            reference_output,
                            candidate_output,
                        )
                    )
                    feat_prop = reference_output
                    outputs.append(feat_prop)
                    continue

                adjacent_index = frame_indices[step_index - 1]
                raw_flow_n1 = (flows_backward if module_name.startswith("backward") else flows_forward)[
                    min(frame_index, adjacent_index)
                ]
                feat_n2 = outputs[-2] if step_index > 1 else zero_feat
                prev_flow_n2 = previous_raw_flow if step_index > 1 else zero_flow
                step_inputs = _build_step_inputs(
                    module_name,
                    feat_prop,
                    feat_current,
                    branch_feats,
                    frame_index,
                    feat_n2,
                    raw_flow_n1,
                    prev_flow_n2,
                )
                reference_output = reference_modules[f"{module_name}_step"](*step_inputs.values())
                candidate_output = run_candidate_module(
                    candidate_restorer,
                    f"{module_name}_step",
                    step_inputs,
                )
                probes.append(
                    build_probe(
                        _probe_name(probe_prefix, f"{module_name}_step/frame_{frame_index}"),
                        reference_output,
                        candidate_output,
                    )
                )
                feat_prop = reference_output
                outputs.append(feat_prop)
                previous_raw_flow = raw_flow_n1

            if module_name.startswith("backward"):
                outputs.reverse()
            branch_feats[module_name] = outputs

        reference_output_frames: list[torch.Tensor] = []
        for frame_index in range(len(reference_lqs)):
            output_inputs = _build_output_frame_inputs(
                reference_lqs,
                spatial,
                branch_feats,
                frame_index,
            )
            reference_output = reference_modules["output_frame"](*output_inputs.values())
            reference_output_frames.append(reference_output)
            candidate_output = run_candidate_module(
                candidate_restorer,
                "output_frame",
                output_inputs,
            )
            probes.append(
                build_probe(
                    _probe_name(probe_prefix, f"output_frame/frame_{frame_index}"),
                    reference_output,
                    candidate_output,
                )
            )
            probes.append(
                build_probe(
                    _probe_name(probe_prefix, f"output_frame_quantized/frame_{frame_index}"),
                    quantize_unit_interval_output(reference_output),
                    quantize_unit_interval_output(candidate_output),
                )
            )

        append_rollout_probes(
            probes,
            branch_names=_BRANCH_NAMES,
            probe_name=lambda name: _probe_name(probe_prefix, name),
            variant_name="rollout",
            candidate_restorer=candidate_restorer,
            spatial_feats=spatial,
            flows_backward=flows_backward,
            flows_forward=flows_forward,
            reference_branch_feats=branch_feats,
            reference_output_frames=reference_output_frames,
            output_lqs=reference_lqs,
            output_spatial_feats=spatial,
        )
        append_rollout_probes(
            probes,
            branch_names=_BRANCH_NAMES,
            probe_name=lambda name: _probe_name(probe_prefix, name),
            variant_name="rollout_candidate_spatial",
            candidate_restorer=candidate_restorer,
            spatial_feats=candidate_spatial_feats,
            flows_backward=flows_backward,
            flows_forward=flows_forward,
            reference_branch_feats=branch_feats,
            reference_output_frames=reference_output_frames,
            output_lqs=candidate_lqs,
            output_spatial_feats=candidate_spatial_feats,
        )
        append_rollout_probes(
            probes,
            branch_names=_BRANCH_NAMES,
            probe_name=lambda name: _probe_name(probe_prefix, name),
            variant_name="rollout_candidate_flow",
            candidate_restorer=candidate_restorer,
            spatial_feats=spatial,
            flows_backward=candidate_flows_backward,
            flows_forward=candidate_flows_forward,
            reference_branch_feats=branch_feats,
            reference_output_frames=reference_output_frames,
            output_lqs=reference_lqs,
            output_spatial_feats=spatial,
        )
        append_rollout_probes(
            probes,
            branch_names=_BRANCH_NAMES,
            probe_name=lambda name: _probe_name(probe_prefix, name),
            variant_name="rollout_candidate_backward_flow",
            candidate_restorer=candidate_restorer,
            spatial_feats=spatial,
            flows_backward=candidate_flows_backward,
            flows_forward=flows_forward,
            reference_branch_feats=branch_feats,
            reference_output_frames=reference_output_frames,
            output_lqs=reference_lqs,
            output_spatial_feats=spatial,
        )
        append_rollout_probes(
            probes,
            branch_names=_BRANCH_NAMES,
            probe_name=lambda name: _probe_name(probe_prefix, name),
            variant_name="rollout_candidate_forward_flow",
            candidate_restorer=candidate_restorer,
            spatial_feats=spatial,
            flows_backward=flows_backward,
            flows_forward=candidate_flows_forward,
            reference_branch_feats=branch_feats,
            reference_output_frames=reference_output_frames,
            output_lqs=reference_lqs,
            output_spatial_feats=spatial,
        )
        append_rollout_probes(
            probes,
            branch_names=_BRANCH_NAMES,
            probe_name=lambda name: _probe_name(probe_prefix, name),
            variant_name="rollout_candidate_input",
            candidate_restorer=candidate_restorer,
            spatial_feats=candidate_spatial_feats,
            flows_backward=candidate_flows_backward,
            flows_forward=candidate_flows_forward,
            reference_branch_feats=branch_feats,
            reference_output_frames=reference_output_frames,
            output_lqs=candidate_lqs,
            output_spatial_feats=candidate_spatial_feats,
        )

        try:
            reference_restore_frames = reference_restorer.restore(
                [
                    _clone_value(frame)
                    if isinstance(frame, torch.Tensor)
                    else torch.from_numpy(np.ascontiguousarray(frame).copy())
                    for frame in frames
                ]
            )
            candidate_restore_frames = candidate_restorer.restore(
                [_clone_value(frame) for frame in frames]
            )
            for index, (reference_frame, candidate_frame) in enumerate(
                zip(reference_restore_frames, candidate_restore_frames, strict=True)
            ):
                probes.append(
                    build_probe(
                        _probe_name(probe_prefix, f"restore/frame_{index}"),
                        reference_frame,
                        candidate_frame,
                    )
                )
        except RuntimeError as exc:
            errors.append(
                {
                    "probe": _probe_name(probe_prefix, "restore"),
                    "error": str(exc),
                }
            )

    return {
        "probes": probes,
        "errors": errors,
    }
