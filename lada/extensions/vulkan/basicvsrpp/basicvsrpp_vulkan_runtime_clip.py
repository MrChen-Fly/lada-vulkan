# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import torch
import torch.nn as nn

from .basicvsrpp_vulkan_runtime_core import (
    BasicVsrppDeformAlignModule,
    BasicVsrppFlowWarpModule,
    get_basicvsrpp_generator,
    prepare_deform_align_inputs,
)
from .basicvsrpp_vulkan_runtime_heads import (
    BasicVsrppBackboneModule,
    BasicVsrppOutputHeadModule,
    prepare_backward1_backbone_input,
    prepare_backward2_backbone_input,
    prepare_forward1_backbone_input,
    prepare_forward2_backbone_input,
    prepare_reconstruction_input,
)

_BRANCH_NAMES = ("backward_1", "forward_1", "backward_2", "forward_2")
_CLIP_FRAME_COUNT = 5


class BasicVsrppFusedRestoreClipModule(nn.Module):
    """Fuse the 5-frame propagation tail into one fixed-shape clip module."""

    def __init__(
        self,
        *,
        backward_1_align: BasicVsrppDeformAlignModule,
        forward_1_align: BasicVsrppDeformAlignModule,
        backward_2_align: BasicVsrppDeformAlignModule,
        forward_2_align: BasicVsrppDeformAlignModule,
        backward_1_backbone: BasicVsrppBackboneModule,
        forward_1_backbone: BasicVsrppBackboneModule,
        backward_2_backbone: BasicVsrppBackboneModule,
        forward_2_backbone: BasicVsrppBackboneModule,
        output_head: BasicVsrppOutputHeadModule,
    ):
        super().__init__()
        self.deform_align = nn.ModuleDict(
            {
                "backward_1": backward_1_align,
                "forward_1": forward_1_align,
                "backward_2": backward_2_align,
                "forward_2": forward_2_align,
            }
        )
        self.backbone = nn.ModuleDict(
            {
                "backward_1": backward_1_backbone,
                "forward_1": forward_1_backbone,
                "backward_2": backward_2_backbone,
                "forward_2": forward_2_backbone,
            }
        )
        self.output_head = output_head
        self.flow_warp_module = BasicVsrppFlowWarpModule()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppFusedRestoreClipModule":
        generator = get_basicvsrpp_generator(model)
        return cls(
            backward_1_align=BasicVsrppDeformAlignModule.from_model(
                generator,
                "backward_1",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            forward_1_align=BasicVsrppDeformAlignModule.from_model(
                generator,
                "forward_1",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            backward_2_align=BasicVsrppDeformAlignModule.from_model(
                generator,
                "backward_2",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            forward_2_align=BasicVsrppDeformAlignModule.from_model(
                generator,
                "forward_2",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            backward_1_backbone=BasicVsrppBackboneModule.from_model(
                generator,
                "backward_1",
                clone_module=clone_modules,
            ),
            forward_1_backbone=BasicVsrppBackboneModule.from_model(
                generator,
                "forward_1",
                clone_module=clone_modules,
            ),
            backward_2_backbone=BasicVsrppBackboneModule.from_model(
                generator,
                "backward_2",
                clone_module=clone_modules,
            ),
            forward_2_backbone=BasicVsrppBackboneModule.from_model(
                generator,
                "forward_2",
                clone_module=clone_modules,
            ),
            output_head=BasicVsrppOutputHeadModule.from_model(
                generator,
                clone_modules=clone_modules,
            ),
        )

    def _run_backbone_step(
        self,
        module_name: str,
        feat_current: torch.Tensor,
        feat_prop: torch.Tensor,
        branch_feats: dict[str, list[torch.Tensor]],
        frame_index: int,
    ) -> torch.Tensor:
        if module_name == "backward_1":
            backbone_input = prepare_backward1_backbone_input(feat_current, feat_prop)
        elif module_name == "forward_1":
            backbone_input = prepare_forward1_backbone_input(
                feat_current,
                branch_feats["backward_1"][frame_index],
                feat_prop,
            )
        elif module_name == "backward_2":
            backbone_input = prepare_backward2_backbone_input(
                feat_current,
                branch_feats["backward_1"][frame_index],
                branch_feats["forward_1"][frame_index],
                feat_prop,
            )
        else:
            backbone_input = prepare_forward2_backbone_input(
                feat_current,
                branch_feats["backward_1"][frame_index],
                branch_feats["forward_1"][frame_index],
                branch_feats["backward_2"][frame_index],
                feat_prop,
            )
        return feat_prop + self.backbone[module_name](backbone_input)

    def _run_propagate_step(
        self,
        module_name: str,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        branch_feats: dict[str, list[torch.Tensor]],
        frame_index: int,
        feat_n2: torch.Tensor,
        flow_n1: torch.Tensor,
        prev_flow_n2: torch.Tensor,
    ) -> torch.Tensor:
        feature_pair, extra_feat, effective_flow_n1, effective_flow_n2 = prepare_deform_align_inputs(
            feat_prop,
            feat_current,
            flow_n1,
            flow_warp_module=self.flow_warp_module,
            feat_n2=feat_n2,
            flow_n2=prev_flow_n2,
        )
        aligned_feat_prop = self.deform_align[module_name](
            feature_pair,
            extra_feat,
            effective_flow_n1,
            effective_flow_n2,
        )

        if module_name == "backward_1":
            backbone_input = prepare_backward1_backbone_input(
                feat_current,
                aligned_feat_prop,
            )
        elif module_name == "forward_1":
            backbone_input = prepare_forward1_backbone_input(
                feat_current,
                branch_feats["backward_1"][frame_index],
                aligned_feat_prop,
            )
        elif module_name == "backward_2":
            backbone_input = prepare_backward2_backbone_input(
                feat_current,
                branch_feats["backward_1"][frame_index],
                branch_feats["forward_1"][frame_index],
                aligned_feat_prop,
            )
        else:
            backbone_input = prepare_forward2_backbone_input(
                feat_current,
                branch_feats["backward_1"][frame_index],
                branch_feats["forward_1"][frame_index],
                branch_feats["backward_2"][frame_index],
                aligned_feat_prop,
            )
        return aligned_feat_prop + self.backbone[module_name](backbone_input)

    def _run_branch(
        self,
        module_name: str,
        spatial_feats: list[torch.Tensor],
        flows: list[torch.Tensor],
        branch_feats: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        frame_indices = list(range(len(spatial_feats)))
        if module_name.startswith("backward"):
            frame_indices.reverse()

        feat_prop = torch.zeros_like(spatial_feats[0])
        zero_feat = torch.zeros_like(feat_prop)
        zero_flow = torch.zeros_like(flows[0])
        outputs: list[torch.Tensor] = []
        previous_raw_flow = zero_flow

        for step_index, frame_index in enumerate(frame_indices):
            feat_current = spatial_feats[frame_index]
            if step_index == 0:
                feat_prop = self._run_backbone_step(
                    module_name,
                    feat_current,
                    feat_prop,
                    branch_feats,
                    frame_index,
                )
                outputs.append(feat_prop)
                continue

            adjacent_index = frame_indices[step_index - 1]
            raw_flow_n1 = flows[min(frame_index, adjacent_index)]
            feat_n2 = outputs[-2] if step_index > 1 else zero_feat
            prev_flow_n2 = previous_raw_flow if step_index > 1 else zero_flow
            feat_prop = self._run_propagate_step(
                module_name,
                feat_prop,
                feat_current,
                branch_feats,
                frame_index,
                feat_n2,
                raw_flow_n1,
                prev_flow_n2,
            )
            outputs.append(feat_prop)
            previous_raw_flow = raw_flow_n1

        if module_name.startswith("backward"):
            outputs.reverse()
        return outputs

    def forward(
        self,
        lq0: torch.Tensor,
        lq1: torch.Tensor,
        lq2: torch.Tensor,
        lq3: torch.Tensor,
        lq4: torch.Tensor,
        spatial0: torch.Tensor,
        spatial1: torch.Tensor,
        spatial2: torch.Tensor,
        spatial3: torch.Tensor,
        spatial4: torch.Tensor,
        backward_flow0: torch.Tensor,
        backward_flow1: torch.Tensor,
        backward_flow2: torch.Tensor,
        backward_flow3: torch.Tensor,
        forward_flow0: torch.Tensor,
        forward_flow1: torch.Tensor,
        forward_flow2: torch.Tensor,
        forward_flow3: torch.Tensor,
    ) -> torch.Tensor:
        lqs = [lq0, lq1, lq2, lq3, lq4]
        spatial_feats = [spatial0, spatial1, spatial2, spatial3, spatial4]
        flows_backward = [backward_flow0, backward_flow1, backward_flow2, backward_flow3]
        flows_forward = [forward_flow0, forward_flow1, forward_flow2, forward_flow3]

        if len(lqs) != _CLIP_FRAME_COUNT or len(spatial_feats) != _CLIP_FRAME_COUNT:
            raise RuntimeError("BasicVsrppFusedRestoreClipModule expects exactly 5 frames.")

        branch_feats: dict[str, list[torch.Tensor]] = {}
        for module_name in _BRANCH_NAMES:
            flows = flows_backward if module_name.startswith("backward") else flows_forward
            branch_feats[module_name] = self._run_branch(
                module_name,
                spatial_feats,
                flows,
                branch_feats,
            )

        outputs: list[torch.Tensor] = []
        for frame_index in range(_CLIP_FRAME_COUNT):
            reconstruction_input = prepare_reconstruction_input(
                spatial_feats[frame_index],
                branch_feats["backward_1"][frame_index],
                branch_feats["forward_1"][frame_index],
                branch_feats["backward_2"][frame_index],
                branch_feats["forward_2"][frame_index],
            )
            outputs.append(self.output_head(lqs[frame_index], reconstruction_input))
        return torch.cat(outputs, dim=1)
