# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import torch
import torch.nn as nn

from .vulkan_runtime import (
    BasicVsrppBackboneModule,
    BasicVsrppDeformAlignModule,
    BasicVsrppFlowWarpModule,
    prepare_backward1_backbone_input,
    prepare_backward2_backbone_input,
    prepare_deform_align_inputs,
    prepare_forward1_backbone_input,
    prepare_forward2_backbone_input,
)


class BasicVsrppBackward1PropagateWithFlowChain(nn.Module):
    """``backward_1`` propagate chain that reuses a precomputed raw flow."""

    def __init__(
        self,
        deform_align: BasicVsrppDeformAlignModule,
        backbone: BasicVsrppBackboneModule,
        flow_warp_module: BasicVsrppFlowWarpModule | None = None,
    ):
        super().__init__()
        self.deform_align = deform_align
        self.backbone = backbone
        self.flow_warp_module = flow_warp_module or BasicVsrppFlowWarpModule()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppBackward1PropagateWithFlowChain":
        return cls(
            BasicVsrppDeformAlignModule.from_model(
                model,
                "backward_1",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppBackboneModule.from_model(
                model,
                "backward_1",
                clone_module=clone_modules,
            ),
            BasicVsrppFlowWarpModule(),
        )

    def forward(
        self,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
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
        aligned_feat_prop = self.deform_align(
            feature_pair,
            extra_feat,
            effective_flow_n1,
            effective_flow_n2,
        )
        backbone_input = prepare_backward1_backbone_input(
            feat_current,
            aligned_feat_prop,
        )
        return aligned_feat_prop + self.backbone(backbone_input)


class BasicVsrppForward1PropagateWithFlowChain(nn.Module):
    """``forward_1`` propagate chain that reuses a precomputed raw flow."""

    def __init__(
        self,
        deform_align: BasicVsrppDeformAlignModule,
        backbone: BasicVsrppBackboneModule,
        flow_warp_module: BasicVsrppFlowWarpModule | None = None,
    ):
        super().__init__()
        self.deform_align = deform_align
        self.backbone = backbone
        self.flow_warp_module = flow_warp_module or BasicVsrppFlowWarpModule()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppForward1PropagateWithFlowChain":
        return cls(
            BasicVsrppDeformAlignModule.from_model(
                model,
                "forward_1",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppBackboneModule.from_model(
                model,
                "forward_1",
                clone_module=clone_modules,
            ),
            BasicVsrppFlowWarpModule(),
        )

    def forward(
        self,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        backward1_feat_current: torch.Tensor,
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
        aligned_feat_prop = self.deform_align(
            feature_pair,
            extra_feat,
            effective_flow_n1,
            effective_flow_n2,
        )
        backbone_input = prepare_forward1_backbone_input(
            feat_current,
            backward1_feat_current,
            aligned_feat_prop,
        )
        return aligned_feat_prop + self.backbone(backbone_input)


class BasicVsrppBackward2PropagateWithFlowChain(nn.Module):
    """``backward_2`` propagate chain that reuses a precomputed raw flow."""

    def __init__(
        self,
        deform_align: BasicVsrppDeformAlignModule,
        backbone: BasicVsrppBackboneModule,
        flow_warp_module: BasicVsrppFlowWarpModule | None = None,
    ):
        super().__init__()
        self.deform_align = deform_align
        self.backbone = backbone
        self.flow_warp_module = flow_warp_module or BasicVsrppFlowWarpModule()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppBackward2PropagateWithFlowChain":
        return cls(
            BasicVsrppDeformAlignModule.from_model(
                model,
                "backward_2",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppBackboneModule.from_model(
                model,
                "backward_2",
                clone_module=clone_modules,
            ),
            BasicVsrppFlowWarpModule(),
        )

    def forward(
        self,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        backward1_feat_current: torch.Tensor,
        forward1_feat_current: torch.Tensor,
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
        aligned_feat_prop = self.deform_align(
            feature_pair,
            extra_feat,
            effective_flow_n1,
            effective_flow_n2,
        )
        backbone_input = prepare_backward2_backbone_input(
            feat_current,
            backward1_feat_current,
            forward1_feat_current,
            aligned_feat_prop,
        )
        return aligned_feat_prop + self.backbone(backbone_input)


class BasicVsrppForward2PropagateWithFlowChain(nn.Module):
    """``forward_2`` propagate chain that reuses a precomputed raw flow."""

    def __init__(
        self,
        deform_align: BasicVsrppDeformAlignModule,
        backbone: BasicVsrppBackboneModule,
        flow_warp_module: BasicVsrppFlowWarpModule | None = None,
    ):
        super().__init__()
        self.deform_align = deform_align
        self.backbone = backbone
        self.flow_warp_module = flow_warp_module or BasicVsrppFlowWarpModule()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppForward2PropagateWithFlowChain":
        return cls(
            BasicVsrppDeformAlignModule.from_model(
                model,
                "forward_2",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppBackboneModule.from_model(
                model,
                "forward_2",
                clone_module=clone_modules,
            ),
            BasicVsrppFlowWarpModule(),
        )

    def forward(
        self,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        backward1_feat_current: torch.Tensor,
        forward1_feat_current: torch.Tensor,
        backward2_feat_current: torch.Tensor,
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
        aligned_feat_prop = self.deform_align(
            feature_pair,
            extra_feat,
            effective_flow_n1,
            effective_flow_n2,
        )
        backbone_input = prepare_forward2_backbone_input(
            feat_current,
            backward1_feat_current,
            forward1_feat_current,
            backward2_feat_current,
            aligned_feat_prop,
        )
        return aligned_feat_prop + self.backbone(backbone_input)


__all__ = [
    "BasicVsrppBackward1PropagateWithFlowChain",
    "BasicVsrppForward1PropagateWithFlowChain",
    "BasicVsrppBackward2PropagateWithFlowChain",
    "BasicVsrppForward2PropagateWithFlowChain",
]
