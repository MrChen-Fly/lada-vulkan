# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import torch
import torch.nn as nn

from .vulkan_runtime import (
    BasicVsrppDeformAlignModule,
    BasicVsrppFlowWarpModule,
    BasicVsrppSpynetFlowModule,
    prepare_deform_align_inputs,
)


class BasicVsrppBackward1AlignChain(nn.Module):
    """First executable module chain for the staged Vulkan runtime."""

    def __init__(
        self,
        spynet_flow: BasicVsrppSpynetFlowModule,
        deform_align: BasicVsrppDeformAlignModule,
        flow_warp_module: BasicVsrppFlowWarpModule | None = None,
    ):
        super().__init__()
        self.spynet_flow = spynet_flow
        self.deform_align = deform_align
        self.flow_warp_module = flow_warp_module or BasicVsrppFlowWarpModule()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppBackward1AlignChain":
        return cls(
            BasicVsrppSpynetFlowModule.from_model(
                model,
                clone_module=clone_modules,
            ),
            BasicVsrppDeformAlignModule.from_model(
                model,
                "backward_1",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppFlowWarpModule(),
        )

    def forward(
        self,
        current_lq: torch.Tensor,
        next_lq: torch.Tensor,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        feat_n2: torch.Tensor,
        prev_flow_n2: torch.Tensor,
    ) -> torch.Tensor:
        flow_n1 = self.spynet_flow(current_lq, next_lq)
        feature_pair, extra_feat, effective_flow_n1, effective_flow_n2 = prepare_deform_align_inputs(
            feat_prop,
            feat_current,
            flow_n1,
            flow_warp_module=self.flow_warp_module,
            feat_n2=feat_n2,
            flow_n2=prev_flow_n2,
        )
        return self.deform_align(
            feature_pair,
            extra_feat,
            effective_flow_n1,
            effective_flow_n2,
        )


class BasicVsrppForward1AlignChain(nn.Module):
    """First forward-time align chain for the staged Vulkan runtime."""

    def __init__(
        self,
        spynet_flow: BasicVsrppSpynetFlowModule,
        deform_align: BasicVsrppDeformAlignModule,
        flow_warp_module: BasicVsrppFlowWarpModule | None = None,
    ):
        super().__init__()
        self.spynet_flow = spynet_flow
        self.deform_align = deform_align
        self.flow_warp_module = flow_warp_module or BasicVsrppFlowWarpModule()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppForward1AlignChain":
        return cls(
            BasicVsrppSpynetFlowModule.from_model(
                model,
                clone_module=clone_modules,
            ),
            BasicVsrppDeformAlignModule.from_model(
                model,
                "forward_1",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppFlowWarpModule(),
        )

    def forward(
        self,
        current_lq: torch.Tensor,
        previous_lq: torch.Tensor,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        feat_n2: torch.Tensor,
        prev_flow_n2: torch.Tensor,
    ) -> torch.Tensor:
        flow_n1 = self.spynet_flow(current_lq, previous_lq)
        feature_pair, extra_feat, effective_flow_n1, effective_flow_n2 = prepare_deform_align_inputs(
            feat_prop,
            feat_current,
            flow_n1,
            flow_warp_module=self.flow_warp_module,
            feat_n2=feat_n2,
            flow_n2=prev_flow_n2,
        )
        return self.deform_align(
            feature_pair,
            extra_feat,
            effective_flow_n1,
            effective_flow_n2,
        )


class BasicVsrppBackward2AlignChain(nn.Module):
    """Second backward-time align chain for the staged Vulkan runtime."""

    def __init__(
        self,
        spynet_flow: BasicVsrppSpynetFlowModule,
        deform_align: BasicVsrppDeformAlignModule,
        flow_warp_module: BasicVsrppFlowWarpModule | None = None,
    ):
        super().__init__()
        self.spynet_flow = spynet_flow
        self.deform_align = deform_align
        self.flow_warp_module = flow_warp_module or BasicVsrppFlowWarpModule()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppBackward2AlignChain":
        return cls(
            BasicVsrppSpynetFlowModule.from_model(
                model,
                clone_module=clone_modules,
            ),
            BasicVsrppDeformAlignModule.from_model(
                model,
                "backward_2",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppFlowWarpModule(),
        )

    def forward(
        self,
        current_lq: torch.Tensor,
        next_lq: torch.Tensor,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        feat_n2: torch.Tensor,
        prev_flow_n2: torch.Tensor,
    ) -> torch.Tensor:
        flow_n1 = self.spynet_flow(current_lq, next_lq)
        feature_pair, extra_feat, effective_flow_n1, effective_flow_n2 = prepare_deform_align_inputs(
            feat_prop,
            feat_current,
            flow_n1,
            flow_warp_module=self.flow_warp_module,
            feat_n2=feat_n2,
            flow_n2=prev_flow_n2,
        )
        return self.deform_align(
            feature_pair,
            extra_feat,
            effective_flow_n1,
            effective_flow_n2,
        )


class BasicVsrppForward2AlignChain(nn.Module):
    """Second forward-time align chain for the staged Vulkan runtime."""

    def __init__(
        self,
        spynet_flow: BasicVsrppSpynetFlowModule,
        deform_align: BasicVsrppDeformAlignModule,
        flow_warp_module: BasicVsrppFlowWarpModule | None = None,
    ):
        super().__init__()
        self.spynet_flow = spynet_flow
        self.deform_align = deform_align
        self.flow_warp_module = flow_warp_module or BasicVsrppFlowWarpModule()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppForward2AlignChain":
        return cls(
            BasicVsrppSpynetFlowModule.from_model(
                model,
                clone_module=clone_modules,
            ),
            BasicVsrppDeformAlignModule.from_model(
                model,
                "forward_2",
                clone_module=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppFlowWarpModule(),
        )

    def forward(
        self,
        current_lq: torch.Tensor,
        previous_lq: torch.Tensor,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        feat_n2: torch.Tensor,
        prev_flow_n2: torch.Tensor,
    ) -> torch.Tensor:
        flow_n1 = self.spynet_flow(current_lq, previous_lq)
        feature_pair, extra_feat, effective_flow_n1, effective_flow_n2 = prepare_deform_align_inputs(
            feat_prop,
            feat_current,
            flow_n1,
            flow_warp_module=self.flow_warp_module,
            feat_n2=feat_n2,
            flow_n2=prev_flow_n2,
        )
        return self.deform_align(
            feature_pair,
            extra_feat,
            effective_flow_n1,
            effective_flow_n2,
        )


__all__ = [
    "BasicVsrppBackward1AlignChain",
    "BasicVsrppForward1AlignChain",
    "BasicVsrppBackward2AlignChain",
    "BasicVsrppForward2AlignChain",
]
