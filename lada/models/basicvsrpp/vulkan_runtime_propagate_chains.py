# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import torch
import torch.nn as nn

from .vulkan_runtime import (
    BasicVsrppBackboneModule,
    prepare_backward1_backbone_input,
    prepare_backward2_backbone_input,
    prepare_forward1_backbone_input,
    prepare_forward2_backbone_input,
)
from .vulkan_runtime_align_chains import (
    BasicVsrppBackward1AlignChain,
    BasicVsrppBackward2AlignChain,
    BasicVsrppForward1AlignChain,
    BasicVsrppForward2AlignChain,
)


class BasicVsrppBackward1PropagateChain(nn.Module):
    """Extended ``backward_1`` chain with deform-align plus residual backbone."""

    def __init__(
        self,
        align_chain: BasicVsrppBackward1AlignChain,
        backbone: BasicVsrppBackboneModule,
    ):
        super().__init__()
        self.align_chain = align_chain
        self.backbone = backbone

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppBackward1PropagateChain":
        return cls(
            BasicVsrppBackward1AlignChain.from_model(
                model,
                clone_modules=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppBackboneModule.from_model(
                model,
                "backward_1",
                clone_module=clone_modules,
            ),
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
        aligned_feat_prop = self.align_chain(
            current_lq,
            next_lq,
            feat_prop,
            feat_current,
            feat_n2,
            prev_flow_n2,
        )
        backbone_input = prepare_backward1_backbone_input(
            feat_current,
            aligned_feat_prop,
        )
        return aligned_feat_prop + self.backbone(backbone_input)


class BasicVsrppForward1PropagateChain(nn.Module):
    """Extended ``forward_1`` chain with deform-align plus residual backbone."""

    def __init__(
        self,
        align_chain: BasicVsrppForward1AlignChain,
        backbone: BasicVsrppBackboneModule,
    ):
        super().__init__()
        self.align_chain = align_chain
        self.backbone = backbone

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppForward1PropagateChain":
        return cls(
            BasicVsrppForward1AlignChain.from_model(
                model,
                clone_modules=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppBackboneModule.from_model(
                model,
                "forward_1",
                clone_module=clone_modules,
            ),
        )

    def forward(
        self,
        current_lq: torch.Tensor,
        previous_lq: torch.Tensor,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        backward1_feat_current: torch.Tensor,
        feat_n2: torch.Tensor,
        prev_flow_n2: torch.Tensor,
    ) -> torch.Tensor:
        aligned_feat_prop = self.align_chain(
            current_lq,
            previous_lq,
            feat_prop,
            feat_current,
            feat_n2,
            prev_flow_n2,
        )
        backbone_input = prepare_forward1_backbone_input(
            feat_current,
            backward1_feat_current,
            aligned_feat_prop,
        )
        return aligned_feat_prop + self.backbone(backbone_input)


class BasicVsrppBackward2PropagateChain(nn.Module):
    """Extended ``backward_2`` chain with deform-align plus residual backbone."""

    def __init__(
        self,
        align_chain: BasicVsrppBackward2AlignChain,
        backbone: BasicVsrppBackboneModule,
    ):
        super().__init__()
        self.align_chain = align_chain
        self.backbone = backbone

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppBackward2PropagateChain":
        return cls(
            BasicVsrppBackward2AlignChain.from_model(
                model,
                clone_modules=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppBackboneModule.from_model(
                model,
                "backward_2",
                clone_module=clone_modules,
            ),
        )

    def forward(
        self,
        current_lq: torch.Tensor,
        next_lq: torch.Tensor,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        backward1_feat_current: torch.Tensor,
        forward1_feat_current: torch.Tensor,
        feat_n2: torch.Tensor,
        prev_flow_n2: torch.Tensor,
    ) -> torch.Tensor:
        aligned_feat_prop = self.align_chain(
            current_lq,
            next_lq,
            feat_prop,
            feat_current,
            feat_n2,
            prev_flow_n2,
        )
        backbone_input = prepare_backward2_backbone_input(
            feat_current,
            backward1_feat_current,
            forward1_feat_current,
            aligned_feat_prop,
        )
        return aligned_feat_prop + self.backbone(backbone_input)


class BasicVsrppForward2PropagateChain(nn.Module):
    """Extended ``forward_2`` chain with deform-align plus residual backbone."""

    def __init__(
        self,
        align_chain: BasicVsrppForward2AlignChain,
        backbone: BasicVsrppBackboneModule,
    ):
        super().__init__()
        self.align_chain = align_chain
        self.backbone = backbone

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppForward2PropagateChain":
        return cls(
            BasicVsrppForward2AlignChain.from_model(
                model,
                clone_modules=clone_modules,
                export_mode=export_mode,
            ),
            BasicVsrppBackboneModule.from_model(
                model,
                "forward_2",
                clone_module=clone_modules,
            ),
        )

    def forward(
        self,
        current_lq: torch.Tensor,
        previous_lq: torch.Tensor,
        feat_prop: torch.Tensor,
        feat_current: torch.Tensor,
        backward1_feat_current: torch.Tensor,
        forward1_feat_current: torch.Tensor,
        backward2_feat_current: torch.Tensor,
        feat_n2: torch.Tensor,
        prev_flow_n2: torch.Tensor,
    ) -> torch.Tensor:
        aligned_feat_prop = self.align_chain(
            current_lq,
            previous_lq,
            feat_prop,
            feat_current,
            feat_n2,
            prev_flow_n2,
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
    "BasicVsrppBackward1PropagateChain",
    "BasicVsrppForward1PropagateChain",
    "BasicVsrppBackward2PropagateChain",
    "BasicVsrppForward2PropagateChain",
]
