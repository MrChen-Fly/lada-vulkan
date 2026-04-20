# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import (
    ResidualBlocksWithInputConv,
)

from .basicvsrpp_vulkan_runtime_core import get_basicvsrpp_generator


class BasicVsrppBackboneModule(nn.Module):
    """Standalone propagation-backbone wrapper for module-wise runtime export."""

    def __init__(
        self,
        backbone: ResidualBlocksWithInputConv,
        *,
        clone_module: bool = True,
    ):
        super().__init__()
        self.backbone = deepcopy(backbone) if clone_module else backbone

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        module_name: str,
        *,
        clone_module: bool = True,
    ) -> "BasicVsrppBackboneModule":
        generator = get_basicvsrpp_generator(model)
        return cls(
            generator.backbone[module_name],
            clone_module=clone_module,
        )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if not features:
            raise RuntimeError("BasicVsrppBackboneModule expects at least one input tensor.")

        feat = features[0] if len(features) == 1 else torch.cat(features, dim=1)
        return self.backbone(feat)


class BasicVsrppReconstructionModule(nn.Module):
    """Standalone reconstruction wrapper for module-wise runtime export."""

    def __init__(
        self,
        reconstruction: ResidualBlocksWithInputConv,
        *,
        clone_module: bool = True,
    ):
        super().__init__()
        self.reconstruction = deepcopy(reconstruction) if clone_module else reconstruction

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_module: bool = True,
    ) -> "BasicVsrppReconstructionModule":
        generator = get_basicvsrpp_generator(model)
        return cls(
            generator.reconstruction,
            clone_module=clone_module,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.reconstruction(feat)


class BasicVsrppReconstructionConcatModule(nn.Module):
    """Standalone feature concat wrapper for GPU-side reconstruction input assembly."""

    def forward(
        self,
        spatial_feat: torch.Tensor,
        backward1_feat: torch.Tensor,
        forward1_feat: torch.Tensor,
        backward2_feat: torch.Tensor,
        forward2_feat: torch.Tensor,
    ) -> torch.Tensor:
        if spatial_feat.dim() == 4:
            return prepare_reconstruction_input(
                spatial_feat,
                backward1_feat,
                forward1_feat,
                backward2_feat,
                forward2_feat,
            )
        if spatial_feat.dim() == 3:
            return torch.cat(
                (
                    spatial_feat,
                    backward1_feat,
                    forward1_feat,
                    backward2_feat,
                    forward2_feat,
                ),
                dim=0,
            )
        raise RuntimeError(
            "BasicVsrppReconstructionConcatModule expects 3D or 4D feature tensors."
        )


class BasicVsrppOutputFrameModule(nn.Module):
    """Fuse reconstruction-input assembly and output head for one frame."""

    def __init__(
        self,
        output_head: "BasicVsrppOutputHeadModule",
        *,
        clone_module: bool = True,
    ):
        super().__init__()
        self.output_head = deepcopy(output_head) if clone_module else output_head

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
    ) -> "BasicVsrppOutputFrameModule":
        generator = get_basicvsrpp_generator(model)
        return cls(
            BasicVsrppOutputHeadModule.from_model(
                generator,
                clone_modules=clone_modules,
            ),
            clone_module=False,
        )

    def forward(
        self,
        current_lq: torch.Tensor,
        spatial_feat: torch.Tensor,
        backward1_feat: torch.Tensor,
        forward1_feat: torch.Tensor,
        backward2_feat: torch.Tensor,
        forward2_feat: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_feat = prepare_reconstruction_input(
            spatial_feat,
            backward1_feat,
            forward1_feat,
            backward2_feat,
            forward2_feat,
        )
        return self.output_head(current_lq, reconstruction_feat)


class BasicVsrppOutputHeadModule(nn.Module):
    """Standalone reconstruction-plus-upsample head for frame output."""

    def __init__(
        self,
        reconstruction: ResidualBlocksWithInputConv,
        upsample1: nn.Module,
        upsample2: nn.Module,
        conv_hr: nn.Module,
        conv_last: nn.Module,
        lrelu: nn.Module,
        *,
        clone_modules: bool = True,
    ):
        super().__init__()
        self.reconstruction = deepcopy(reconstruction) if clone_modules else reconstruction
        self.upsample1 = deepcopy(upsample1) if clone_modules else upsample1
        self.upsample2 = deepcopy(upsample2) if clone_modules else upsample2
        self.conv_hr = deepcopy(conv_hr) if clone_modules else conv_hr
        self.conv_last = deepcopy(conv_last) if clone_modules else conv_last
        self.lrelu = deepcopy(lrelu) if clone_modules else lrelu

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_modules: bool = True,
    ) -> "BasicVsrppOutputHeadModule":
        generator = get_basicvsrpp_generator(model)
        return cls(
            generator.reconstruction,
            generator.upsample1,
            generator.upsample2,
            generator.conv_hr,
            generator.conv_last,
            generator.lrelu,
            clone_modules=clone_modules,
        )

    def forward(
        self,
        current_lq: torch.Tensor,
        reconstruction_feat: torch.Tensor,
    ) -> torch.Tensor:
        hr = self.reconstruction(reconstruction_feat)
        hr = self.lrelu(self.upsample1(hr))
        hr = self.lrelu(self.upsample2(hr))
        hr = self.lrelu(self.conv_hr(hr))
        hr = self.conv_last(hr)
        return hr + current_lq


def prepare_propagation_backbone_input(
    feat_current: torch.Tensor,
    feat_prop: torch.Tensor,
    *,
    context_feats: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Build the per-branch residual-backbone input tensor."""
    return torch.cat([feat_current, *context_feats, feat_prop], dim=1)


def prepare_backward1_backbone_input(
    feat_current: torch.Tensor,
    feat_prop: torch.Tensor,
) -> torch.Tensor:
    """Build the residual-backbone input tensor for the ``backward_1`` branch."""
    return prepare_propagation_backbone_input(feat_current, feat_prop)


def prepare_forward1_backbone_input(
    feat_current: torch.Tensor,
    backward1_feat_current: torch.Tensor,
    feat_prop: torch.Tensor,
) -> torch.Tensor:
    """Build the residual-backbone input tensor for the ``forward_1`` branch."""
    return prepare_propagation_backbone_input(
        feat_current,
        feat_prop,
        context_feats=(backward1_feat_current,),
    )


def prepare_backward2_backbone_input(
    feat_current: torch.Tensor,
    backward1_feat_current: torch.Tensor,
    forward1_feat_current: torch.Tensor,
    feat_prop: torch.Tensor,
) -> torch.Tensor:
    """Build the residual-backbone input tensor for the ``backward_2`` branch."""
    return prepare_propagation_backbone_input(
        feat_current,
        feat_prop,
        context_feats=(backward1_feat_current, forward1_feat_current),
    )


def prepare_forward2_backbone_input(
    feat_current: torch.Tensor,
    backward1_feat_current: torch.Tensor,
    forward1_feat_current: torch.Tensor,
    backward2_feat_current: torch.Tensor,
    feat_prop: torch.Tensor,
) -> torch.Tensor:
    """Build the residual-backbone input tensor for the ``forward_2`` branch."""
    return prepare_propagation_backbone_input(
        feat_current,
        feat_prop,
        context_feats=(
            backward1_feat_current,
            forward1_feat_current,
            backward2_feat_current,
        ),
    )


def prepare_reconstruction_input(
    spatial_feat: torch.Tensor,
    backward1_feat: torch.Tensor,
    forward1_feat: torch.Tensor,
    backward2_feat: torch.Tensor,
    forward2_feat: torch.Tensor,
) -> torch.Tensor:
    """Build the five-branch feature tensor consumed by reconstruction."""
    return torch.cat(
        (
            spatial_feat,
            backward1_feat,
            forward1_feat,
            backward2_feat,
            forward2_feat,
        ),
        dim=1,
    )
