# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deformconv import ModulatedDeformConv2d
from .mmagic.basicvsr_plusplus_net import (
    BasicVSRPlusPlusNet,
    ResidualBlocksWithInputConv,
    SPyNet,
    SecondOrderDeformableAlignment,
)
from .mmagic.flow_warp import flow_warp
from .vulkan_param_patch import patch_ncnn_param_for_vulkan_runtime


def get_basicvsrpp_generator(model: nn.Module) -> BasicVSRPlusPlusNet:
    """Resolve the actual BasicVSR++ generator from a top-level model wrapper."""
    if isinstance(model, BasicVSRPlusPlusNet):
        return model

    generator_ema = getattr(model, "generator_ema", None)
    if isinstance(generator_ema, BasicVSRPlusPlusNet):
        return generator_ema

    generator = getattr(model, "generator", None)
    if isinstance(generator, BasicVSRPlusPlusNet):
        return generator

    raise TypeError(
        f"Unsupported BasicVSR++ container type '{type(model).__name__}'."
    )


def set_deformconv_export_mode(module: nn.Module, enabled: bool) -> None:
    """Toggle export mode on all deform-conv modules inside ``module``."""
    for child in module.modules():
        if isinstance(child, ModulatedDeformConv2d):
            child.export_mode = enabled


class BasicVsrppSpynetFlowModule(nn.Module):
    """Standalone SPyNet wrapper for module-wise runtime export."""

    def __init__(self, spynet: SPyNet, *, clone_module: bool = True):
        super().__init__()
        self.spynet = deepcopy(spynet) if clone_module else spynet

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_module: bool = True,
    ) -> "BasicVsrppSpynetFlowModule":
        generator = get_basicvsrpp_generator(model)
        return cls(generator.spynet, clone_module=clone_module)

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        return self.spynet(ref, supp)


class BasicVsrppSpynetComputeFlowModule(nn.Module):
    """Export only the SPyNet pyramid core without the outer resize wrapper."""

    def __init__(self, spynet: SPyNet, *, clone_module: bool = True):
        super().__init__()
        self.spynet = deepcopy(spynet) if clone_module else spynet

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_module: bool = True,
    ) -> "BasicVsrppSpynetComputeFlowModule":
        generator = get_basicvsrpp_generator(model)
        return cls(generator.spynet, clone_module=clone_module)

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        return self.spynet.compute_flow(ref, supp)


class BasicVsrppFlowWarpModule(nn.Module):
    """Standalone wrapper around ``flow_warp`` for runtime assembly."""

    def __init__(
        self,
        *,
        interpolation: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return flow_warp(
            x,
            flow.permute(0, 2, 3, 1),
            interpolation=self.interpolation,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


class BasicVsrppQuarterDownsampleModule(nn.Module):
    """Quarter-resolution bicubic downsample used before SPyNet flow inference."""

    def forward(self, current_lq: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            current_lq,
            scale_factor=0.25,
            mode="bicubic",
        )


class BasicVsrppFeatExtractModule(nn.Module):
    """Standalone spatial-feature extractor for module-wise runtime export."""

    def __init__(
        self,
        feat_extract: nn.Module,
        *,
        clone_module: bool = True,
    ):
        super().__init__()
        self.feat_extract = deepcopy(feat_extract) if clone_module else feat_extract

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        clone_module: bool = True,
    ) -> "BasicVsrppFeatExtractModule":
        generator = get_basicvsrpp_generator(model)
        return cls(generator.feat_extract, clone_module=clone_module)

    def forward(self, current_lq: torch.Tensor) -> torch.Tensor:
        return self.feat_extract(current_lq)


def prepare_deform_align_inputs(
    feat_prop: torch.Tensor,
    feat_current: torch.Tensor,
    flow_n1: torch.Tensor,
    *,
    flow_warp_module: BasicVsrppFlowWarpModule,
    feat_n2: torch.Tensor | None = None,
    flow_n2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the four tensors consumed by ``SecondOrderDeformableAlignment``."""
    if feat_n2 is None:
        feat_n2 = torch.zeros_like(feat_prop)

    if flow_n2 is None:
        effective_flow_n2 = torch.zeros_like(flow_n1)
        cond_n2 = torch.zeros_like(feat_prop)
    else:
        effective_flow_n2 = flow_n1 + flow_warp_module(flow_n2, flow_n1)
        cond_n2 = flow_warp_module(feat_n2, effective_flow_n2)

    cond_n1 = flow_warp_module(feat_prop, flow_n1)
    extra_feat = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
    feature_pair = torch.cat([feat_prop, feat_n2], dim=1)
    return feature_pair, extra_feat, flow_n1, effective_flow_n2


class BasicVsrppDeformAlignModule(nn.Module):
    """Standalone deform-align wrapper for module-wise runtime export."""

    def __init__(
        self,
        deform_align: SecondOrderDeformableAlignment,
        *,
        clone_module: bool = True,
        export_mode: bool = False,
    ):
        super().__init__()
        self.deform_align = deepcopy(deform_align) if clone_module else deform_align
        set_deformconv_export_mode(self.deform_align, export_mode)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        module_name: str,
        *,
        clone_module: bool = True,
        export_mode: bool = False,
    ) -> "BasicVsrppDeformAlignModule":
        generator = get_basicvsrpp_generator(model)
        return cls(
            generator.deform_align[module_name],
            clone_module=clone_module,
            export_mode=export_mode,
        )

    def set_export_mode(self, enabled: bool) -> None:
        set_deformconv_export_mode(self.deform_align, enabled)

    def forward(
        self,
        feature_pair: torch.Tensor,
        extra_feat: torch.Tensor,
        flow_n1: torch.Tensor,
        flow_n2: torch.Tensor,
    ) -> torch.Tensor:
        return self.deform_align(feature_pair, extra_feat, flow_n1, flow_n2)


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



from .vulkan_runtime_chains import (
    BasicVsrppBackward1AlignChain,
    BasicVsrppBackward1PropagateChain,
    BasicVsrppForward1AlignChain,
    BasicVsrppForward1PropagateChain,
    BasicVsrppBackward2AlignChain,
    BasicVsrppBackward2PropagateChain,
    BasicVsrppForward2AlignChain,
    BasicVsrppForward2PropagateChain,
    BasicVsrppBackward1PropagateWithFlowChain,
    BasicVsrppForward1PropagateWithFlowChain,
    BasicVsrppBackward2PropagateWithFlowChain,
    BasicVsrppForward2PropagateWithFlowChain,
)
