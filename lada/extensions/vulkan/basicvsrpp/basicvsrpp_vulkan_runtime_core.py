# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lada.models.basicvsrpp.deformconv import ModulatedDeformConv2d
from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import (
    BasicVSRPlusPlusNet,
    SPyNet,
    SecondOrderDeformableAlignment,
)
from lada.models.basicvsrpp.mmagic.flow_warp import flow_warp


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
