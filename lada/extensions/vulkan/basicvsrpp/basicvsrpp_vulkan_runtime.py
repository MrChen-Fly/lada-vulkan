# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from .basicvsrpp_vulkan_param_patch import patch_ncnn_param_for_vulkan_runtime
from .basicvsrpp_vulkan_runtime_core import (
    BasicVsrppDeformAlignModule,
    BasicVsrppFeatExtractModule,
    BasicVsrppFlowWarpModule,
    BasicVsrppQuarterDownsampleModule,
    BasicVsrppSpynetComputeFlowModule,
    BasicVsrppSpynetFlowModule,
    get_basicvsrpp_generator,
    prepare_deform_align_inputs,
    set_deformconv_export_mode,
)
from .basicvsrpp_vulkan_runtime_heads import (
    BasicVsrppBackboneModule,
    BasicVsrppOutputFrameModule,
    BasicVsrppOutputHeadModule,
    BasicVsrppReconstructionConcatModule,
    BasicVsrppReconstructionModule,
    prepare_backward1_backbone_input,
    prepare_backward2_backbone_input,
    prepare_forward1_backbone_input,
    prepare_forward2_backbone_input,
    prepare_propagation_backbone_input,
    prepare_reconstruction_input,
)
from .basicvsrpp_vulkan_runtime_chains import (
    BasicVsrppBackward1AlignChain,
    BasicVsrppBackward1PropagateChain,
    BasicVsrppBackward1PropagateWithFlowChain,
    BasicVsrppBackward2AlignChain,
    BasicVsrppBackward2PropagateChain,
    BasicVsrppBackward2PropagateWithFlowChain,
    BasicVsrppForward1AlignChain,
    BasicVsrppForward1PropagateChain,
    BasicVsrppForward1PropagateWithFlowChain,
    BasicVsrppForward2AlignChain,
    BasicVsrppForward2PropagateChain,
    BasicVsrppForward2PropagateWithFlowChain,
)
