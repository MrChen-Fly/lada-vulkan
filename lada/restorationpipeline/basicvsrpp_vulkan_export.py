from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import torch

from lada.compute_targets import UnsupportedComputeTargetError
from lada.models.basicvsrpp.vulkan_runtime import (
    BasicVsrppBackboneModule,
    BasicVsrppBackward1PropagateWithFlowChain,
    BasicVsrppBackward2PropagateWithFlowChain,
    BasicVsrppFeatExtractModule,
    BasicVsrppForward1PropagateWithFlowChain,
    BasicVsrppForward2PropagateWithFlowChain,
    BasicVsrppOutputFrameModule,
    BasicVsrppQuarterDownsampleModule,
    BasicVsrppSpynetFlowModule,
    get_basicvsrpp_generator,
    patch_ncnn_param_for_vulkan_runtime,
)
from lada.models.basicvsrpp.vulkan_runtime_clip import BasicVsrppFusedRestoreClipModule
from .basicvsrpp_vulkan_common import (
    NcnnArtifacts,
    _EXPORT_FRAME_SIZE,
    _FEATURE_SIZE,
    _ModuleExportSpec,
    _as_pnnx_path,
    _import_pnnx,
    _validate_ncnn_artifacts,
)


def _enable_basicvsrpp_export_mode(model: torch.nn.Module) -> None:
    from lada.models.basicvsrpp.deformconv import ModulatedDeformConv2d

    for module in model.modules():
        if isinstance(module, ModulatedDeformConv2d):
            module.export_mode = True


def _build_window_indices(center_index: int, total_frames: int, frame_count: int) -> list[int]:
    radius = frame_count // 2
    return [
        min(max(center_index + offset, 0), total_frames - 1)
        for offset in range(-radius, radius + 1)
    ]


def _format_pnnx_inputshape(example_inputs: tuple[torch.Tensor, ...]) -> str:
    parts = []
    for tensor in example_inputs:
        shape = ",".join(str(size) for size in tensor.shape)
        parts.append(f"[{shape}]f32")
    return "inputshape=" + ",".join(parts)


def _export_ncnn_module(
    spec: _ModuleExportSpec,
    artifacts: NcnnArtifacts,
) -> None:
    pnnx = _import_pnnx()
    spec.module.eval()

    with tempfile.TemporaryDirectory(prefix=f"lada-{spec.name}-pnnx-") as temp_dir:
        traced_path = Path(temp_dir) / f"{spec.name}.pt"
        traced = torch.jit.trace(spec.module, spec.example_inputs, check_trace=False)
        traced.save(str(traced_path))

        result = subprocess.run(
            [
                pnnx.EXEC_PATH,
                _as_pnnx_path(traced_path),
                _format_pnnx_inputshape(spec.example_inputs),
                "device=cpu",
                "fp16=0",
                f"ncnnparam={_as_pnnx_path(artifacts.param_path)}",
                f"ncnnbin={_as_pnnx_path(artifacts.bin_path)}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise UnsupportedComputeTargetError(
                f"Failed to export Vulkan module '{spec.name}' to ncnn.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    patch_ncnn_param_for_vulkan_runtime(artifacts.param_path)
    _validate_ncnn_artifacts(artifacts)


def _build_modular_export_specs(
    model: torch.nn.Module,
) -> list[_ModuleExportSpec]:
    generator = get_basicvsrpp_generator(model)
    current_lq = torch.zeros((1, 3, _EXPORT_FRAME_SIZE, _EXPORT_FRAME_SIZE), dtype=torch.float32)
    quarter_lq = torch.zeros((1, 3, _FEATURE_SIZE, _FEATURE_SIZE), dtype=torch.float32)
    feature = torch.zeros((1, 64, _FEATURE_SIZE, _FEATURE_SIZE), dtype=torch.float32)
    flow = torch.zeros((1, 2, _FEATURE_SIZE, _FEATURE_SIZE), dtype=torch.float32)
    restore_clip_inputs = (
        torch.zeros_like(current_lq),
        torch.zeros_like(current_lq),
        torch.zeros_like(current_lq),
        torch.zeros_like(current_lq),
        torch.zeros_like(current_lq),
        torch.zeros_like(feature),
        torch.zeros_like(feature),
        torch.zeros_like(feature),
        torch.zeros_like(feature),
        torch.zeros_like(feature),
        torch.zeros_like(flow),
        torch.zeros_like(flow),
        torch.zeros_like(flow),
        torch.zeros_like(flow),
        torch.zeros_like(flow),
        torch.zeros_like(flow),
        torch.zeros_like(flow),
        torch.zeros_like(flow),
    )
    return [
        _ModuleExportSpec(
            "quarter_downsample",
            BasicVsrppQuarterDownsampleModule(),
            (current_lq,),
        ),
        _ModuleExportSpec(
            "feat_extract",
            BasicVsrppFeatExtractModule.from_model(generator),
            (current_lq,),
        ),
        _ModuleExportSpec(
            "spynet",
            BasicVsrppSpynetFlowModule.from_model(generator),
            (quarter_lq, quarter_lq),
        ),
        _ModuleExportSpec(
            "backward_1_backbone",
            BasicVsrppBackboneModule.from_model(generator, "backward_1"),
            (feature, feature),
        ),
        _ModuleExportSpec(
            "forward_1_backbone",
            BasicVsrppBackboneModule.from_model(generator, "forward_1"),
            (feature, feature, feature),
        ),
        _ModuleExportSpec(
            "backward_2_backbone",
            BasicVsrppBackboneModule.from_model(generator, "backward_2"),
            (feature, feature, feature, feature),
        ),
        _ModuleExportSpec(
            "forward_2_backbone",
            BasicVsrppBackboneModule.from_model(generator, "forward_2"),
            (feature, feature, feature, feature, feature),
        ),
        _ModuleExportSpec(
            "backward_1_step",
            BasicVsrppBackward1PropagateWithFlowChain.from_model(
                generator,
                export_mode=False,
            ),
            (feature, feature, feature, flow, flow),
        ),
        _ModuleExportSpec(
            "forward_1_step",
            BasicVsrppForward1PropagateWithFlowChain.from_model(
                generator,
                export_mode=False,
            ),
            (feature, feature, feature, feature, flow, flow),
        ),
        _ModuleExportSpec(
            "backward_2_step",
            BasicVsrppBackward2PropagateWithFlowChain.from_model(
                generator,
                export_mode=False,
            ),
            (feature, feature, feature, feature, feature, flow, flow),
        ),
        _ModuleExportSpec(
            "forward_2_step",
            BasicVsrppForward2PropagateWithFlowChain.from_model(
                generator,
                export_mode=False,
            ),
            (feature, feature, feature, feature, feature, feature, flow, flow),
        ),
        _ModuleExportSpec(
            "output_frame",
            BasicVsrppOutputFrameModule.from_model(generator),
            (current_lq, feature, feature, feature, feature, feature),
        ),
        _ModuleExportSpec(
            "restore_clip",
            BasicVsrppFusedRestoreClipModule.from_model(
                generator,
                export_mode=False,
            ),
            restore_clip_inputs,
        ),
    ]
