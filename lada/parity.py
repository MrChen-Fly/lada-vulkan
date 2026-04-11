from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lada.compute_targets import (
    describe_compute_target_issue,
    get_compute_target,
    normalize_compute_target_id,
    resolve_torch_device,
)
from lada.models.basicvsrpp.inference import load_model
from lada.models.yolo.detection_backends import build_mosaic_detection_model
from lada.parity_restoration_core import (
    build_reference_basicvsrpp_modules,
    build_restoration_core_probes,
)
from lada.parity_restoration_pipeline import build_restoration_pipeline_probes
from lada.extensions.vulkan.basicvsrpp_restorer import NcnnVulkanBasicvsrppMosaicRestorer
from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from lada.parity_report import build_probe, extract_detection_arrays
from lada.utils import video_utils


def load_video_clip(input_path: str | Path, *, start_frame: int, frame_count: int) -> list[np.ndarray]:
    """Load a fixed-size clip from one input video as contiguous uint8 BGR frames."""
    if frame_count < 2:
        raise RuntimeError(f"frame_count must be >= 2 for temporal parity, got {frame_count}.")
    frames = video_utils.read_video_frames(
        str(input_path),
        float32=False,
        start_idx=start_frame,
        end_idx=start_frame + frame_count,
    )
    if len(frames) < frame_count:
        raise RuntimeError(
            f"Input '{input_path}' only produced {len(frames)} frames from [{start_frame}, "
            f"{start_frame + frame_count}), expected {frame_count}."
        )
    return [np.ascontiguousarray(frame) for frame in frames]


def resolve_reference_torch_device(reference_device_id: str) -> tuple[str, torch.device]:
    """Resolve and validate the Torch reference device used for semantic parity."""
    normalized_target_id = normalize_compute_target_id(reference_device_id)
    issue = describe_compute_target_issue(normalized_target_id)
    if issue:
        raise RuntimeError(issue)
    target = get_compute_target(normalized_target_id, include_experimental=True)
    if target is None or target.runtime != "torch":
        raise RuntimeError(
            f"Reference device '{normalized_target_id}' is not a Torch-backed compute target."
        )
    return normalized_target_id, resolve_torch_device(normalized_target_id)


def _run_detection_parity(
    frame: np.ndarray,
    *,
    detection_model_path: str,
    reference_device_id: str,
    fp16: bool,
) -> dict[str, Any]:
    reference_device = resolve_torch_device(reference_device_id)
    reference_fp16 = bool(fp16 and reference_device.type != "cpu")
    reference_model = build_mosaic_detection_model(
        detection_model_path,
        reference_device_id,
        conf=0.15,
        fp16=reference_fp16,
    )
    candidate_model = build_mosaic_detection_model(
        detection_model_path,
        "vulkan:0",
        conf=0.15,
        fp16=fp16,
    )
    try:
        reference_preprocessed = reference_model.preprocess([frame])
        reference_input = reference_model.prepare_input(reference_preprocessed)
        candidate_preprocessed = candidate_model.preprocess([frame])
        reference_raw = reference_model.inference(reference_input)
        candidate_raw = candidate_model._run_raw_outputs_to_cpu(candidate_preprocessed[0])
        # Keep a stable snapshot of the first Torch inference outputs. Re-running
        # inference_and_postprocess() can reuse or overwrite backend buffers and
        # corrupt raw-output parity probes.
        reference_pred = reference_raw[0][0][0].detach().clone()
        reference_proto = reference_raw[0][1][0].detach().clone()
        reference_results = reference_model.postprocess(reference_raw, reference_input, [frame])
        candidate_results = candidate_model.inference_and_postprocess(candidate_preprocessed, [frame])
        reference_boxes, reference_masks = extract_detection_arrays(reference_results[0])
        candidate_boxes, candidate_masks = extract_detection_arrays(candidate_results[0])
        return {
            "implemented": [
                "detector/preprocess",
                "detector/raw_outputs/pred",
                "detector/raw_outputs/proto",
                "detector/postprocess/boxes",
                "detector/postprocess/masks",
            ],
            "not_implemented": [],
            "probes": [
                build_probe("detector/preprocess", reference_input[0], candidate_preprocessed[0]),
                build_probe("detector/raw_outputs/pred", reference_pred, candidate_raw[0]),
                build_probe("detector/raw_outputs/proto", reference_proto, candidate_raw[1]),
                build_probe("detector/postprocess/boxes", reference_boxes, candidate_boxes),
                build_probe("detector/postprocess/masks", reference_masks, candidate_masks),
            ],
        }
    finally:
        reference_model.release_cached_memory()
        candidate_model.release_cached_memory()
        del reference_model
        del candidate_model
        gc.collect()


def _run_restoration_parity(
    frames: list[np.ndarray],
    *,
    detection_model_path: str,
    reference_device_id: str,
    restoration_model_path: str,
    restoration_config_path: str | None,
    reference_device: torch.device,
    fp16: bool,
    artifacts_dir: str | None,
) -> dict[str, Any]:
    reference_model = load_model(restoration_config_path, restoration_model_path, reference_device, fp16=False)
    reference_modules = build_reference_basicvsrpp_modules(reference_model, reference_device)
    reference_restorer = BasicvsrppMosaicRestorer(reference_model, reference_device, fp16=False)
    candidate_restorer = NcnnVulkanBasicvsrppMosaicRestorer(
        restoration_model_path,
        config_path=restoration_config_path,
        fp16=fp16,
        artifacts_dir=artifacts_dir,
    )
    try:
        core_result = build_restoration_core_probes(
            frames,
            reference_device=reference_device,
            reference_modules=reference_modules,
            reference_restorer=reference_restorer,
            candidate_restorer=candidate_restorer,
        )
        probes = list(core_result["probes"])
        errors = list(core_result["errors"])
        pipeline_result = {
            "implemented": [],
            "not_implemented": [
                "descriptor/materialize",
                "descriptor/core",
                "descriptor/restored_patch",
                "descriptor/restored_patch_quantized",
                "blend/pre_patch",
                "blend/blend_mask",
                "blend/final_frame",
                "blend/final_frame_quantized",
            ],
            "probes": [],
            "errors": [],
        }
        try:
            pipeline_result = build_restoration_pipeline_probes(
                frames,
                detection_model_path=detection_model_path,
                reference_device_id=reference_device_id,
                reference_device=reference_device,
                reference_modules=reference_modules,
                reference_restorer=reference_restorer,
                candidate_restorer=candidate_restorer,
                fp16=fp16,
            )
        except RuntimeError as exc:
            pipeline_result["errors"].append(
                {
                    "probe": "restoration_pipeline",
                    "error": str(exc),
                }
            )
        probes.extend(pipeline_result["probes"])
        errors.extend(pipeline_result["errors"])
        return {
            "probe_strategy": "isolated_module_parity_with_reference_inputs",
            "implemented": list(
                dict.fromkeys(
                    [
                        "quarter_downsample",
                        "feat_extract",
                        "spynet",
                        "branch_backbone",
                        "branch_step",
                        "output_frame",
                        "restore",
                        *pipeline_result["implemented"],
                    ]
                )
            ),
            "not_implemented": pipeline_result["not_implemented"],
            "probes": probes,
            "errors": errors,
        }
    finally:
        reference_restorer.release_cached_memory()
        candidate_restorer.release_cached_memory()
        reference_modules.clear()
        del reference_model
        del reference_modules
        del reference_restorer
        del candidate_restorer
        gc.collect()


def run_device_parity(
    input_path: str | Path,
    *,
    reference_device_id: str,
    detection_model_path: str,
    restoration_model_path: str,
    restoration_config_path: str | None,
    start_frame: int,
    frame_count: int,
    fp16: bool,
    artifacts_dir: str | None = None,
) -> dict[str, Any]:
    """Run one semantic parity pass between one Torch device and the Vulkan runtime."""
    normalized_reference_device_id, reference_device = resolve_reference_torch_device(reference_device_id)
    frames = load_video_clip(input_path, start_frame=start_frame, frame_count=frame_count)
    return {
        "meta": {
            "input_path": str(Path(input_path).resolve()),
            "reference_device": normalized_reference_device_id,
            "candidate_device": "vulkan:0",
            "start_frame": int(start_frame),
            "frame_count": int(frame_count),
            "fp16": bool(fp16),
        },
        "detection": _run_detection_parity(
            frames[0],
            detection_model_path=detection_model_path,
            reference_device_id=normalized_reference_device_id,
            fp16=fp16,
        ),
        "restoration": _run_restoration_parity(
            frames,
            detection_model_path=detection_model_path,
            reference_device_id=normalized_reference_device_id,
            restoration_model_path=restoration_model_path,
            restoration_config_path=restoration_config_path,
            reference_device=reference_device,
            fp16=fp16,
            artifacts_dir=artifacts_dir,
        ),
    }
