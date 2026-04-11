from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from lada.models.yolo.detection_backends import build_mosaic_detection_model
from lada.extensions.vulkan.basicvsrpp_restore_paths import (
    _resize_cropped_frames_for_runtime,
)
from lada.parity_restoration_core import build_restoration_core_probes
from lada.parity_report import build_probe, extract_detection_arrays, quantize_image_output
from lada.restorationpipeline.clip_units import Clip, ClipDescriptor, Scene
from lada.restorationpipeline.frame_restorer_blend import restore_frame
from lada.restorationpipeline.frame_restorer_clip_ops import (
    prepare_descriptor_for_native_restore,
)
from lada.utils import image_utils, mask_utils


class _NoopProfiler:
    def measure(self, _bucket: str | None):
        return nullcontext()

    def add_count(self, _bucket: str) -> None:
        return None

    def add_duration(self, _bucket: str, _duration: float) -> None:
        return None


@dataclass
class _ParityFrameRestorerAdapter:
    device: torch.device
    mosaic_restoration_model: Any
    profiler: _NoopProfiler


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    return np.ascontiguousarray(value).copy()


def _clone_frame_tensor(frame: Any) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        return frame.detach().clone().to(device="cpu")
    return torch.from_numpy(np.ascontiguousarray(frame).copy())


def _coerce_scene_box(
    mask: np.ndarray,
    raw_box: np.ndarray,
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    if np.count_nonzero(mask) > 0:
        return mask_utils.get_box(mask)

    height, width = frame_shape[:2]
    candidates = [
        tuple(int(value) for value in raw_box[:4]),
        (
            int(raw_box[1]),
            int(raw_box[0]),
            int(raw_box[3]),
            int(raw_box[2]),
        ),
    ]
    for top, left, bottom, right in candidates:
        if 0 <= top <= bottom < height and 0 <= left <= right < width:
            return top, left, bottom, right
    raise RuntimeError(
        f"Could not resolve a valid scene box from detection box {raw_box[:4].tolist()} "
        f"for frame shape {frame_shape[:2]}."
    )


def _build_probe_scene(
    frames: list[np.ndarray],
    *,
    detection_model_path: str,
    reference_device_id: str,
    fp16: bool,
) -> Scene | None:
    reference_model = build_mosaic_detection_model(
        detection_model_path,
        reference_device_id,
        conf=0.15,
        fp16=bool(fp16 and reference_device_id != "cpu"),
    )
    try:
        preprocessed = reference_model.preprocess(frames)
        results = reference_model.inference_and_postprocess(preprocessed, frames)
        scene: Scene | None = None
        for frame_index, result in enumerate(results):
            boxes, masks = extract_detection_arrays(result)
            if len(boxes) == 0:
                if scene is not None:
                    break
                continue
            mask = np.ascontiguousarray(masks[0])
            box = _coerce_scene_box(mask, boxes[0], frames[frame_index].shape)
            if scene is None:
                scene = Scene("parity-probe", None)
            scene.add_frame(frame_index, np.ascontiguousarray(frames[frame_index]).copy(), mask, box)
        return scene
    finally:
        reference_model.release_cached_memory()


def _build_materialized_single_frame_clip(
    clip: Clip,
    restored_frame: Any,
    frame_index: int,
) -> Clip:
    return Clip.from_processed_data(
        file_path=clip.file_path,
        frame_start=clip.frame_start + frame_index,
        size=clip.size,
        pad_mode=clip.pad_mode,
        id=f"{clip.id}:{frame_index}",
        frames=[_clone_value(restored_frame)],
        masks=[_clone_value(clip.masks[frame_index])],
        boxes=[clip.boxes[frame_index]],
        crop_shapes=[clip.crop_shapes[frame_index]],
        pad_after_resizes=[clip.pad_after_resizes[frame_index]],
    )


def _prepare_blend_probe_inputs(
    clip_img: Any,
    clip_mask: torch.Tensor,
    orig_crop_shape: tuple[int, ...],
    pad_after_resize: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
    clip_mask = image_utils.unpad_image(clip_mask, pad_after_resize)
    clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
    clip_mask = image_utils.resize(
        clip_mask,
        orig_crop_shape[:2],
        interpolation=cv2.INTER_NEAREST,
    )
    blend_mask = mask_utils.create_blend_mask(clip_mask.float()).to(
        device=clip_img.device,
        dtype=torch.float32,
    )
    return clip_img, clip_mask, blend_mask


def build_restoration_pipeline_probes(
    frames: list[np.ndarray],
    *,
    detection_model_path: str,
    reference_device_id: str,
    reference_device: torch.device,
    reference_modules: dict[str, torch.nn.Module],
    reference_restorer: Any,
    candidate_restorer: Any,
    fp16: bool,
) -> dict[str, Any]:
    scene = _build_probe_scene(
        frames,
        detection_model_path=detection_model_path,
        reference_device_id=reference_device_id,
        fp16=fp16,
    )
    if scene is None or len(scene.frames) == 0:
        return {
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
            "errors": [{"probe": "restoration_pipeline", "error": "No contiguous detected scene available for outer-pipeline probes."}],
        }

    descriptor = ClipDescriptor.from_scene(scene, size=256, pad_mode="reflect", clip_id="parity")
    probe_restorer = _ParityFrameRestorerAdapter(
        device=torch.device("cpu"),
        mosaic_restoration_model=reference_restorer,
        profiler=_NoopProfiler(),
    )
    materialized_clip = Clip.from_descriptor(descriptor, resize_mode="torch_bilinear")
    cropped_frames, padded_masks, boxes, crop_shapes, pad_after_resizes = (
        prepare_descriptor_for_native_restore(probe_restorer, descriptor)
    )
    resized_frames = _resize_cropped_frames_for_runtime(
        cropped_frames,
        size=descriptor.size,
        resize_reference_shape=descriptor.resize_reference_shape,
        pad_mode=descriptor.pad_mode,
    )
    core_result = build_restoration_core_probes(
        materialized_clip.frames,
        reference_device=reference_device,
        reference_modules=reference_modules,
        reference_restorer=reference_restorer,
        candidate_restorer=candidate_restorer,
        probe_prefix="descriptor/core",
    )
    probes: list[dict[str, Any]] = list(core_result["probes"])
    errors = list(core_result["errors"])
    for frame_index in range(len(materialized_clip.frames)):
        probes.append(
            build_probe(
                f"descriptor/materialize/frame_{frame_index}",
                materialized_clip.frames[frame_index],
                resized_frames[frame_index],
            )
        )
        probes.append(
            build_probe(
                f"descriptor/masks/frame_{frame_index}",
                materialized_clip.masks[frame_index],
                padded_masks[frame_index],
            )
        )
        probes.append(
            build_probe(
                f"descriptor/boxes/frame_{frame_index}",
                np.asarray(materialized_clip.boxes[frame_index], dtype=np.int32),
                np.asarray(boxes[frame_index], dtype=np.int32),
            )
        )
        probes.append(
            build_probe(
                f"descriptor/pad_after_resize/frame_{frame_index}",
                np.asarray(materialized_clip.pad_after_resizes[frame_index], dtype=np.int32),
                np.asarray(pad_after_resizes[frame_index], dtype=np.int32),
            )
        )
        probes.append(
            build_probe(
                f"descriptor/crop_shape/frame_{frame_index}",
                np.asarray(materialized_clip.crop_shapes[frame_index], dtype=np.int32),
                np.asarray(crop_shapes[frame_index], dtype=np.int32),
            )
        )

    reference_patch_frames = reference_restorer.restore([
        _clone_frame_tensor(frame) for frame in materialized_clip.frames
    ])
    candidate_patch_frames = candidate_restorer.restore(
        [_clone_value(frame) for frame in materialized_clip.frames]
    )
    blend_reference_restorer = _ParityFrameRestorerAdapter(
        device=torch.device("cpu"),
        mosaic_restoration_model=reference_restorer,
        profiler=_NoopProfiler(),
    )
    blend_candidate_restorer = _ParityFrameRestorerAdapter(
        device=torch.device("cpu"),
        mosaic_restoration_model=candidate_restorer,
        profiler=_NoopProfiler(),
    )
    for frame_index in range(len(materialized_clip.frames)):
        probes.append(
            build_probe(
                f"descriptor/restored_patch/frame_{frame_index}",
                reference_patch_frames[frame_index],
                candidate_patch_frames[frame_index],
            )
        )
        probes.append(
            build_probe(
                f"descriptor/restored_patch_quantized/frame_{frame_index}",
                quantize_image_output(reference_patch_frames[frame_index]),
                quantize_image_output(candidate_patch_frames[frame_index]),
            )
        )
        reference_clip = _build_materialized_single_frame_clip(
            materialized_clip,
            reference_patch_frames[frame_index],
            frame_index,
        )
        candidate_clip = _build_materialized_single_frame_clip(
            materialized_clip,
            candidate_patch_frames[frame_index],
            frame_index,
        )
        reference_clip_img, reference_clip_mask, _ = _prepare_blend_probe_inputs(
            reference_clip.frames[0],
            reference_clip.masks[0],
            reference_clip.crop_shapes[0],
            reference_clip.pad_after_resizes[0],
        )
        candidate_clip_img, candidate_clip_mask, _ = _prepare_blend_probe_inputs(
            candidate_clip.frames[0],
            candidate_clip.masks[0],
            candidate_clip.crop_shapes[0],
            candidate_clip.pad_after_resizes[0],
        )
        probes.append(
            build_probe(
                f"blend/pre_patch/frame_{frame_index}",
                reference_clip_img,
                candidate_clip_img,
            )
        )
        probes.append(
            build_probe(
                f"blend/pre_mask/frame_{frame_index}",
                reference_clip_mask,
                candidate_clip_mask,
            )
        )
        probes.append(
            build_probe(
                f"blend/blend_mask/frame_{frame_index}",
                mask_utils.create_blend_mask(reference_clip_mask.float()),
                mask_utils.create_blend_mask(candidate_clip_mask.float()),
            )
        )

        reference_frame = _clone_frame_tensor(frames[descriptor.frame_start + frame_index])
        candidate_frame = _clone_frame_tensor(frames[descriptor.frame_start + frame_index])
        restore_frame(
            blend_reference_restorer,
            reference_frame,
            descriptor.frame_start + frame_index,
            [reference_clip],
        )
        restore_frame(
            blend_candidate_restorer,
            candidate_frame,
            descriptor.frame_start + frame_index,
            [candidate_clip],
        )
        probes.append(
            build_probe(
                f"blend/final_frame/frame_{frame_index}",
                reference_frame,
                candidate_frame,
            )
        )
        probes.append(
            build_probe(
                f"blend/final_frame_quantized/frame_{frame_index}",
                quantize_image_output(reference_frame),
                quantize_image_output(candidate_frame),
            )
        )

    return {
        "implemented": [
            "descriptor/materialize",
            "descriptor/core",
            "descriptor/restored_patch",
            "descriptor/restored_patch_quantized",
            "blend/pre_patch",
            "blend/blend_mask",
            "blend/final_frame",
            "blend/final_frame_quantized",
        ],
        "not_implemented": [],
        "probes": probes,
        "errors": errors,
    }
