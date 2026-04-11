import argparse
import json
import time
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch

from lada import ModelFiles
from lada.compute_targets import get_compute_target, normalize_compute_target_id
from lada.restorationpipeline import load_models
from lada.restorationpipeline.basicvsrpp_vulkan_restorer import (
    NcnnVulkanBasicvsrppMosaicRestorer,
    _array_to_uint8_frame,
)
from lada.restorationpipeline.clip_units import (
    ClipDescriptor,
    build_clip_resize_plans,
    crop_descriptor_with_profile,
    materialize_clip_frames_with_profile,
    materialize_clip_masks_with_profile,
)
from lada.restorationpipeline.mosaic_detector import MosaicDetector
from lada.utils import video_utils
from lada.utils.os_utils import gpu_has_fp16_acceleration
from lada.utils.threading_utils import EOF_MARKER, ErrorMarker, PipelineQueue, STOP_MARKER


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the old CPU resize+pad + native restore path against the "
            "new native resize-preprocess Vulkan path on the first detected clip."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("resources/main.webm"),
        help="Video used to extract the first detected clip descriptor.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".helloagents/tmp/native_resize_path_profile.json"),
        help="JSON file used to store the benchmark result.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="vulkan:0",
        help="Compute target used for detection and restoration.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frames taken from the first emitted clip descriptor.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured repeats per benchmark case.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations per benchmark case.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Maximum seconds to wait for the first clip descriptor.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=gpu_has_fp16_acceleration(),
        help="Match the CLI default fp16 behavior when possible.",
    )
    parser.add_argument(
        "--max-clip-length",
        type=int,
        default=180,
        help="Match the production detector max clip length.",
    )
    return parser


def _slice_descriptor(descriptor: ClipDescriptor, frame_count: int) -> ClipDescriptor:
    if frame_count <= 0 or len(descriptor) <= frame_count:
        return descriptor
    return ClipDescriptor(
        file_path=descriptor.file_path,
        frame_start=descriptor.frame_start,
        size=descriptor.size,
        pad_mode=descriptor.pad_mode,
        id=f"{descriptor.id}:slice0",
        frames=list(descriptor.frames[:frame_count]),
        masks=list(descriptor.masks[:frame_count]),
        boxes=list(descriptor.boxes[:frame_count]),
        crop_boxes=list(descriptor.crop_boxes[:frame_count]),
        resize_reference_shape=descriptor.resize_reference_shape,
    )


def _extract_first_descriptor(
    *,
    detection_model,
    video_path: Path,
    pad_mode: str,
    timeout_s: float,
    max_clip_length: int,
    segment_length: int,
) -> ClipDescriptor:
    video_metadata = video_utils.get_video_meta_data(str(video_path))
    frame_detection_queue = PipelineQueue("bench_frame_detection_queue", maxsize=0)
    mosaic_clip_queue = PipelineQueue("bench_mosaic_clip_queue", maxsize=0)
    worker_errors: list[ErrorMarker] = []

    def _on_error(error: ErrorMarker) -> None:
        worker_errors.append(error)

    detector = MosaicDetector(
        detection_model,
        video_metadata,
        frame_detection_queue=frame_detection_queue,
        mosaic_clip_queue=mosaic_clip_queue,
        error_handler=_on_error,
        max_clip_length=max_clip_length,
        pad_mode=pad_mode,
        segment_length=segment_length,
    )

    started_at = time.perf_counter()
    detector.start(start_ns=0)
    try:
        while time.perf_counter() - started_at < timeout_s:
            if worker_errors:
                error = worker_errors[0]
                raise RuntimeError(f"Detector worker failed: {error}\n{error.stack_trace}")
            try:
                item = mosaic_clip_queue.get(timeout=0.5)
            except Empty:
                continue

            if isinstance(item, ClipDescriptor):
                return item
            if isinstance(item, ErrorMarker):
                raise RuntimeError(f"Detector worker failed: {item}\n{item.stack_trace}")
            if item is EOF_MARKER:
                raise RuntimeError("The detector reached EOF before emitting any mosaic clips.")
            if item is STOP_MARKER:
                raise RuntimeError("The detector stopped before emitting a clip descriptor.")
        raise TimeoutError(
            f"Timed out after {timeout_s:.1f}s while waiting for the first clip descriptor."
        )
    finally:
        detector.stop()


def _measure_case(
    *,
    name: str,
    fn,
    repeats: int,
    warmup: int,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    outputs: list[torch.Tensor] = []
    for _ in range(max(warmup, 0)):
        outputs = fn()

    wall_started_at = time.perf_counter()
    cpu_started_at = time.process_time()
    for _ in range(repeats):
        outputs = fn()
    cpu_total_s = time.process_time() - cpu_started_at
    wall_total_s = time.perf_counter() - wall_started_at

    result = {
        "name": name,
        "repeats": int(repeats),
        "warmup": int(max(warmup, 0)),
        "wall_s": float(wall_total_s),
        "cpu_s": float(cpu_total_s),
        "wall_per_iter_s": float(wall_total_s / repeats),
        "cpu_per_iter_s": float(cpu_total_s / repeats),
        "output_frames": len(outputs),
        "output_shape": [int(dim) for dim in outputs[0].shape] if outputs else None,
    }
    return result, outputs


def _tensor_difference_summary(
    baseline: list[torch.Tensor],
    candidate: list[torch.Tensor],
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise RuntimeError("Output frame count mismatch between benchmark cases.")

    per_frame_mae: list[float] = []
    per_frame_max_abs: list[int] = []
    for lhs, rhs in zip(baseline, candidate):
        diff = (lhs.to(dtype=torch.int16) - rhs.to(dtype=torch.int16)).abs()
        per_frame_mae.append(float(diff.float().mean().item()))
        per_frame_max_abs.append(int(diff.max().item()))

    return {
        "frames_compared": len(baseline),
        "mae_mean": float(sum(per_frame_mae) / len(per_frame_mae)),
        "mae_per_frame": per_frame_mae,
        "max_abs_overall": max(per_frame_max_abs),
        "max_abs_per_frame": per_frame_max_abs,
    }


def main() -> None:
    args = _build_argparser().parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    restoration_model_file = ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2")
    detection_model_file = ModelFiles.get_detection_model_by_name("v4-fast")
    if restoration_model_file is None or detection_model_file is None:
        raise RuntimeError("Required default model weights were not found in model_weights/.")

    compute_target_id = normalize_compute_target_id(args.device)
    compute_target = get_compute_target(compute_target_id, include_experimental=True)
    if compute_target is None or not compute_target.available:
        raise RuntimeError(f"Compute target '{compute_target_id}' is unavailable.")

    torch_device = (
        torch.device(compute_target.torch_device)
        if compute_target.torch_device is not None
        else torch.device("cpu")
    )

    loaded_models = load_models(
        compute_target_id,
        torch_device,
        "basicvsrpp-v1.2",
        restoration_model_file.path,
        None,
        detection_model_file.path,
        args.fp16,
        False,
    )
    detection_model = loaded_models.detection_model
    restoration_model = loaded_models.restoration_model
    preferred_pad_mode = loaded_models.preferred_pad_mode
    if not isinstance(restoration_model, NcnnVulkanBasicvsrppMosaicRestorer):
        raise RuntimeError("This benchmark requires the Vulkan BasicVSR++ restorer.")
    if restoration_model.native_clip_runner is None:
        raise RuntimeError("Native BasicVSR++ clip runner is unavailable.")
    if not restoration_model.native_clip_runner.supports_resized_bgr_u8_input:
        raise RuntimeError("The local ncnn runtime does not expose restore_bgr_u8_resized().")

    detector_segment_length = (
        min(
            int(args.max_clip_length),
            int(restoration_model.stream_restore_chunk_size) * 2,
        )
        if restoration_model.stream_restore_chunk_size
        else int(args.max_clip_length)
    )

    descriptor = _extract_first_descriptor(
        detection_model=detection_model,
        video_path=args.input,
        pad_mode=preferred_pad_mode,
        timeout_s=float(args.timeout),
        max_clip_length=int(args.max_clip_length),
        segment_length=detector_segment_length,
    )
    descriptor = _slice_descriptor(descriptor, int(args.frames))

    crop_profile: dict[str, float] = {}
    cropped_frames, cropped_masks, _, crop_shapes = crop_descriptor_with_profile(
        descriptor,
        crop_profile,
    )
    resize_plans = build_clip_resize_plans(descriptor, crop_shapes)
    mask_started_at = time.perf_counter()
    padded_masks, _ = materialize_clip_masks_with_profile(
        cropped_masks,
        resize_plans,
        size=descriptor.size,
        profile=None,
    )
    mask_resize_pad_s = time.perf_counter() - mask_started_at
    frame_started_at = time.perf_counter()
    materialized_frames, _ = materialize_clip_frames_with_profile(
        cropped_frames,
        resize_plans,
        size=descriptor.size,
        pad_mode=descriptor.pad_mode,
        profile=None,
    )
    frame_resize_pad_s = time.perf_counter() - frame_started_at

    def _run_materialized_old_path() -> list[torch.Tensor]:
        raw_outputs = restoration_model.native_clip_runner.restore_bgr_u8(materialized_frames)
        return [_array_to_uint8_frame(np.asarray(output)) for output in raw_outputs]

    def _run_native_resized_new_path() -> list[torch.Tensor]:
        raw_outputs = restoration_model.native_clip_runner.restore_bgr_u8_resized(
            cropped_frames,
            target_size=descriptor.size,
            resize_reference_shape=descriptor.resize_reference_shape,
            pad_mode=descriptor.pad_mode,
        )
        return [_array_to_uint8_frame(np.asarray(output)) for output in raw_outputs]

    old_case, old_outputs = _measure_case(
        name="materialized_bgr_u8_restore",
        fn=_run_materialized_old_path,
        repeats=int(args.repeats),
        warmup=int(args.warmup),
    )
    new_case, new_outputs = _measure_case(
        name="native_resized_bgr_u8_restore",
        fn=_run_native_resized_new_path,
        repeats=int(args.repeats),
        warmup=int(args.warmup),
    )

    result = {
        "input_path": str(args.input),
        "compute_target": compute_target_id,
        "fp16": bool(args.fp16),
        "descriptor": {
            "id": str(descriptor.id),
            "frame_start": int(descriptor.frame_start),
            "frame_end": int(descriptor.frame_end),
            "frame_count": len(descriptor),
            "size": int(descriptor.size),
            "pad_mode": descriptor.pad_mode,
            "resize_reference_shape": [
                int(descriptor.resize_reference_shape[0]),
                int(descriptor.resize_reference_shape[1]),
            ],
            "crop_shapes": [[int(dim) for dim in shape] for shape in crop_shapes],
        },
        "detector_config": {
            "max_clip_length": int(args.max_clip_length),
            "segment_length": int(detector_segment_length),
            "stream_restore_chunk_size": int(restoration_model.stream_restore_chunk_size or 0),
        },
        "preprocess_once": {
            "clip_crop_s": float(crop_profile.get("clip_crop_s", 0.0)),
            "clip_mask_resize_pad_s": float(mask_resize_pad_s),
            "frame_resize_pad_s": float(frame_resize_pad_s),
            "padded_mask_shape": [int(dim) for dim in padded_masks[0].shape] if padded_masks else None,
            "materialized_frame_shape": (
                [int(dim) for dim in materialized_frames[0].shape] if materialized_frames else None
            ),
        },
        "cases": [old_case, new_case],
        "new_over_old_wall_ratio": (
            float(new_case["wall_per_iter_s"] / old_case["wall_per_iter_s"])
            if float(old_case["wall_per_iter_s"]) > 0.0
            else None
        ),
        "output_diff": _tensor_difference_summary(old_outputs, new_outputs),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if hasattr(restoration_model, "release_cached_memory"):
        restoration_model.release_cached_memory()
    if hasattr(detection_model, "release_cached_memory"):
        detection_model.release_cached_memory()


if __name__ == "__main__":
    main()
