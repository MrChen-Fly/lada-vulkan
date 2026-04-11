# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch

from lada import ModelFiles
from lada.cli import utils
from lada.cli.pipeline_runner import (
    VideoProcessingRequest,
    build_output_metadata,
    compute_target_to_report,
    process_video_file,
    write_timing_report,
)
from lada.compute_targets import (
    default_fp16_enabled_for_compute_target,
    describe_compute_target_issue,
    get_compute_target,
    normalize_compute_target_id,
    resolve_torch_device,
)
from lada.restorationpipeline import load_models
from lada.utils import video_utils
from lada.utils.video_utils import get_default_preset_name

_PIPELINE_REVISION_ENV = "LADA_OUTPUT_PIPELINE_REVISION"


@dataclass(frozen=True)
class DeviceRun:
    """Describe one device-specific render pass used by the regression tool."""

    compute_target_id: str
    output_suffix: str
    fp16: bool


@dataclass(frozen=True)
class RenderArtifacts:
    """Describe the media and JSON artifacts produced for one input/device pair."""

    input_path: Path
    output_path: Path
    report_path: Path


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser for the manual device regression tool."""

    parser = argparse.ArgumentParser(
        description=(
            "Render the same input video(s) on two devices and compare the "
            "resulting outputs with ffprobe/ffmpeg/PyAV metrics."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input video file or directory containing video files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory used to store rendered outputs and comparison JSON.",
    )
    parser.add_argument(
        "--baseline-device",
        default="cpu",
        help="Baseline compute target. Usually cpu.",
    )
    parser.add_argument(
        "--candidate-device",
        default="vulkan:0",
        help="Candidate compute target. Usually vulkan:0.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override fp16 for both compared devices. Default: auto per device.",
    )
    parser.add_argument(
        "--reuse-existing-outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse an existing rendered output instead of regenerating it.",
    )
    parser.add_argument(
        "--temporary-directory",
        default=tempfile.gettempdir(),
        help="Directory for intermediate encoded files.",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="Optional master comparison report file name.",
    )
    parser.add_argument(
        "--pipeline-revision",
        default="manual-device-regression",
        help="Revision marker embedded into output metadata.",
    )
    parser.add_argument(
        "--encoding-preset",
        default=get_default_preset_name(),
        help="Encoding preset used when --encoder is not set.",
    )
    parser.add_argument("--encoder", help="Optional explicit encoder name.")
    parser.add_argument(
        "--encoder-options",
        help="Optional space-separated encoder options for --encoder.",
    )
    parser.add_argument(
        "--mp4-fast-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fast-start mp4 flags on generated outputs.",
    )
    parser.add_argument(
        "--mosaic-restoration-model",
        default="basicvsrpp-v1.2",
        help="Restoration model name or path.",
    )
    parser.add_argument(
        "--mosaic-restoration-config-path",
        default=None,
        help="Optional restoration config path for custom weights.",
    )
    parser.add_argument(
        "--mosaic-detection-model",
        default="v4-fast",
        help="Detection model name or path.",
    )
    parser.add_argument(
        "--detect-face-mosaics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep face mosaics instead of filtering them out.",
    )
    parser.add_argument(
        "--max-clip-length",
        type=int,
        default=180,
        help="Maximum number of frames restored in one clip.",
    )
    return parser


def resolve_input_files(input_arg: str) -> tuple[list[Path], Path | None]:
    """Resolve the input path into concrete files plus an optional root directory."""

    input_path = Path(input_arg).resolve()
    if input_path.is_file():
        return [input_path], None
    if input_path.is_dir():
        files = [Path(path).resolve() for path in utils.list_video_files(str(input_path))]
        if not files:
            raise RuntimeError(f"No video files found in '{input_path}'.")
        return files, input_path
    raise RuntimeError(f"Invalid input path '{input_arg}'.")


def slugify_compute_target(compute_target_id: str) -> str:
    """Convert one compute target id into a filename-safe suffix."""

    return re.sub(r"[^a-zA-Z0-9]+", "-", compute_target_id).strip("-")


def build_render_artifacts(
    *,
    input_path: Path,
    input_root: Path | None,
    output_dir: Path,
    output_suffix: str,
) -> RenderArtifacts:
    """Build output media/report paths for one input/device pair."""

    relative_parent = Path()
    if input_root is not None:
        relative_parent = input_path.parent.relative_to(input_root)
    output_stem = f"{input_path.stem}-{output_suffix}"
    output_path = output_dir / relative_parent / f"{output_stem}.mp4"
    report_path = output_dir / relative_parent / f"{output_stem}.json"
    return RenderArtifacts(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
    )


def resolve_encoder_settings(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve the encoder and options using the same policy as the CLI."""

    if args.encoder:
        return args.encoder, args.encoder_options or ""

    for preset in video_utils.get_encoding_presets():
        if preset.name == args.encoding_preset:
            return preset.encoder_name, preset.encoder_options
    raise RuntimeError(f"Invalid encoding preset '{args.encoding_preset}'.")


def resolve_model_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    """Resolve detection/restoration model names into concrete filesystem paths."""

    detection_model = ModelFiles.get_detection_model_by_name(args.mosaic_detection_model)
    if detection_model is not None:
        mosaic_detection_model_path = detection_model.path
    elif os.path.isfile(args.mosaic_detection_model):
        mosaic_detection_model_path = args.mosaic_detection_model
    else:
        raise RuntimeError("Invalid mosaic detection model.")

    restoration_model = ModelFiles.get_restoration_model_by_name(args.mosaic_restoration_model)
    if restoration_model is not None:
        mosaic_restoration_model_name = args.mosaic_restoration_model
        mosaic_restoration_model_path = restoration_model.path
    elif os.path.isfile(args.mosaic_restoration_model):
        mosaic_restoration_model_name = "basicvsrpp"
        mosaic_restoration_model_path = args.mosaic_restoration_model
    else:
        raise RuntimeError("Invalid mosaic restoration model.")

    return (
        mosaic_restoration_model_name,
        mosaic_restoration_model_path,
        mosaic_detection_model_path,
    )


def resolve_device_run(compute_target_id: str, requested_fp16: bool | None) -> DeviceRun:
    """Resolve one device id into the device run configuration used by the script."""

    normalized_target_id = normalize_compute_target_id(compute_target_id)
    compute_target_issue = describe_compute_target_issue(normalized_target_id)
    if compute_target_issue:
        raise RuntimeError(compute_target_issue)
    fp16 = (
        default_fp16_enabled_for_compute_target(normalized_target_id)
        if requested_fp16 is None
        else requested_fp16
    )
    return DeviceRun(
        compute_target_id=normalized_target_id,
        output_suffix=slugify_compute_target(normalized_target_id),
        fp16=fp16,
    )


def load_models_for_device(
    *,
    args: argparse.Namespace,
    device_run: DeviceRun,
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_detection_model_path: str,
) -> tuple[Any, Any]:
    """Load the detection/restoration models needed for one device pass."""

    compute_target = get_compute_target(device_run.compute_target_id, include_experimental=True)
    if compute_target is None:
        raise RuntimeError(f"Unknown compute target '{device_run.compute_target_id}'.")
    torch_device = (
        resolve_torch_device(device_run.compute_target_id)
        if compute_target.torch_device is not None
        else None
    )
    loaded_models = load_models(
        device_run.compute_target_id,
        torch_device,
        mosaic_restoration_model_name,
        mosaic_restoration_model_path,
        args.mosaic_restoration_config_path,
        mosaic_detection_model_path,
        device_run.fp16,
        args.detect_face_mosaics,
    )
    return compute_target, loaded_models


def load_or_build_device_report(
    *,
    args: argparse.Namespace,
    device_run: DeviceRun,
    artifacts: RenderArtifacts,
    compute_target: Any | None,
    loaded_models: Any | None,
    encoder: str,
    encoder_options: str,
    mosaic_restoration_model_name: str,
) -> dict[str, Any]:
    """Render one input on one device, or reuse an existing output when requested."""

    artifacts.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing_outputs and artifacts.output_path.exists():
        if artifacts.report_path.exists():
            with open(artifacts.report_path, encoding="utf-8") as file_obj:
                report = json.load(file_obj)
        else:
            report = {
                "input_path": str(artifacts.input_path),
                "output_path": str(artifacts.output_path),
                "success": True,
                "output_exists": True,
            }
        report["reused_existing_output"] = True
        return report

    if compute_target is None or loaded_models is None:
        raise RuntimeError(
            f"Missing loaded models for compute target '{device_run.compute_target_id}'."
        )

    request = VideoProcessingRequest(
        input_path=str(artifacts.input_path),
        output_path=str(artifacts.output_path),
        temp_dir_path=args.temporary_directory,
        loaded_models=loaded_models,
        mosaic_restoration_model_name=mosaic_restoration_model_name,
        max_clip_length=args.max_clip_length,
        encoder=encoder,
        encoder_options=encoder_options,
        mp4_fast_start=args.mp4_fast_start,
        output_metadata=build_output_metadata(
            device_id=device_run.compute_target_id,
            mosaic_restoration_model_name=mosaic_restoration_model_name,
            mosaic_detection_model_name=args.mosaic_detection_model,
            encoder=encoder,
        ),
    )
    report = process_video_file(request)

    report["device_id"] = device_run.compute_target_id
    report["fp16"] = device_run.fp16
    report["compute_target"] = compute_target_to_report(compute_target)
    report["reused_existing_output"] = False
    write_timing_report(str(artifacts.report_path), report)
    if not report.get("success"):
        raise RuntimeError(
            f"Rendering failed for '{artifacts.input_path}' on '{device_run.compute_target_id}': "
            f"{report.get('error_message', 'unknown error')}"
        )
    return report


def render_device_reports(
    *,
    args: argparse.Namespace,
    device_run: DeviceRun,
    artifacts_map: dict[str, RenderArtifacts],
    encoder: str,
    encoder_options: str,
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_detection_model_path: str,
) -> dict[str, dict[str, Any]]:
    """Render or reuse every requested input for one device with one model load."""

    needs_render = any(
        not (args.reuse_existing_outputs and artifacts.output_path.exists())
        for artifacts in artifacts_map.values()
    )
    compute_target = None
    loaded_models = None
    if needs_render:
        compute_target, loaded_models = load_models_for_device(
            args=args,
            device_run=device_run,
            mosaic_restoration_model_name=mosaic_restoration_model_name,
            mosaic_restoration_model_path=mosaic_restoration_model_path,
            mosaic_detection_model_path=mosaic_detection_model_path,
        )

    reports: dict[str, dict[str, Any]] = {}
    try:
        for input_key, artifacts in artifacts_map.items():
            reports[input_key] = load_or_build_device_report(
                args=args,
                device_run=device_run,
                artifacts=artifacts,
                compute_target=compute_target,
                loaded_models=loaded_models,
                encoder=encoder,
                encoder_options=encoder_options,
                mosaic_restoration_model_name=mosaic_restoration_model_name,
            )
    finally:
        if loaded_models is not None:
            release_loaded_models(loaded_models)
    return reports


def release_loaded_models(loaded_models: Any) -> None:
    """Release cached runtime state after one device render pass completes."""

    for model in (loaded_models.detection_model, loaded_models.restoration_model):
        release_cached_memory = getattr(model, "release_cached_memory", None)
        if callable(release_cached_memory):
            release_cached_memory()
    del loaded_models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()


def probe_media(path: Path) -> dict[str, Any]:
    """Read basic media metadata with ffprobe for one rendered output."""

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    data = json.loads(result.stdout)
    video_stream = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
    audio_stream = next(
        (stream for stream in data["streams"] if stream["codec_type"] == "audio"),
        None,
    )
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "nb_frames": int(video_stream["nb_frames"]),
        "pix_fmt": video_stream["pix_fmt"],
        "video_codec": video_stream["codec_name"],
        "audio_codec": audio_stream["codec_name"] if audio_stream is not None else None,
        "video_duration": float(video_stream["duration"]),
        "format_duration": float(data["format"]["duration"]),
    }


def parse_metric_value(value: str) -> float:
    """Parse one ffmpeg metric token into a float, keeping infinity values."""

    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return float(value)


def collect_ffmpeg_metric(filter_name: str, left_path: Path, right_path: Path) -> dict[str, float]:
    """Run one ffmpeg comparison filter and parse the final summary values."""

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(left_path),
            "-i",
            str(right_path),
            "-lavfi",
            filter_name,
            "-an",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    text = result.stderr
    if filter_name == "psnr":
        matches = re.findall(
            r"PSNR\s+y:([0-9.inf-]+)\s+u:([0-9.inf-]+)\s+v:([0-9.inf-]+)\s+average:([0-9.inf-]+)\s+min:([0-9.inf-]+)\s+max:([0-9.inf-]+)",
            text,
        )
        if not matches:
            raise RuntimeError(f"Unable to parse PSNR output for '{left_path}' vs '{right_path}'.")
        y_value, u_value, v_value, average_value, min_value, max_value = matches[-1]
        return {
            "y": parse_metric_value(y_value),
            "u": parse_metric_value(u_value),
            "v": parse_metric_value(v_value),
            "average": parse_metric_value(average_value),
            "min": parse_metric_value(min_value),
            "max": parse_metric_value(max_value),
        }

    matches = re.findall(
        r"SSIM\s+Y:([0-9.]+)\s+\(([0-9.inf-]+)\)\s+U:([0-9.]+)\s+\(([0-9.inf-]+)\)\s+V:([0-9.]+)\s+\(([0-9.inf-]+)\)\s+All:([0-9.]+)\s+\(([0-9.inf-]+)\)",
        text,
    )
    if not matches:
        raise RuntimeError(f"Unable to parse SSIM output for '{left_path}' vs '{right_path}'.")
    y_value, _, u_value, _, v_value, _, all_value, all_db_value = matches[-1]
    return {
        "y": parse_metric_value(y_value),
        "u": parse_metric_value(u_value),
        "v": parse_metric_value(v_value),
        "all": parse_metric_value(all_value),
        "all_db": parse_metric_value(all_db_value),
    }


def collect_frame_pair_stats(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Decode both videos and collect brightness plus frame-difference statistics."""

    frame_count = 0
    identical_frames = 0
    global_max_abs_diff = 0
    average_frame_mae = 0.0
    max_frame_mae = 0.0
    left_sum_mean = 0.0
    right_sum_mean = 0.0
    left_min_mean = math.inf
    right_min_mean = math.inf
    left_max_mean = -math.inf
    right_max_mean = -math.inf

    with av.open(str(left_path)) as left_container, av.open(str(right_path)) as right_container:
        left_frames = left_container.decode(video=0)
        right_frames = right_container.decode(video=0)
        while True:
            try:
                left_frame = next(left_frames)
            except StopIteration:
                left_frame = None
            try:
                right_frame = next(right_frames)
            except StopIteration:
                right_frame = None

            if left_frame is None and right_frame is None:
                break
            if left_frame is None or right_frame is None:
                raise RuntimeError("Decoded frame count mismatch between compared outputs.")

            left_array = left_frame.to_ndarray(format="rgb24").astype(np.int16)
            right_array = right_frame.to_ndarray(format="rgb24").astype(np.int16)
            if left_array.shape != right_array.shape:
                raise RuntimeError(
                    f"Decoded frame shape mismatch: {left_array.shape} vs {right_array.shape}"
                )

            left_mean = float(left_array.mean())
            right_mean = float(right_array.mean())
            left_sum_mean += left_mean
            right_sum_mean += right_mean
            left_min_mean = min(left_min_mean, left_mean)
            right_min_mean = min(right_min_mean, right_mean)
            left_max_mean = max(left_max_mean, left_mean)
            right_max_mean = max(right_max_mean, right_mean)

            frame_diff = np.abs(left_array - right_array)
            frame_mae = float(frame_diff.mean())
            frame_max = int(frame_diff.max())

            frame_count += 1
            average_frame_mae += frame_mae
            max_frame_mae = max(max_frame_mae, frame_mae)
            global_max_abs_diff = max(global_max_abs_diff, frame_max)
            if frame_max == 0:
                identical_frames += 1

    return {
        "frame_diff": {
            "frame_count": frame_count,
            "avg_frame_mae": average_frame_mae / frame_count if frame_count else 0.0,
            "max_frame_mae": max_frame_mae,
            "global_max_abs_diff": global_max_abs_diff,
            "identical_frames": identical_frames,
        },
        "left_frame_stats": {
            "frames": frame_count,
            "min_mean": left_min_mean,
            "max_mean": left_max_mean,
            "avg_mean": left_sum_mean / frame_count if frame_count else 0.0,
        },
        "right_frame_stats": {
            "frames": frame_count,
            "min_mean": right_min_mean,
            "max_mean": right_max_mean,
            "avg_mean": right_sum_mean / frame_count if frame_count else 0.0,
        },
    }


def compare_outputs(
    *,
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    """Compare two rendered outputs and return one structured report block."""

    baseline_media = probe_media(baseline_path)
    candidate_media = probe_media(candidate_path)
    frame_pair_stats = collect_frame_pair_stats(baseline_path, candidate_path)
    metadata_match = {
        key: baseline_media[key] == candidate_media[key]
        for key in ("width", "height", "nb_frames", "pix_fmt", "video_codec", "audio_codec")
    }
    metadata_match["all"] = all(metadata_match.values())
    return {
        "baseline_media": baseline_media,
        "candidate_media": candidate_media,
        "metadata_match": metadata_match,
        "psnr": collect_ffmpeg_metric("psnr", baseline_path, candidate_path),
        "ssim": collect_ffmpeg_metric("ssim", baseline_path, candidate_path),
        "baseline_frame_stats": frame_pair_stats["left_frame_stats"],
        "candidate_frame_stats": frame_pair_stats["right_frame_stats"],
        "frame_diff": frame_pair_stats["frame_diff"],
    }


def run_regression(args: argparse.Namespace) -> Path:
    """Run the full render-and-compare workflow and write the master report."""

    baseline_device = resolve_device_run(args.baseline_device, args.fp16)
    candidate_device = resolve_device_run(args.candidate_device, args.fp16)
    if baseline_device.compute_target_id == candidate_device.compute_target_id:
        raise RuntimeError("Baseline and candidate devices must be different.")
    if not os.path.isdir(args.temporary_directory):
        os.makedirs(args.temporary_directory)

    input_files, input_root = resolve_input_files(args.input)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_name = args.report_name or (
        f"{baseline_device.output_suffix}-vs-{candidate_device.output_suffix}.json"
    )
    report_path = output_dir / report_name
    encoder, encoder_options = resolve_encoder_settings(args)
    mosaic_restoration_model_name, mosaic_restoration_model_path, mosaic_detection_model_path = resolve_model_paths(args)

    if args.pipeline_revision:
        os.environ[_PIPELINE_REVISION_ENV] = args.pipeline_revision

    baseline_artifacts_map = {
        str(input_path): build_render_artifacts(
            input_path=input_path,
            input_root=input_root,
            output_dir=output_dir,
            output_suffix=baseline_device.output_suffix,
        )
        for input_path in input_files
    }
    candidate_artifacts_map = {
        str(input_path): build_render_artifacts(
            input_path=input_path,
            input_root=input_root,
            output_dir=output_dir,
            output_suffix=candidate_device.output_suffix,
        )
        for input_path in input_files
    }
    baseline_reports = render_device_reports(
        args=args,
        device_run=baseline_device,
        artifacts_map=baseline_artifacts_map,
        encoder=encoder,
        encoder_options=encoder_options,
        mosaic_restoration_model_name=mosaic_restoration_model_name,
        mosaic_restoration_model_path=mosaic_restoration_model_path,
        mosaic_detection_model_path=mosaic_detection_model_path,
    )
    candidate_reports = render_device_reports(
        args=args,
        device_run=candidate_device,
        artifacts_map=candidate_artifacts_map,
        encoder=encoder,
        encoder_options=encoder_options,
        mosaic_restoration_model_name=mosaic_restoration_model_name,
        mosaic_restoration_model_path=mosaic_restoration_model_path,
        mosaic_detection_model_path=mosaic_detection_model_path,
    )

    per_file_results: list[dict[str, Any]] = []
    for input_path in input_files:
        input_key = str(input_path)
        baseline_artifacts = baseline_artifacts_map[input_key]
        candidate_artifacts = candidate_artifacts_map[input_key]
        baseline_report = baseline_reports[input_key]
        candidate_report = candidate_reports[input_key]
        comparison = compare_outputs(
            baseline_path=baseline_artifacts.output_path,
            candidate_path=candidate_artifacts.output_path,
        )
        per_file_results.append(
            {
                "input_path": str(input_path),
                "baseline": {
                    "device_id": baseline_device.compute_target_id,
                    "output_path": str(baseline_artifacts.output_path),
                    "report_path": str(baseline_artifacts.report_path),
                    "report": baseline_report,
                },
                "candidate": {
                    "device_id": candidate_device.compute_target_id,
                    "output_path": str(candidate_artifacts.output_path),
                    "report_path": str(candidate_artifacts.report_path),
                    "report": candidate_report,
                },
                "comparison": comparison,
            }
        )
        print(
            f"{input_path.name}: "
            f"PSNR={comparison['psnr']['average']:.2f}dB, "
            f"SSIM={comparison['ssim']['all']:.6f}, "
            f"avg_frame_mae={comparison['frame_diff']['avg_frame_mae']:.4f}"
        )

    master_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_path": str(Path(args.input).resolve()),
        "output_dir": str(output_dir),
        "baseline_device": baseline_device.compute_target_id,
        "candidate_device": candidate_device.compute_target_id,
        "encoder": encoder,
        "encoder_options": encoder_options,
        "pipeline_revision": args.pipeline_revision,
        "files": per_file_results,
    }
    with open(report_path, "w", encoding="utf-8") as file_obj:
        json.dump(master_report, file_obj, indent=2, ensure_ascii=False)
    print(f"Wrote comparison report to {report_path}")
    return report_path


def main() -> None:
    """Parse arguments and run the manual device regression workflow."""

    args = build_argparser().parse_args()
    run_regression(args)
