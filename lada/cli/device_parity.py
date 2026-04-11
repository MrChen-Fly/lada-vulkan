from __future__ import annotations

import argparse
import json
from pathlib import Path

from lada import ModelFiles
from lada.cli import utils
from lada.compute_targets import get_default_compute_target
from lada.parity import run_device_parity


def _resolve_input_files(input_args: list[str]) -> list[Path]:
    files: list[Path] = []
    for input_arg in input_args:
        input_path = Path(input_arg).resolve()
        if input_path.is_file():
            files.append(input_path)
            continue
        if input_path.is_dir():
            files.extend(Path(path).resolve() for path in utils.list_video_files(str(input_path)))
            continue
        raise RuntimeError(f"Invalid input path '{input_arg}'.")
    if not files:
        raise RuntimeError("No input video files were resolved.")
    return files


def _resolve_model_path(model_name_or_path: str, *, kind: str) -> str:
    if kind == "detection":
        model = ModelFiles.get_detection_model_by_name(model_name_or_path)
    else:
        model = ModelFiles.get_restoration_model_by_name(model_name_or_path)
    if model is not None:
        return model.path
    if Path(model_name_or_path).is_file():
        return str(Path(model_name_or_path).resolve())
    raise RuntimeError(f"Invalid {kind} model '{model_name_or_path}'.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one Torch reference device against vulkan:0 and dump probe statistics.",
    )
    parser.add_argument("--input", required=True, nargs="+", help="Input video file(s) or directory/directories.")
    parser.add_argument("--output-dir", required=True, help="Directory used to store parity JSON reports.")
    parser.add_argument(
        "--reference-device",
        default=get_default_compute_target(),
        help="Torch reference device. Use cuda:0 when available; CPU is acceptable for semantic baselines.",
    )
    parser.add_argument(
        "--mosaic-restoration-model",
        default="basicvsrpp-v1.2",
        help="Restoration model name or checkpoint path.",
    )
    parser.add_argument(
        "--mosaic-restoration-config-path",
        default=None,
        help="Optional restoration config path for custom checkpoints.",
    )
    parser.add_argument(
        "--mosaic-detection-model",
        default="v4-fast",
        help="Detection model name or checkpoint path.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index used for the clip.")
    parser.add_argument("--frame-count", type=int, default=5, help="Number of consecutive frames used for restore probes.")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fp16 where supported. Default is false to keep parity runs conservative.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional directory used to cache Vulkan modular artifacts.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = _resolve_input_files(args.input)
    detection_model_path = _resolve_model_path(args.mosaic_detection_model, kind="detection")
    restoration_model_path = _resolve_model_path(args.mosaic_restoration_model, kind="restoration")
    for input_path in input_files:
        report = run_device_parity(
            input_path,
            reference_device_id=args.reference_device,
            detection_model_path=detection_model_path,
            restoration_model_path=restoration_model_path,
            restoration_config_path=args.mosaic_restoration_config_path,
            start_frame=args.start_frame,
            frame_count=args.frame_count,
            fp16=bool(args.fp16),
            artifacts_dir=args.artifacts_dir,
        )
        output_path = output_dir / f"{input_path.stem}.device-parity.json"
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(output_path)


if __name__ == "__main__":
    main()

