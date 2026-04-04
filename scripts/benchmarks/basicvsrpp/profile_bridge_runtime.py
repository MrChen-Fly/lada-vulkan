import json
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lada.models.basicvsrpp.ncnn_vulkan import is_ncnn_vulkan_tensor
from lada.restorationpipeline.basicvsrpp_vulkan_restorer import (
    NcnnVulkanBasicvsrppMosaicRestorer,
)
from lada.utils.video_utils import read_video_frames


def _summarize_inputs(
    model: NcnnVulkanBasicvsrppMosaicRestorer,
    inputs: dict[str, object],
) -> dict[str, Any]:
    first_runner = next(iter(model.runners.values()))
    gpu_inputs = 0
    cpu_inputs = 0
    shapes: dict[str, list[int] | None] = {}
    for name, value in inputs.items():
        if is_ncnn_vulkan_tensor(first_runner.ncnn, value):
            gpu_inputs += 1
            shape = [int(value.c), int(value.h), int(value.w)] if value.dims == 3 else None
        else:
            cpu_inputs += 1
            array = np.asarray(value)
            shape = [int(dim) for dim in array.shape]
        shapes[name] = shape
    return {
        "gpu_inputs": gpu_inputs,
        "cpu_inputs": cpu_inputs,
        "shapes": shapes,
    }


def _benchmark_case(*, name: str, fn, repeats: int, warmup: int = 1) -> dict[str, Any]:
    last_output = None
    for _ in range(max(warmup, 0)):
        last_output = fn()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for _ in range(repeats):
        last_output = fn()
    cpu_end = time.process_time()
    wall_end = time.perf_counter()

    output_kind = type(last_output).__name__
    output_shape = None
    if isinstance(last_output, np.ndarray):
        output_shape = [int(dim) for dim in last_output.shape]
    elif hasattr(last_output, "dims") and int(last_output.dims) == 3:
        output_shape = [int(last_output.c), int(last_output.h), int(last_output.w)]

    return {
        "name": name,
        "repeats": repeats,
        "wall_s": wall_end - wall_start,
        "cpu_s": cpu_end - cpu_start,
        "wall_per_iter_s": (wall_end - wall_start) / repeats,
        "cpu_per_iter_s": (cpu_end - cpu_start) / repeats,
        "output_kind": output_kind,
        "output_shape": output_shape,
    }


def _prepare_fused_inputs(
    model: NcnnVulkanBasicvsrppMosaicRestorer,
    frames: list[torch.Tensor],
) -> dict[str, object]:
    lqs, spatial_feats, flows_backward, flows_forward = model._prepare_clip_features(frames)
    return {
        "in0": lqs[0],
        "in1": lqs[1],
        "in2": lqs[2],
        "in3": lqs[3],
        "in4": lqs[4],
        "in5": spatial_feats[0],
        "in6": spatial_feats[1],
        "in7": spatial_feats[2],
        "in8": spatial_feats[3],
        "in9": spatial_feats[4],
        "in10": flows_backward[0],
        "in11": flows_backward[1],
        "in12": flows_backward[2],
        "in13": flows_backward[3],
        "in14": flows_forward[0],
        "in15": flows_forward[1],
        "in16": flows_forward[2],
        "in17": flows_forward[3],
    }


def _build_argparser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Profile the fused BasicVSR++ restore_clip Vulkan runtime."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to a temp-backed .helloagents/tmp location.",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    root = Path.cwd()
    input_path = root / "resources" / "main.webm"
    default_output_root = root / ".helloagents" / "tmp"
    if not default_output_root.exists():
        default_output_root = Path(tempfile.gettempdir()) / "lada"
    result_path = args.output or (default_output_root / "bridge_runtime_profile_5f.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = root / "model_weights" / "lada_mosaic_restoration_model_generic_v1.2.pth"

    frames_np = read_video_frames(str(input_path), float32=False, start_idx=0, end_idx=5)
    frames = [torch.from_numpy(np.ascontiguousarray(frame.copy())) for frame in frames_np]

    model = NcnnVulkanBasicvsrppMosaicRestorer(
        str(model_path),
        fp16=False,
        frame_count=5,
    )
    if not model.use_gpu_blob_bridge:
        raise RuntimeError("GPU blob bridge is unavailable; cannot profile fused restore clip runtime.")

    fused_inputs = _prepare_fused_inputs(model, frames)
    runner = model.runners["restore_clip"]
    cases = [
        _benchmark_case(
            name="restore_clip_gpu_download",
            fn=lambda: runner.run_gpu_download(fused_inputs),
            repeats=5,
        ),
    ]

    for case in cases:
        case["module_name"] = "restore_clip"
        case["download_output"] = True
        case["input_summary"] = _summarize_inputs(model, fused_inputs)

    output = {
        "input_path": str(input_path),
        "frame_count": len(frames),
        "use_gpu_blob_bridge": model.use_gpu_blob_bridge,
        "use_fused_restore_clip": model.use_fused_restore_clip,
        "num_threads": model.num_threads,
        "cases": cases,
    }
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
