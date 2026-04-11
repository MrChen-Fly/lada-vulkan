from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from lada.models.basicvsrpp.ncnn_vulkan import (
    import_ncnn_module,
    ncnn_has_lada_basicvsrpp_clip_runner,
)
from lada.restorationpipeline.basicvsrpp_vulkan_io import (
    _build_output_frame_inputs,
    _frame_to_chw_float32,
)
from lada.restorationpipeline.basicvsrpp_vulkan_recurrent_runtime import (
    run_branch_recurrent,
)
from lada.restorationpipeline.basicvsrpp_vulkan_restorer import (
    NcnnVulkanBasicvsrppMosaicRestorer,
)
from lada.restorationpipeline.clip_units import Clip, ClipDescriptor


def _build_smooth_descriptor(*, size: int = 64) -> ClipDescriptor:
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []
    crop_boxes: list[tuple[int, int, int, int]] = []

    height, width = 96, 160
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    for index in range(5):
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.clip((x * 140 + y * 40 + index * 7), 0, 255).astype(np.uint8)
        frame[..., 1] = np.clip((x * 60 + y * 120 + 30 + index * 5), 0, 255).astype(np.uint8)
        frame[..., 2] = np.clip((x * 20 + y * 180 + 50 + index * 3), 0, 255).astype(np.uint8)
        frames.append(np.ascontiguousarray(frame))

        top = 10 + index
        left = 25 + index * 2
        bottom = 74 + index
        right = 118 + index * 2
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[top : bottom + 1, left : right + 1] = 255
        masks.append(mask)
        boxes.append((top, left, bottom, right))
        crop_boxes.append((top, left, bottom, right))

    resize_reference_shape = (
        max(right - left + 1 for _, left, _, right in crop_boxes),
        max(bottom - top + 1 for top, _, bottom, _ in crop_boxes),
    )
    return ClipDescriptor(
        file_path="synthetic.webm",
        frame_start=0,
        size=int(size),
        pad_mode="reflect",
        id="synthetic-small-runtime",
        frames=frames,
        masks=masks,
        boxes=boxes,
        crop_boxes=crop_boxes,
        resize_reference_shape=resize_reference_shape,
    )


def _stage_diff_summary(
    baseline: list[np.ndarray],
    candidate: list[np.ndarray],
) -> dict[str, float | bool]:
    max_mean_abs_diff = 0.0
    max_max_abs_diff = 0.0
    finite = True
    for baseline_value, candidate_value in zip(baseline, candidate, strict=False):
        baseline_array = np.asarray(baseline_value, dtype=np.float32)
        candidate_array = np.asarray(candidate_value, dtype=np.float32)
        finite = finite and bool(np.isfinite(candidate_array).all())
        diff = np.abs(baseline_array - candidate_array)
        max_mean_abs_diff = max(max_mean_abs_diff, float(diff.mean()))
        max_max_abs_diff = max(max_max_abs_diff, float(diff.max()))
    return {
        "finite": finite,
        "max_mean_abs_diff": max_mean_abs_diff,
        "max_max_abs_diff": max_max_abs_diff,
    }


def _build_modular_gpu_bridge_trace(
    restorer: NcnnVulkanBasicvsrppMosaicRestorer,
    lqs: list[np.ndarray],
) -> dict[str, list[np.ndarray]]:
    quarter_gpu = [
        restorer._run_profiled_module(
            "quarter_downsample",
            {"in0": lq},
            bucket=None,
            prefer_gpu=True,
        )
        for lq in lqs
    ]
    spatial_gpu = [
        restorer._run_profiled_module(
            "feat_extract",
            {"in0": lq},
            bucket=None,
            prefer_gpu=True,
        )
        for lq in lqs
    ]
    quarter_cpu = [
        np.asarray(restorer.runners["quarter_downsample"].download_gpu(value), dtype=np.float32)
        for value in quarter_gpu
    ]
    spatial_cpu = [
        np.asarray(restorer.runners["feat_extract"].download_gpu(value), dtype=np.float32)
        for value in spatial_gpu
    ]
    flows_backward = [
        np.asarray(
            restorer.run_spynet(
                quarter_cpu[index],
                quarter_cpu[index + 1],
                prefer_gpu_download=True,
            ),
            dtype=np.float32,
        )
        for index in range(len(quarter_cpu) - 1)
    ]
    flows_forward = [
        np.asarray(
            restorer.run_spynet(
                quarter_cpu[index + 1],
                quarter_cpu[index],
                prefer_gpu_download=True,
            ),
            dtype=np.float32,
        )
        for index in range(len(quarter_cpu) - 1)
    ]

    branch_feats_gpu: dict[str, list[object]] = {}
    branch_feats_cpu: dict[str, list[np.ndarray]] = {}
    for module_name in ("backward_1", "forward_1", "backward_2", "forward_2"):
        flows = flows_backward if module_name.startswith("backward") else flows_forward
        branch_outputs = run_branch_recurrent(
            restorer,
            module_name,
            spatial_gpu,
            branch_feats_gpu,
            flows,
            use_gpu_bridge=True,
        )
        branch_feats_gpu[module_name] = branch_outputs
        branch_feats_cpu[module_name] = [
            np.asarray(
                restorer.runners[f"{module_name}_backbone"].download_gpu(value),
                dtype=np.float32,
            )
            for value in branch_outputs
        ]

    output_frame = [
        np.asarray(
            restorer._run_profiled_module(
                "output_frame",
                _build_output_frame_inputs(lqs, spatial_gpu, branch_feats_gpu, frame_index),
                bucket=None,
                prefer_gpu_download=True,
            ),
            dtype=np.float32,
        )
        for frame_index in range(len(lqs))
    ]
    return {
        "quarter_downsample": quarter_cpu,
        "feat_extract": spatial_cpu,
        "flows_backward": flows_backward,
        "flows_forward": flows_forward,
        "backward_1": branch_feats_cpu["backward_1"],
        "forward_1": branch_feats_cpu["forward_1"],
        "backward_2": branch_feats_cpu["backward_2"],
        "forward_2": branch_feats_cpu["forward_2"],
        "output_frame": output_frame,
    }


class BasicvsrppVulkanSmallRuntimeTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ncnn = import_ncnn_module()
        if not ncnn_has_lada_basicvsrpp_clip_runner(cls.ncnn):
            raise unittest.SkipTest(
                "Local ncnn runtime does not expose the BasicVSR++ clip runner."
            )

        cls.model_path = Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")
        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Missing restoration model artifact: {cls.model_path}")

        cls.restorer = NcnnVulkanBasicvsrppMosaicRestorer(
            str(cls.model_path),
            config_path=None,
            fp16=True,
            artifacts_dir=None,
        )
        cls.restorer._ensure_runtime_for_input_shape((64, 64))
        if cls.restorer.native_clip_runner is None:
            raise unittest.SkipTest("Native Vulkan clip runner is unavailable.")

    @classmethod
    def tearDownClass(cls) -> None:
        restorer = getattr(cls, "restorer", None)
        if restorer is not None:
            restorer.release_cached_memory()

    def test_shape_matched_small_runtime_native_trace_matches_modular_gpu_bridge(self) -> None:
        descriptor = _build_smooth_descriptor(size=64)
        clip = Clip.from_descriptor(descriptor, resize_mode="torch_bilinear")
        baseline_lqs = [_frame_to_chw_float32(frame) for frame in clip.frames]

        native_trace = self.restorer.native_clip_runner.debug_trace(baseline_lqs)
        modular_trace = _build_modular_gpu_bridge_trace(self.restorer, baseline_lqs)

        for stage_name in (
            "quarter_downsample",
            "feat_extract",
            "flows_backward",
            "flows_forward",
            "backward_1",
            "forward_1",
            "backward_2",
            "forward_2",
            "output_frame",
        ):
            summary = _stage_diff_summary(
                modular_trace[stage_name],
                native_trace[stage_name],
            )
            self.assertTrue(bool(summary["finite"]), msg=stage_name)
            self.assertLess(float(summary["max_mean_abs_diff"]), 1e-7, msg=stage_name)
            self.assertLess(float(summary["max_max_abs_diff"]), 1e-6, msg=stage_name)


if __name__ == "__main__":
    unittest.main()
