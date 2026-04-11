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
)
from lada.restorationpipeline.basicvsrpp_vulkan_recurrent_runtime import (
    run_branch_recurrent,
)
from lada.restorationpipeline.basicvsrpp_vulkan_restorer import (
    NcnnVulkanBasicvsrppMosaicRestorer,
)


def _build_smooth_lqs(
    height: int,
    width: int,
    *,
    frame_count: int = 5,
) -> list[np.ndarray]:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    lqs: list[np.ndarray] = []
    for index in range(frame_count):
        frame = np.empty((3, height, width), dtype=np.float32)
        frame[0] = np.clip(x * 0.55 + y * 0.15 + index * 0.013, 0.0, 1.0)
        frame[1] = np.clip(x * 0.20 + y * 0.70 + 0.10 + index * 0.009, 0.0, 1.0)
        frame[2] = np.clip(x * 0.10 + y * 0.85 + 0.18 + index * 0.007, 0.0, 1.0)
        lqs.append(np.ascontiguousarray(frame, dtype=np.float32))
    return lqs


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


def _build_dynamic_runtime_restorer(*, fp16: bool) -> NcnnVulkanBasicvsrppMosaicRestorer:
    ncnn = import_ncnn_module()
    if not ncnn_has_lada_basicvsrpp_clip_runner(ncnn):
        raise unittest.SkipTest(
            "Local ncnn runtime does not expose the BasicVSR++ clip runner."
        )

    model_path = Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")
    if not model_path.exists():
        raise unittest.SkipTest(f"Missing restoration model artifact: {model_path}")

    restorer = NcnnVulkanBasicvsrppMosaicRestorer(
        str(model_path),
        config_path=None,
        fp16=fp16,
        artifacts_dir=None,
    )
    restorer._ensure_runtime_for_input_shape((320, 448))
    if restorer.native_clip_runner is None:
        restorer.release_cached_memory()
        raise unittest.SkipTest("Native Vulkan clip runner is unavailable.")
    return restorer


def _assert_dynamic_runtime_trace_matches_modular_gpu_bridge(
    testcase: unittest.TestCase,
    restorer: NcnnVulkanBasicvsrppMosaicRestorer,
) -> None:
    baseline_lqs = _build_smooth_lqs(320, 448)
    native_trace = restorer.native_clip_runner.debug_trace(baseline_lqs)
    modular_trace = _build_modular_gpu_bridge_trace(restorer, baseline_lqs)

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
        testcase.assertTrue(bool(summary["finite"]), msg=stage_name)
        testcase.assertLess(float(summary["max_mean_abs_diff"]), 1e-7, msg=stage_name)
        testcase.assertLess(float(summary["max_max_abs_diff"]), 1e-6, msg=stage_name)


class BasicvsrppVulkanDynamicRuntimeTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.restorer = _build_dynamic_runtime_restorer(fp16=True)

    @classmethod
    def tearDownClass(cls) -> None:
        restorer = getattr(cls, "restorer", None)
        if restorer is not None:
            restorer.release_cached_memory()

    def test_shape_matched_dynamic_runtime_native_trace_matches_modular_gpu_bridge(self) -> None:
        _assert_dynamic_runtime_trace_matches_modular_gpu_bridge(self, self.restorer)


class BasicvsrppVulkanDynamicRuntimeFp32TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.restorer = _build_dynamic_runtime_restorer(fp16=False)

    @classmethod
    def tearDownClass(cls) -> None:
        restorer = getattr(cls, "restorer", None)
        if restorer is not None:
            restorer.release_cached_memory()

    def test_shape_matched_dynamic_runtime_native_trace_matches_modular_gpu_bridge(self) -> None:
        _assert_dynamic_runtime_trace_matches_modular_gpu_bridge(self, self.restorer)


if __name__ == "__main__":
    unittest.main()
