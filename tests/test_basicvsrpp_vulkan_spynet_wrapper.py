from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from lada.restorationpipeline.basicvsrpp_vulkan_restorer import (
    NcnnVulkanBasicvsrppMosaicRestorer,
)
from lada.restorationpipeline.basicvsrpp_vulkan_runtime_support import (
    resolve_basicvsrpp_runtime_shape,
)


class _FakeProfiler:
    def measure(self, _bucket):
        return nullcontext()


class _FakeSpynetRunner:
    def __init__(self, output: np.ndarray):
        self.output = output
        self.gpu_runner = None
        self.last_inputs: dict[str, np.ndarray] | None = None
        self.run_calls = 0
        self.gpu_run_calls = 0
        self.gpu_download_calls = 0

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        self.run_calls += 1
        self.last_inputs = inputs
        return self.output

    def run_gpu(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        self.gpu_run_calls += 1
        self.last_inputs = inputs
        return self.output

    def run_gpu_download(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        self.gpu_download_calls += 1
        self.last_inputs = inputs
        return self.output


def _make_restorer(runners: dict[str, _FakeSpynetRunner]) -> SimpleNamespace:
    restorer = SimpleNamespace(
        profiler=_FakeProfiler(),
        runners=runners,
        ncnn=SimpleNamespace(),
        runtime_shape=resolve_basicvsrpp_runtime_shape((256, 256)),
    )
    restorer._should_use_gpu_download = (
        NcnnVulkanBasicvsrppMosaicRestorer._should_use_gpu_download.__get__(
            restorer,
            SimpleNamespace,
        )
    )
    restorer._run_profiled_module = (
        NcnnVulkanBasicvsrppMosaicRestorer._run_profiled_module.__get__(
            restorer,
            SimpleNamespace,
        )
    )
    return restorer


class BasicvsrppVulkanSpynetWrapperTest(unittest.TestCase):
    def test_run_spynet_resizes_fixed_runtime_flow_back_to_input_shape(self) -> None:
        export_flow = np.zeros((2, 192, 320), dtype=np.float32)
        export_flow[0, :, :] = 1.0
        export_flow[1, :, :] = 2.0
        runner = _FakeSpynetRunner(export_flow)
        restorer = _make_restorer({"spynet": runner})

        ref = np.zeros((3, 64, 64), dtype=np.float32)
        supp = np.zeros_like(ref)
        flow = NcnnVulkanBasicvsrppMosaicRestorer.run_spynet(restorer, ref, supp)

        self.assertEqual((2, 64, 64), flow.shape)
        self.assertIsNotNone(runner.last_inputs)
        self.assertEqual((3, 192, 320), runner.last_inputs["in0"].shape)
        self.assertEqual((3, 192, 320), runner.last_inputs["in1"].shape)
        self.assertTrue(np.allclose(flow[0], 64.0 / 320.0))
        self.assertTrue(np.allclose(flow[1], 64.0 / 192.0 * 2.0))

    def test_run_spynet_uses_patch_runner_for_64x64_inputs(self) -> None:
        core_runner = _FakeSpynetRunner(np.zeros((2, 192, 320), dtype=np.float32))
        patch_runner = _FakeSpynetRunner(np.ones((2, 64, 64), dtype=np.float32))
        restorer = _make_restorer(
            {
                "spynet": core_runner,
                "spynet_patch": patch_runner,
            }
        )

        ref = np.zeros((3, 64, 64), dtype=np.float32)
        supp = np.zeros_like(ref)
        flow = NcnnVulkanBasicvsrppMosaicRestorer.run_spynet(restorer, ref, supp)

        self.assertEqual((2, 64, 64), flow.shape)
        self.assertIsNone(core_runner.last_inputs)
        self.assertIsNotNone(patch_runner.last_inputs)
        self.assertEqual((3, 64, 64), patch_runner.last_inputs["in0"].shape)
        self.assertTrue(np.allclose(flow, 1.0))

    def test_run_spynet_uses_shape_specific_patch_runner(self) -> None:
        core_runner = _FakeSpynetRunner(np.zeros((2, 320, 448), dtype=np.float32))
        patch_runner = _FakeSpynetRunner(np.ones((2, 80, 112), dtype=np.float32))
        restorer = _make_restorer(
            {
                "spynet": core_runner,
                "spynet_patch": patch_runner,
            }
        )
        restorer.runtime_shape = resolve_basicvsrpp_runtime_shape((320, 448))

        ref = np.zeros((3, 80, 112), dtype=np.float32)
        supp = np.zeros_like(ref)
        flow = NcnnVulkanBasicvsrppMosaicRestorer.run_spynet(restorer, ref, supp)

        self.assertEqual((2, 80, 112), flow.shape)
        self.assertIsNone(core_runner.last_inputs)
        self.assertEqual((3, 80, 112), patch_runner.last_inputs["in0"].shape)
        self.assertTrue(np.allclose(flow, 1.0))

    def test_run_spynet_patch_uses_gpu_download_path(self) -> None:
        core_runner = _FakeSpynetRunner(np.zeros((2, 192, 320), dtype=np.float32))
        patch_runner = _FakeSpynetRunner(np.ones((2, 64, 64), dtype=np.float32))
        patch_runner.gpu_runner = object()
        restorer = _make_restorer(
            {
                "spynet": core_runner,
                "spynet_patch": patch_runner,
            }
        )

        ref = np.zeros((3, 64, 64), dtype=np.float32)
        supp = np.zeros_like(ref)
        flow = NcnnVulkanBasicvsrppMosaicRestorer.run_spynet(
            restorer,
            ref,
            supp,
            prefer_gpu_download=True,
        )

        self.assertEqual((2, 64, 64), flow.shape)
        self.assertEqual(1, patch_runner.gpu_download_calls)
        self.assertEqual(0, patch_runner.run_calls)
        self.assertIsNotNone(patch_runner.last_inputs)
        self.assertTrue(np.allclose(flow, 1.0))

    def test_run_profiled_module_output_frame_uses_gpu_download_path(self) -> None:
        output_runner = _FakeSpynetRunner(np.ones((3, 64, 64), dtype=np.float32))
        output_runner.gpu_runner = object()
        restorer = _make_restorer({"output_frame": output_runner})

        output = restorer._run_profiled_module(
            "output_frame",
            {"in0": np.zeros((3, 64, 64), dtype=np.float32)},
            bucket=None,
            prefer_gpu_download=True,
        )

        self.assertEqual(0, output_runner.run_calls)
        self.assertEqual(1, output_runner.gpu_download_calls)
        self.assertTrue(np.allclose(output, 1.0))

    def test_run_profiled_module_output_frame_passes_gpu_blob_inputs_to_gpu_download(self) -> None:
        output_runner = _FakeSpynetRunner(np.ones((3, 64, 64), dtype=np.float32))
        output_runner.gpu_runner = object()
        restorer = _make_restorer({"output_frame": output_runner})
        gpu_blob = object()
        restorer.ncnn.LadaVulkanTensor = object

        restorer._run_profiled_module(
            "output_frame",
            {"in0": gpu_blob},
            bucket=None,
            prefer_gpu_download=True,
        )

        self.assertIs(output_runner.last_inputs["in0"], gpu_blob)
        self.assertEqual(0, output_runner.run_calls)
        self.assertEqual(1, output_runner.gpu_download_calls)

    def test_restore_switches_runtime_for_uniform_input_shape(self) -> None:
        restorer = SimpleNamespace(
            runtime_shape=resolve_basicvsrpp_runtime_shape((256, 256)),
            _ensure_runtime_for_input_shape=mock.Mock(),
        )
        restorer._resolve_runtime_input_shape = (
            NcnnVulkanBasicvsrppMosaicRestorer._resolve_runtime_input_shape.__get__(
                restorer,
                SimpleNamespace,
            )
        )
        frames = [np.zeros((320, 448, 3), dtype=np.uint8)]

        with mock.patch(
            "lada.restorationpipeline.basicvsrpp_vulkan_restorer.run_restore",
            return_value=[],
        ) as run_restore:
            result = NcnnVulkanBasicvsrppMosaicRestorer.restore(restorer, frames)

        self.assertEqual([], result)
        restorer._ensure_runtime_for_input_shape.assert_called_once_with((320, 448))
        run_restore.assert_called_once_with(restorer, frames, max_frames=-1)

    def test_restore_cropped_clip_frames_switches_runtime_to_descriptor_size(self) -> None:
        restorer = SimpleNamespace(
            _ensure_runtime_for_input_shape=mock.Mock(),
        )
        frames = [np.zeros((120, 160, 3), dtype=np.uint8)]

        with mock.patch(
            "lada.restorationpipeline.basicvsrpp_vulkan_restorer.run_restore_cropped_clip_frames",
            return_value=[],
        ) as run_restore:
            result = NcnnVulkanBasicvsrppMosaicRestorer.restore_cropped_clip_frames(
                restorer,
                frames,
                size=320,
                resize_reference_shape=(160, 120),
                pad_mode="zero",
            )

        self.assertEqual([], result)
        restorer._ensure_runtime_for_input_shape.assert_called_once_with((320, 320))
        run_restore.assert_called_once_with(
            restorer,
            frames,
            size=320,
            resize_reference_shape=(160, 120),
            pad_mode="zero",
        )


if __name__ == "__main__":
    unittest.main()
