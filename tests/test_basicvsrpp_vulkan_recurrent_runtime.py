from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import numpy as np

from lada.restorationpipeline.basicvsrpp_vulkan_recurrent_runtime import (
    run_branch_recurrent,
    run_propagate_step,
)


class _FakeProfiler:
    def measure(self, _bucket):
        return nullcontext()


class _FakeGpuBlob:
    def __init__(self, name: str):
        self.name = name
        self.c = 1
        self.elempack = 1
        self.h = 2
        self.w = 2


class _FakeRunner:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.gpu_runner = object()
        self.run_calls = 0
        self.gpu_calls = 0
        self.gpu_download_calls = 0
        self.last_inputs: dict[str, object] | None = None

    def _make_cpu_output(self, index: int) -> np.ndarray:
        return np.full((1, 2, 2), float(index), dtype=np.float32)

    def run(self, inputs: dict[str, object]) -> np.ndarray:
        self.run_calls += 1
        self.last_inputs = inputs
        return self._make_cpu_output(self.run_calls)

    def run_gpu(self, inputs: dict[str, object]) -> _FakeGpuBlob:
        self.gpu_calls += 1
        self.last_inputs = inputs
        return _FakeGpuBlob(f"{self.prefix}_gpu_{self.gpu_calls}")

    def run_gpu_download(self, inputs: dict[str, object]) -> np.ndarray:
        self.gpu_download_calls += 1
        self.last_inputs = inputs
        return self._make_cpu_output(self.gpu_download_calls)


def _make_restorer(runners: dict[str, _FakeRunner]) -> SimpleNamespace:
    restorer = SimpleNamespace(
        profiler=_FakeProfiler(),
        runners=runners,
        ncnn=SimpleNamespace(LadaVulkanTensor=_FakeGpuBlob),
    )

    def _run_profiled_module(
        module_name: str,
        inputs: dict[str, object],
        *,
        bucket: str | None,
        prefer_gpu: bool = False,
        prefer_gpu_download: bool = False,
    ) -> object:
        del bucket
        if prefer_gpu_download:
            return restorer.runners[module_name].run_gpu_download(inputs)
        if prefer_gpu:
            return restorer.runners[module_name].run_gpu(inputs)
        return restorer.runners[module_name].run(inputs)

    restorer._run_profiled_module = _run_profiled_module
    restorer.run_propagate_step = lambda module_name, step_inputs, **kwargs: run_propagate_step(
        restorer,
        module_name,
        step_inputs,
        **kwargs,
    )
    return restorer


class BasicvsrppVulkanRecurrentRuntimeTest(unittest.TestCase):
    def test_run_propagate_step_uses_gpu_download_without_bridge(self) -> None:
        step_runner = _FakeRunner("step")
        restorer = _make_restorer({"backward_1_step": step_runner})

        output = run_propagate_step(
            restorer,
            "backward_1",
            {"in0": np.zeros((1, 2, 2), dtype=np.float32)},
            prefer_gpu_download=True,
        )

        self.assertEqual(0, step_runner.run_calls)
        self.assertEqual(0, step_runner.gpu_calls)
        self.assertEqual(1, step_runner.gpu_download_calls)
        self.assertEqual((1, 2, 2), output.shape)

    def test_run_branch_recurrent_uses_vulkan_download_path_without_bridge(self) -> None:
        backbone_runner = _FakeRunner("backbone")
        step_runner = _FakeRunner("step")
        restorer = _make_restorer(
            {
                "backward_1_backbone": backbone_runner,
                "backward_1_step": step_runner,
            }
        )
        spatial_feats = [
            np.full((1, 2, 2), float(index), dtype=np.float32)
            for index in range(3)
        ]
        flows = [
            np.full((2, 2, 2), float(index + 1), dtype=np.float32)
            for index in range(2)
        ]

        outputs = run_branch_recurrent(
            restorer,
            "backward_1",
            spatial_feats,
            {},
            flows,
            use_gpu_bridge=False,
        )

        self.assertEqual(0, backbone_runner.run_calls)
        self.assertEqual(0, backbone_runner.gpu_calls)
        self.assertEqual(1, backbone_runner.gpu_download_calls)
        self.assertIs(backbone_runner.last_inputs["in0"], spatial_feats[2])
        self.assertTrue(
            np.array_equal(
                backbone_runner.last_inputs["in1"],
                np.zeros((1, 2, 2), dtype=np.float32),
            )
        )
        self.assertEqual(0, step_runner.run_calls)
        self.assertEqual(0, step_runner.gpu_calls)
        self.assertEqual(2, step_runner.gpu_download_calls)
        self.assertEqual(3, len(outputs))
        self.assertEqual((1, 2, 2), outputs[0].shape)

    def test_run_branch_recurrent_keeps_gpu_tensors_when_bridge_is_enabled(self) -> None:
        backbone_runner = _FakeRunner("backbone")
        step_runner = _FakeRunner("step")
        restorer = _make_restorer(
            {
                "forward_1_backbone": backbone_runner,
                "forward_1_step": step_runner,
            }
        )
        spatial_feats = [_FakeGpuBlob(f"feat_{index}") for index in range(3)]
        flows = [
            np.full((2, 2, 2), float(index + 1), dtype=np.float32)
            for index in range(2)
        ]
        branch_feats = {"backward_1": [_FakeGpuBlob(f"b1_{index}") for index in range(3)]}

        outputs = run_branch_recurrent(
            restorer,
            "forward_1",
            spatial_feats,
            branch_feats,
            flows,
            use_gpu_bridge=True,
        )

        self.assertEqual(0, backbone_runner.run_calls)
        self.assertEqual(0, backbone_runner.gpu_download_calls)
        self.assertEqual(1, backbone_runner.gpu_calls)
        self.assertEqual(0, step_runner.run_calls)
        self.assertEqual(0, step_runner.gpu_download_calls)
        self.assertEqual(2, step_runner.gpu_calls)
        self.assertEqual(3, len(outputs))
        self.assertIsInstance(outputs[0], _FakeGpuBlob)


if __name__ == "__main__":
    unittest.main()
