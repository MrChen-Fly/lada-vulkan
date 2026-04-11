import unittest
from contextlib import nullcontext
from unittest import mock

import numpy as np

from lada.restorationpipeline import basicvsrpp_vulkan_cpu_extractor as extractor


class _FakeProfiler:
    def measure(self, _bucket):
        return nullcontext()


class _FakeFlowWarpRunner:
    def __init__(self, downloaded):
        self.downloaded = downloaded

    def download_gpu(self, _value):
        return self.downloaded


class _CapturingRunner:
    def __init__(self, output):
        self.output = output
        self.last_inputs = None

    def run(self, inputs):
        self.last_inputs = inputs
        return self.output


class _FakeRestorer:
    def __init__(self, flow_warp_runner, module_runner):
        self.ncnn = object()
        self.profiler = _FakeProfiler()
        self.runners = {
            "flow_warp": flow_warp_runner,
            "module": module_runner,
        }


class _FakeVulkanTensor:
    pass


class RunCpuExtractorModuleTests(unittest.TestCase):
    def test_downloads_gpu_blob_inputs_before_runner_call(self):
        gpu_value = _FakeVulkanTensor()
        downloaded = np.ones((64, 8, 8), dtype=np.float32)
        module_output = np.full((64, 8, 8), 2.0, dtype=np.float32)
        flow_warp_runner = _FakeFlowWarpRunner(downloaded)
        module_runner = _CapturingRunner(module_output)
        restorer = _FakeRestorer(flow_warp_runner, module_runner)
        cpu_value = np.zeros((64, 8, 8), dtype=np.float32)

        with mock.patch.object(
            extractor,
            "is_ncnn_vulkan_tensor",
            side_effect=lambda _ncnn, value: isinstance(value, _FakeVulkanTensor),
        ):
            output = extractor.run_cpu_extractor_module(
                restorer,
                "module",
                {"in0": gpu_value, "in1": cpu_value},
                bucket=None,
            )

        self.assertIsNotNone(module_runner.last_inputs)
        self.assertIsInstance(module_runner.last_inputs["in0"], np.ndarray)
        self.assertEqual(module_runner.last_inputs["in0"].dtype, np.float32)
        np.testing.assert_allclose(module_runner.last_inputs["in0"], downloaded)
        self.assertIsInstance(module_runner.last_inputs["in1"], np.ndarray)
        self.assertEqual(module_runner.last_inputs["in1"].dtype, np.float32)
        np.testing.assert_allclose(module_runner.last_inputs["in1"], cpu_value)
        np.testing.assert_allclose(output, module_output)


if __name__ == "__main__":
    unittest.main()
