from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from lada.restorationpipeline.basicvsrpp_vulkan_restore_paths import (
    _resize_cropped_frames_for_runtime,
    preprocess_clip_frames_to_runtime_inputs,
    restore_clip_recurrent_native_resized,
)
from lada.restorationpipeline.clip_units import Clip, ClipDescriptor


class _FakeProfiler:
    def __init__(self) -> None:
        self.duration_calls: list[tuple[str, float]] = []
        self.measure_calls: list[str] = []

    def add_duration(self, bucket: str, duration: float) -> None:
        self.duration_calls.append((bucket, duration))

    def measure(self, bucket: str):
        self.measure_calls.append(bucket)
        return nullcontext()


class _FakeNativeClipRunner:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []
        self.supports_resized_bgr_u8_input = True

    def restore_bgr_u8_resized(
        self,
        frames: list[np.ndarray],
        *,
        target_size: int,
        resize_reference_shape: tuple[int, int],
        pad_mode: str,
    ) -> list[np.ndarray]:
        self.calls.append(
            {
                "frame_count": len(frames),
                "frame_shapes": [tuple(frame.shape) for frame in frames],
                "target_size": target_size,
                "resize_reference_shape": tuple(resize_reference_shape),
                "pad_mode": pad_mode,
            }
        )
        return list(self.outputs)

    def get_last_profile(self) -> dict[str, float]:
        return {"input_preprocess_s": 0.25, "output_frame_s": 0.5}


class _FakePreprocessRunner:
    def __init__(self) -> None:
        self.single_calls: list[tuple[tuple[int, int, 3], tuple[int, int]]] = []
        self.batch_calls: list[tuple[int, tuple[int, int]]] = []

    def preprocess_bgr_u8_frame(
        self,
        frame: np.ndarray,
        *,
        input_shape: tuple[int, int],
    ) -> str:
        self.single_calls.append((tuple(frame.shape), input_shape))
        return f"single:{len(self.single_calls)}"

    def preprocess_bgr_u8_frames(
        self,
        frames: list[np.ndarray],
        *,
        input_shape: tuple[int, int],
    ) -> list[str]:
        self.batch_calls.append((len(frames), input_shape))
        return [f"batch:{index}" for index, _ in enumerate(frames)]


class _CloneableTensor:
    def __init__(self, label: str) -> None:
        self.label = label
        self.clone_calls = 0

    def clone(self) -> str:
        self.clone_calls += 1
        return f"{self.label}:clone:{self.clone_calls}"


class _CloneablePreprocessRunner(_FakePreprocessRunner):
    def preprocess_bgr_u8_frame(
        self,
        frame: np.ndarray,
        *,
        input_shape: tuple[int, int],
    ) -> _CloneableTensor:
        self.single_calls.append((tuple(frame.shape), input_shape))
        return _CloneableTensor(f"single:{len(self.single_calls)}")

    def preprocess_bgr_u8_frames(
        self,
        frames: list[np.ndarray],
        *,
        input_shape: tuple[int, int],
    ) -> list[_CloneableTensor]:
        self.batch_calls.append((len(frames), input_shape))
        return [_CloneableTensor(f"batch:{index}") for index, _ in enumerate(frames)]


def _make_restorer(*, batch_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        profiler=_FakeProfiler(),
        frame_preprocess_runner=_FakePreprocessRunner(),
        runtime_features=SimpleNamespace(
            use_native_frame_preprocess=True,
            use_native_frame_preprocess_batch=batch_enabled,
        ),
    )


class BasicvsrppVulkanRestorePathsTest(unittest.TestCase):
    def test_preprocess_clip_frames_uses_native_single_frame_preprocess(self) -> None:
        restorer = _make_restorer(batch_enabled=False)
        frames = [np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6, 3), dtype=np.uint8)]

        outputs = preprocess_clip_frames_to_runtime_inputs(
            restorer,
            frames,
            prefer_gpu_bridge=True,
        )

        self.assertEqual(["single:1", "single:2"], outputs)
        self.assertEqual(
            [((4, 6, 3), (4, 6)), ((4, 6, 3), (4, 6))],
            restorer.frame_preprocess_runner.single_calls,
        )
        self.assertEqual([], restorer.frame_preprocess_runner.batch_calls)
        self.assertEqual(["vulkan_frame_preprocess_s"], restorer.profiler.measure_calls)
        self.assertEqual([], restorer.profiler.duration_calls)

    def test_preprocess_clip_frames_clones_single_native_outputs_when_supported(self) -> None:
        restorer = _make_restorer(batch_enabled=False)
        restorer.frame_preprocess_runner = _CloneablePreprocessRunner()
        frames = [np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6, 3), dtype=np.uint8)]

        outputs = preprocess_clip_frames_to_runtime_inputs(
            restorer,
            frames,
            prefer_gpu_bridge=True,
        )

        self.assertEqual(["single:1:clone:1", "single:2:clone:1"], outputs)
        self.assertEqual(
            [((4, 6, 3), (4, 6)), ((4, 6, 3), (4, 6))],
            restorer.frame_preprocess_runner.single_calls,
        )

    def test_preprocess_clip_frames_prefers_native_batch_preprocess(self) -> None:
        restorer = _make_restorer(batch_enabled=True)
        frames = [np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6, 3), dtype=np.uint8)]

        outputs = preprocess_clip_frames_to_runtime_inputs(
            restorer,
            frames,
            prefer_gpu_bridge=True,
        )

        self.assertEqual(["batch:0", "batch:1"], outputs)
        self.assertEqual([], restorer.frame_preprocess_runner.single_calls)
        self.assertEqual([(2, (4, 6))], restorer.frame_preprocess_runner.batch_calls)
        self.assertEqual(["vulkan_frame_preprocess_s"], restorer.profiler.measure_calls)
        self.assertEqual([], restorer.profiler.duration_calls)

    def test_preprocess_clip_frames_clones_batch_native_outputs_when_supported(self) -> None:
        restorer = _make_restorer(batch_enabled=True)
        restorer.frame_preprocess_runner = _CloneablePreprocessRunner()
        frames = [np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6, 3), dtype=np.uint8)]

        outputs = preprocess_clip_frames_to_runtime_inputs(
            restorer,
            frames,
            prefer_gpu_bridge=True,
        )

        self.assertEqual(["batch:0:clone:1", "batch:1:clone:1"], outputs)
        self.assertEqual([], restorer.frame_preprocess_runner.single_calls)
        self.assertEqual([(2, (4, 6))], restorer.frame_preprocess_runner.batch_calls)

    def test_restore_clip_recurrent_native_resized_uses_native_resized_entry(self) -> None:
        native_outputs = [
            np.full((3, 2, 2), fill_value=32.0 / 255.0, dtype=np.float32),
            np.full((3, 2, 2), fill_value=96.0 / 255.0, dtype=np.float32),
        ]
        runner = _FakeNativeClipRunner(native_outputs)
        restorer = SimpleNamespace(
            profiler=_FakeProfiler(),
            native_clip_runner=runner,
            frame_count=5,
            _native_clip_profile_snapshot={},
        )
        frames = [np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6, 3), dtype=np.uint8)]

        outputs = restore_clip_recurrent_native_resized(
            restorer,
            frames,
            target_size=256,
            resize_reference_shape=(100, 80),
            pad_mode="reflect",
        )

        self.assertEqual(1, len(runner.calls))
        self.assertEqual(2, runner.calls[0]["frame_count"])
        self.assertEqual((100, 80), runner.calls[0]["resize_reference_shape"])
        self.assertEqual("reflect", runner.calls[0]["pad_mode"])
        self.assertEqual(
            [32, 96],
            [int(torch.unique(output).item()) for output in outputs],
        )
        self.assertEqual(
            ["vulkan_recurrent_clip_native_s", "cpu_output_postprocess_s"],
            restorer.profiler.measure_calls,
        )
        self.assertIn(("vulkan_native_input_preprocess_s", 0.25), restorer.profiler.duration_calls)
        self.assertIn(("vulkan_native_output_frame_s", 0.5), restorer.profiler.duration_calls)

    def test_restore_clip_recurrent_native_resized_replicates_single_frame_window(self) -> None:
        native_outputs = [
            np.full((3, 2, 2), fill_value=index / 255.0, dtype=np.float32)
            for index in range(5)
        ]
        runner = _FakeNativeClipRunner(native_outputs)
        restorer = SimpleNamespace(
            profiler=_FakeProfiler(),
            native_clip_runner=runner,
            frame_count=5,
            _native_clip_profile_snapshot={},
        )
        frame = np.full((4, 6, 3), fill_value=7, dtype=np.uint8)

        outputs = restore_clip_recurrent_native_resized(
            restorer,
            [frame],
            target_size=256,
            resize_reference_shape=(100, 80),
            pad_mode="reflect",
        )

        self.assertEqual(1, len(runner.calls))
        self.assertEqual(5, runner.calls[0]["frame_count"])
        self.assertEqual([(4, 6, 3)] * 5, runner.calls[0]["frame_shapes"])
        self.assertEqual(1, len(outputs))
        self.assertEqual(2, int(torch.unique(outputs[0]).item()))

    def test_torch_bilinear_descriptor_materialization_matches_runtime_resize_helper(self) -> None:
        frame = np.array(
            [
                [[0, 10, 20], [30, 40, 50], [60, 70, 80]],
                [[90, 100, 110], [120, 130, 140], [150, 160, 170]],
            ],
            dtype=np.uint8,
        )
        descriptor = ClipDescriptor(
            file_path="synthetic.webm",
            frame_start=0,
            size=6,
            pad_mode="reflect",
            id="clip",
            frames=[frame],
            masks=[np.zeros((2, 3), dtype=np.uint8)],
            boxes=[(0, 0, 1, 2)],
            crop_boxes=[(0, 0, 1, 2)],
            resize_reference_shape=(3, 2),
        )

        clip = Clip.from_descriptor(descriptor, resize_mode="torch_bilinear")
        runtime_frames = _resize_cropped_frames_for_runtime(
            [frame],
            size=descriptor.size,
            resize_reference_shape=descriptor.resize_reference_shape,
            pad_mode=descriptor.pad_mode,
        )

        self.assertEqual(np.float32, clip.frames[0].dtype)
        self.assertEqual(np.float32, runtime_frames[0].dtype)
        np.testing.assert_allclose(clip.frames[0], runtime_frames[0], atol=1e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
