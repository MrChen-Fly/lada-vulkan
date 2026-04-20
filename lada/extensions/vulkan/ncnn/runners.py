# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .audit import audit_ncnn_vulkan_support
from .capabilities import (
    is_ncnn_vulkan_tensor,
    ncnn_has_lada_basicvsrpp_clip_runner,
    ncnn_has_lada_vulkan_net_runner,
    register_lada_custom_layers,
)
from .device import ensure_ncnn_gpu_instance, set_ncnn_vulkan_device
from .loader import import_ncnn_module


def _to_ncnn_input_array(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    if array.ndim > 0 and array.shape[0] == 1:
        array = array.squeeze(0)

    return np.ascontiguousarray(array)


def _to_ncnn_uint8_frame_array(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise RuntimeError("NCNN Vulkan uint8 frame preprocessing expects CPU frames.")
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.ascontiguousarray(array)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(
            f"NCNN Vulkan frame preprocess expects HWC 3-channel images, got {tuple(array.shape)}."
        )
    if array.dtype != np.uint8:
        raise RuntimeError(
            f"NCNN Vulkan frame preprocess expects uint8 input, got {array.dtype}."
        )
    return array


class NcnnVulkanModuleRunner:
    """Execute exported ncnn modules through the Vulkan backend."""

    def __init__(
        self,
        param_path: str | Path,
        bin_path: str | Path,
        *,
        fp16: bool = False,
        use_vulkan: bool = True,
        num_threads: int = 1,
    ):
        self.ncnn = import_ncnn_module()
        self.param_path = Path(param_path)
        self.bin_path = Path(bin_path)
        self.layer_audit = audit_ncnn_vulkan_support(self.param_path, ncnn_module=self.ncnn)
        self.net = self.ncnn.Net()
        register_lada_custom_layers(self.net, ncnn_module=self.ncnn)
        if use_vulkan and hasattr(self.net, "set_vulkan_device"):
            set_ncnn_vulkan_device(self.net, 0, ncnn_module=self.ncnn)
        self.net.opt.use_vulkan_compute = use_vulkan
        self.net.opt.use_fp16_storage = fp16
        self.net.opt.use_fp16_packed = fp16
        self.net.opt.use_fp16_arithmetic = fp16
        self.net.opt.num_threads = max(int(num_threads), 1)
        self.gpu_runner = (
            self.ncnn.LadaVulkanNetRunner(self.net)
            if use_vulkan and ncnn_has_lada_vulkan_net_runner(self.ncnn)
            else None
        )

        if self.net.load_param(str(self.param_path)) != 0:
            raise RuntimeError(f"Failed to load ncnn param '{self.param_path}'.")
        if self.net.load_model(str(self.bin_path)) != 0:
            raise RuntimeError(f"Failed to load ncnn model '{self.bin_path}'.")

    def run(
        self,
        inputs: Mapping[str, np.ndarray | torch.Tensor],
        *,
        output_name: str = "out0",
    ) -> np.ndarray:
        extractor = self.net.create_extractor()
        for input_name, value in inputs.items():
            input_mat = self.ncnn.Mat(_to_ncnn_input_array(value))
            if extractor.input(input_name, input_mat) != 0:
                raise RuntimeError(f"Failed to feed '{input_name}' into ncnn extractor.")

        output_code, output_mat = extractor.extract(output_name)
        if output_code != 0:
            raise RuntimeError(f"Failed to extract '{output_name}' from ncnn extractor.")
        return np.array(output_mat)

    def preprocess_bgr_u8_frame(
        self,
        image: np.ndarray | torch.Tensor,
        *,
        input_shape: tuple[int, int] | None = None,
    ) -> Any:
        """Upload one CPU HWC uint8 BGR frame through the native Vulkan preprocess path."""
        if self.gpu_runner is None or not hasattr(self.gpu_runner, "preprocess_bgr_u8"):
            raise RuntimeError("GPU frame preprocess bridge is unavailable for this ncnn runtime.")

        image_np = _to_ncnn_uint8_frame_array(image)
        if input_shape is None:
            input_shape = (int(image_np.shape[0]), int(image_np.shape[1]))
        return self.gpu_runner.preprocess_bgr_u8(
            image_np,
            [int(input_shape[0]), int(input_shape[1])],
        )

    def preprocess_bgr_u8_frames(
        self,
        images: list[np.ndarray | torch.Tensor] | tuple[np.ndarray | torch.Tensor, ...],
        *,
        input_shape: tuple[int, int] | None = None,
    ) -> list[Any]:
        """Upload a batch of CPU HWC uint8 BGR frames through the native Vulkan preprocess path."""
        if self.gpu_runner is None or not hasattr(self.gpu_runner, "preprocess_bgr_u8_batch"):
            raise RuntimeError("GPU batch frame preprocess bridge is unavailable for this ncnn runtime.")
        if not images:
            return []

        image_arrays = [_to_ncnn_uint8_frame_array(image) for image in images]
        if input_shape is None:
            input_shape = (
                int(image_arrays[0].shape[0]),
                int(image_arrays[0].shape[1]),
            )
        return list(
            self.gpu_runner.preprocess_bgr_u8_batch(
                image_arrays,
                [int(input_shape[0]), int(input_shape[1])],
            )
        )

    def run_gpu(
        self,
        inputs: Mapping[str, np.ndarray | torch.Tensor | Any],
        *,
        output_name: str = "out0",
    ) -> Any:
        if self.gpu_runner is None:
            raise RuntimeError("GPU blob bridge is unavailable for this ncnn runtime.")

        converted_inputs: dict[str, Any] = {}
        for input_name, value in inputs.items():
            if is_ncnn_vulkan_tensor(self.ncnn, value):
                converted_inputs[input_name] = value
            else:
                converted_inputs[input_name] = self.ncnn.Mat(_to_ncnn_input_array(value))
        return self.gpu_runner.run(converted_inputs, output_name)

    def run_many_gpu(
        self,
        inputs: Mapping[str, np.ndarray | torch.Tensor | Any],
        *,
        output_names: list[str] | tuple[str, ...],
    ) -> dict[str, Any]:
        if self.gpu_runner is None or not hasattr(self.gpu_runner, "run_many"):
            raise RuntimeError("GPU multi-output blob bridge is unavailable for this ncnn runtime.")

        converted_inputs: dict[str, Any] = {}
        for input_name, value in inputs.items():
            if is_ncnn_vulkan_tensor(self.ncnn, value):
                converted_inputs[input_name] = value
            else:
                converted_inputs[input_name] = self.ncnn.Mat(_to_ncnn_input_array(value))
        return dict(self.gpu_runner.run_many(converted_inputs, list(output_names)))

    def run_gpu_download(
        self,
        inputs: Mapping[str, np.ndarray | torch.Tensor | Any],
        *,
        output_name: str = "out0",
    ) -> np.ndarray:
        if self.gpu_runner is None:
            raise RuntimeError("GPU blob bridge is unavailable for this ncnn runtime.")

        converted_inputs: dict[str, Any] = {}
        for input_name, value in inputs.items():
            if is_ncnn_vulkan_tensor(self.ncnn, value):
                converted_inputs[input_name] = value
            else:
                converted_inputs[input_name] = self.ncnn.Mat(_to_ncnn_input_array(value))
        return np.array(self.gpu_runner.run_to_cpu(converted_inputs, output_name))

    def download_gpu(self, value: Any) -> np.ndarray:
        if is_ncnn_vulkan_tensor(self.ncnn, value):
            return np.array(value.download())
        return _to_ncnn_input_array(value)


class NcnnVulkanBasicvsrppClipRunner:
    """Execute the recurrent BasicVSR++ clip path inside the local ncnn runtime."""

    def __init__(
        self,
        module_artifacts: Mapping[str, Any],
        *,
        fp16: bool = False,
        num_threads: int = 1,
        spynet_patch_shape: tuple[int, int] | None = None,
        spynet_core_shape: tuple[int, int] | None = None,
    ):
        self.ncnn = import_ncnn_module()
        if not ncnn_has_lada_basicvsrpp_clip_runner(self.ncnn):
            raise RuntimeError("Native BasicVSR++ clip runner is unavailable for this ncnn runtime.")

        runner_type = getattr(self.ncnn, "BasicVsrppClipRunner", None)
        if runner_type is None:
            raise RuntimeError("Native BasicVSR++ clip runner type is missing from the ncnn runtime.")

        module_paths = {
            module_name: (
                str(artifacts.param_path),
                str(artifacts.bin_path),
            )
            for module_name, artifacts in module_artifacts.items()
        }
        ensure_ncnn_gpu_instance(self.ncnn)
        patch_shape = (
            [int(spynet_patch_shape[0]), int(spynet_patch_shape[1])]
            if spynet_patch_shape is not None
            else []
        )
        core_shape = (
            [int(spynet_core_shape[0]), int(spynet_core_shape[1])]
            if spynet_core_shape is not None
            else []
        )
        runner_kwargs = {
            "fp16": fp16,
            "num_threads": max(int(num_threads), 1),
        }
        if patch_shape:
            runner_kwargs["spynet_patch_shape"] = patch_shape
        if core_shape:
            runner_kwargs["spynet_core_shape"] = core_shape
        try:
            self.runner = runner_type(
                module_paths,
                **runner_kwargs,
            )
        except TypeError as exc:
            if (
                ("spynet_patch_shape" not in runner_kwargs and "spynet_core_shape" not in runner_kwargs)
                or (
                    "incompatible constructor arguments" not in str(exc)
                    and "unexpected keyword argument" not in str(exc)
                )
            ):
                raise
            legacy_runner_kwargs = {
                "fp16": runner_kwargs["fp16"],
                "num_threads": runner_kwargs["num_threads"],
            }
            self.runner = runner_type(
                module_paths,
                **legacy_runner_kwargs,
            )
        self.supports_bgr_u8_input = callable(getattr(self.runner, "restore_bgr_u8", None))
        self.supports_resized_bgr_u8_input = callable(
            getattr(self.runner, "restore_bgr_u8_resized", None)
        )
        self._supports_last_profile = callable(getattr(self.runner, "get_last_profile", None))

    def restore(self, lqs: list[np.ndarray | torch.Tensor]) -> list[np.ndarray]:
        return [
            np.asarray(output)
            for output in self.runner.restore([_to_ncnn_input_array(value) for value in lqs])
        ]

    def restore_bgr_u8(self, frames: list[np.ndarray | torch.Tensor]) -> list[np.ndarray]:
        if not self.supports_bgr_u8_input:
            raise RuntimeError("Native BasicVSR++ clip runner does not support uint8 frame inputs.")
        return [
            np.asarray(output)
            for output in self.runner.restore_bgr_u8(
                [_to_ncnn_uint8_frame_array(value) for value in frames]
            )
        ]

    def restore_bgr_u8_resized(
        self,
        frames: list[np.ndarray | torch.Tensor],
        *,
        target_size: int,
        resize_reference_shape: tuple[int, int],
        pad_mode: str,
    ) -> list[np.ndarray]:
        if not self.supports_resized_bgr_u8_input:
            raise RuntimeError(
                "Native BasicVSR++ clip runner does not support resized uint8 frame inputs."
            )
        return [
            np.asarray(output)
            for output in self.runner.restore_bgr_u8_resized(
                [_to_ncnn_uint8_frame_array(value) for value in frames],
                int(target_size),
                [int(resize_reference_shape[0]), int(resize_reference_shape[1])],
                str(pad_mode),
            )
        ]

    def get_last_profile(self) -> dict[str, float]:
        if not self._supports_last_profile:
            return {}
        return {
            str(key): float(value)
            for key, value in dict(self.runner.get_last_profile()).items()
        }

    def debug_trace(self, lqs: list[np.ndarray | torch.Tensor]) -> dict[str, list[np.ndarray]]:
        trace_fn = getattr(self.runner, "debug_trace", None)
        if not callable(trace_fn):
            raise RuntimeError("Native BasicVSR++ clip runner debug trace is unavailable.")
        raw_trace = dict(trace_fn([_to_ncnn_input_array(value) for value in lqs]))
        return {
            str(stage_name): [np.asarray(output) for output in stage_outputs]
            for stage_name, stage_outputs in raw_trace.items()
        }

    def debug_trace_bgr_u8_resized(
        self,
        frames: list[np.ndarray | torch.Tensor],
        *,
        target_size: int,
        resize_reference_shape: tuple[int, int],
        pad_mode: str,
    ) -> dict[str, list[np.ndarray]]:
        trace_fn = getattr(self.runner, "debug_trace_bgr_u8_resized", None)
        if not callable(trace_fn):
            raise RuntimeError(
                "Native BasicVSR++ clip runner resized uint8 debug trace is unavailable."
            )
        raw_trace = dict(
            trace_fn(
                [_to_ncnn_uint8_frame_array(value) for value in frames],
                int(target_size),
                [int(resize_reference_shape[0]), int(resize_reference_shape[1])],
                str(pad_mode),
            )
        )
        return {
            str(stage_name): [np.asarray(output) for output in stage_outputs]
            for stage_name, stage_outputs in raw_trace.items()
        }
