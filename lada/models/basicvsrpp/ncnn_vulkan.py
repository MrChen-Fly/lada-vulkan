# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import importlib
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

_NON_COMPUTE_LAYER_TYPES = frozenset({"Input", "MemoryData", "Noop"})
_CUSTOM_LAYER_SUPPORT_PROBES = {
    "torchvision.deform_conv2d": "deformconv",
    "pnnx.custom_op.torchvision.deform_conv2d": "deformconv",
    "lada.GridSample": "gridsample",
    "pnnx.custom_op.lada.GridSample": "gridsample",
    "lada.YoloAttention": "yolo_attention",
    "pnnx.custom_op.lada.YoloAttention": "yolo_attention",
    "lada.YoloSegPostprocess": "yolo_seg_postprocess",
    "pnnx.custom_op.lada.YoloSegPostprocess": "yolo_seg_postprocess",
}


def _iter_local_ncnn_runtime_dirs() -> list[Path]:
    candidates: list[Path] = []

    runtime_env = os.environ.get("LADA_LOCAL_NCNN_RUNTIME_DIR")
    if runtime_env:
        candidates.append(Path(runtime_env))

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(
            Path(meipass)
            / "native"
            / "ncnn_vulkan_runtime"
            / "build"
            / "local_runtime"
        )

    executable_dir = Path(sys.executable).resolve().parent
    candidates.append(
        executable_dir
        / "_internal"
        / "native"
        / "ncnn_vulkan_runtime"
        / "build"
        / "local_runtime"
    )
    candidates.append(
        executable_dir / "native" / "ncnn_vulkan_runtime" / "build" / "local_runtime"
    )
    candidates.append(
        Path(__file__).resolve().parents[3]
        / "native"
        / "ncnn_vulkan_runtime"
        / "build"
        / "local_runtime"
    )

    ordered_existing: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        ordered_existing.append(candidate)
    return ordered_existing


def _get_local_ncnn_runtime_dir() -> Path:
    runtime_dirs = _iter_local_ncnn_runtime_dirs()
    if runtime_dirs:
        return runtime_dirs[0]
    return (
        Path(__file__).resolve().parents[3]
        / "native"
        / "ncnn_vulkan_runtime"
        / "build"
        / "local_runtime"
    )


def import_ncnn_module(*, prefer_local_runtime: bool = True) -> Any:
    """Import the ncnn Python module, preferring the locally built runtime when available."""
    if prefer_local_runtime and "ncnn" not in sys.modules:
        for local_runtime_dir in reversed(_iter_local_ncnn_runtime_dirs()):
            runtime_dir_str = str(local_runtime_dir)
            if runtime_dir_str not in sys.path:
                sys.path.insert(0, runtime_dir_str)

    return importlib.import_module("ncnn")


def ncnn_has_lada_custom_layer(ncnn_module: Any) -> bool:
    """Return whether the ncnn module already embeds the Lada deform-conv custom layer."""
    return bool(getattr(ncnn_module, "has_lada_torchvision_deform_conv2d", False)) and callable(
        getattr(ncnn_module, "register_torchvision_deform_conv2d_layers", None)
    )


def ncnn_has_lada_gridsample_layer(ncnn_module: Any) -> bool:
    """Return whether the ncnn module already embeds the Lada GridSample custom layer."""
    register_fn = getattr(ncnn_module, "register_lada_gridsample_layers", None)
    register_all_fn = getattr(ncnn_module, "register_lada_custom_layers", None)
    return bool(getattr(ncnn_module, "has_lada_gridsample", False)) and (
        callable(register_fn) or callable(register_all_fn)
    )


def ncnn_has_lada_yolo_attention_layer(ncnn_module: Any) -> bool:
    """Return whether the ncnn module already embeds the Lada YOLO attention custom layer."""
    register_fn = getattr(ncnn_module, "register_lada_yolo_attention_layers", None)
    register_all_fn = getattr(ncnn_module, "register_lada_custom_layers", None)
    return bool(getattr(ncnn_module, "has_lada_yolo_attention", False)) and (
        callable(register_fn) or callable(register_all_fn)
    )


def ncnn_has_lada_yolo_seg_postprocess_layer(ncnn_module: Any) -> bool:
    """Return whether the ncnn module already embeds the Lada YOLO postprocess custom layer."""
    register_fn = getattr(ncnn_module, "register_lada_yolo_seg_postprocess_layers", None)
    register_all_fn = getattr(ncnn_module, "register_lada_custom_layers", None)
    return bool(getattr(ncnn_module, "has_lada_yolo_seg_postprocess_layer", False)) and (
        callable(register_fn) or callable(register_all_fn)
    )


def ncnn_has_lada_yolo_seg_postprocess_vulkan_layer(ncnn_module: Any) -> bool:
    """Return whether the YOLO postprocess custom layer also supports Vulkan compute."""
    return ncnn_has_lada_yolo_seg_postprocess_layer(ncnn_module) and bool(
        getattr(ncnn_module, "has_lada_yolo_seg_postprocess_vulkan", False)
    )


def ncnn_has_lada_vulkan_net_runner(ncnn_module: Any) -> bool:
    """Return whether the ncnn module exposes the local Vulkan blob bridge."""
    runner_type = getattr(ncnn_module, "LadaVulkanNetRunner", None)
    return bool(getattr(ncnn_module, "has_lada_vulkan_net_runner", False)) and callable(
        runner_type
    )


def ncnn_has_lada_basicvsrpp_clip_runner(ncnn_module: Any) -> bool:
    """Return whether the ncnn module exposes the native BasicVSR++ clip runner."""
    runner_type = getattr(ncnn_module, "BasicVsrppClipRunner", None)
    return bool(getattr(ncnn_module, "has_lada_basicvsrpp_clip_runner", False)) and callable(
        runner_type
    )


def is_ncnn_vulkan_tensor(ncnn_module: Any, value: Any) -> bool:
    """Return whether ``value`` is a GPU-resident tensor from the local ncnn runtime."""
    tensor_type = getattr(ncnn_module, "LadaVulkanTensor", None)
    return tensor_type is not None and isinstance(value, tensor_type)


def register_lada_custom_layers(net: Any, *, ncnn_module: Any) -> None:
    """Register built-in Lada custom layers on an ncnn.Net when the runtime supports them."""
    register_all_fn = getattr(ncnn_module, "register_lada_custom_layers", None)
    if callable(register_all_fn):
        result = int(register_all_fn(net))
        if result != 0:
            raise RuntimeError(f"Failed to register Lada custom ncnn layers (code={result}).")
        return

    if ncnn_has_lada_custom_layer(ncnn_module):
        register_fn = getattr(ncnn_module, "register_torchvision_deform_conv2d_layers")
        result = int(register_fn(net))
        if result != 0:
            raise RuntimeError(f"Failed to register Lada deform-conv custom layer (code={result}).")

    if ncnn_has_lada_gridsample_layer(ncnn_module):
        register_fn = getattr(ncnn_module, "register_lada_gridsample_layers")
        result = int(register_fn(net))
        if result != 0:
            raise RuntimeError(f"Failed to register Lada GridSample custom layer (code={result}).")

    if ncnn_has_lada_yolo_attention_layer(ncnn_module):
        register_fn = getattr(ncnn_module, "register_lada_yolo_attention_layers")
        result = int(register_fn(net))
        if result != 0:
            raise RuntimeError(
                f"Failed to register Lada YOLO attention custom layer (code={result})."
            )

    if ncnn_has_lada_yolo_seg_postprocess_layer(ncnn_module):
        register_fn = getattr(ncnn_module, "register_lada_yolo_seg_postprocess_layers")
        result = int(register_fn(net))
        if result != 0:
            raise RuntimeError(
                f"Failed to register Lada YOLO postprocess custom layer (code={result})."
            )


@dataclass(frozen=True)
class NcnnVulkanLayerAudit:
    """Summarize which ncnn layers can actually run through Vulkan."""

    total_layers: int
    layer_counts: dict[str, int]
    unsupported_layer_counts: dict[str, int]
    unsupported_compute_layer_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_layers": self.total_layers,
            "layer_counts": dict(self.layer_counts),
            "unsupported_layer_counts": dict(self.unsupported_layer_counts),
            "unsupported_compute_layer_counts": dict(self.unsupported_compute_layer_counts),
        }


def _iter_ncnn_param_layer_types(param_path: str | Path) -> list[str]:
    lines = Path(param_path).read_text(encoding="utf-8").splitlines()[2:]
    layer_types: list[str] = []
    for line in lines:
        parts = line.split()
        if parts:
            layer_types.append(parts[0])
    return layer_types


def _layer_supports_vulkan(ncnn_module: Any, layer_type: str) -> bool:
    custom_support = _CUSTOM_LAYER_SUPPORT_PROBES.get(layer_type)
    if custom_support == "deformconv" and ncnn_has_lada_custom_layer(ncnn_module):
        return True
    if custom_support == "gridsample" and ncnn_has_lada_gridsample_layer(ncnn_module):
        return True
    if custom_support == "yolo_attention" and ncnn_has_lada_yolo_attention_layer(ncnn_module):
        return True
    if (
        custom_support == "yolo_seg_postprocess"
        and ncnn_has_lada_yolo_seg_postprocess_vulkan_layer(ncnn_module)
    ):
        return True
    try:
        layer = ncnn_module.create_layer(layer_type)
    except Exception:
        return False
    if layer is None:
        return False
    return bool(layer.support_vulkan)


def audit_ncnn_vulkan_support(
    param_path: str | Path,
    *,
    ncnn_module: Any | None = None,
) -> NcnnVulkanLayerAudit:
    """Inspect an ncnn param file and report layers that will fall back off Vulkan."""
    if ncnn_module is None:
        ncnn_module = import_ncnn_module()

    layer_counts = Counter(_iter_ncnn_param_layer_types(param_path))
    unsupported_layer_counts: dict[str, int] = {}
    unsupported_compute_layer_counts: dict[str, int] = {}
    for layer_type, count in sorted(layer_counts.items()):
        if _layer_supports_vulkan(ncnn_module, layer_type):
            continue
        unsupported_layer_counts[layer_type] = count
        if layer_type not in _NON_COMPUTE_LAYER_TYPES:
            unsupported_compute_layer_counts[layer_type] = count

    return NcnnVulkanLayerAudit(
        total_layers=sum(layer_counts.values()),
        layer_counts=dict(sorted(layer_counts.items())),
        unsupported_layer_counts=unsupported_layer_counts,
        unsupported_compute_layer_counts=unsupported_compute_layer_counts,
    )


def summarize_ncnn_vulkan_audits(
    audits: Mapping[str, NcnnVulkanLayerAudit],
) -> dict[str, Any]:
    """Aggregate per-module Vulkan support audits into one serializable report."""
    total_layers = 0
    unsupported_layer_counts: Counter[str] = Counter()
    unsupported_compute_layer_counts: Counter[str] = Counter()
    module_reports: dict[str, dict[str, Any]] = {}

    for module_name, audit in audits.items():
        total_layers += audit.total_layers
        unsupported_layer_counts.update(audit.unsupported_layer_counts)
        unsupported_compute_layer_counts.update(audit.unsupported_compute_layer_counts)
        module_reports[module_name] = audit.to_dict()

    return {
        "total_layers": total_layers,
        "unsupported_layer_counts": dict(sorted(unsupported_layer_counts.items())),
        "unsupported_compute_layer_counts": dict(sorted(unsupported_compute_layer_counts.items())),
        "modules": module_reports,
    }


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
            self.net.set_vulkan_device(0)
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
        self.runner = runner_type(
            module_paths,
            fp16=fp16,
            num_threads=max(int(num_threads), 1),
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
