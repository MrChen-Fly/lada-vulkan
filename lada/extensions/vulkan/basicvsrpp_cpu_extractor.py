from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch

from lada.extensions.vulkan.basicvsrpp_ncnn_runtime import is_ncnn_vulkan_tensor

if TYPE_CHECKING:
    from lada.extensions.vulkan.basicvsrpp_restorer import (
        NcnnVulkanBasicvsrppMosaicRestorer,
    )


def runtime_value_to_numpy(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    value: object,
) -> np.ndarray:
    """Convert one runtime tensor payload into a contiguous CHW float32 array."""
    if is_ncnn_vulkan_tensor(restorer.ncnn, value):
        value = restorer.runners["flow_warp"].download_gpu(value)

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise RuntimeError(
                    f"Vulkan runtime expects single-frame tensors, got {tuple(tensor.shape)}."
                )
            tensor = tensor.squeeze(0)
        array = tensor.numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise RuntimeError(
                    f"Vulkan runtime expects single-frame arrays, got {tuple(array.shape)}."
                )
            array = array[0]

    if array.ndim != 3:
        raise RuntimeError(f"Unexpected runtime tensor shape {tuple(array.shape)}.")
    return np.ascontiguousarray(array, dtype=np.float32)


def normalize_runtime_inputs_for_cpu_extractor(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    inputs: dict[str, object],
) -> dict[str, np.ndarray]:
    """Download runtime blobs so the plain ncnn extractor only sees CPU float inputs."""
    return {
        input_name: runtime_value_to_numpy(restorer, value)
        for input_name, value in inputs.items()
    }


def run_cpu_extractor_module(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    module_name: str,
    inputs: dict[str, object],
    *,
    bucket: str | None,
) -> np.ndarray:
    """Run one modular helper through the CPU extractor after normalizing bridge inputs."""
    normalized_inputs = normalize_runtime_inputs_for_cpu_extractor(restorer, inputs)
    with restorer.profiler.measure(bucket) if bucket is not None else nullcontext():
        output = restorer.runners[module_name].run(normalized_inputs)
    return runtime_value_to_numpy(restorer, output)
