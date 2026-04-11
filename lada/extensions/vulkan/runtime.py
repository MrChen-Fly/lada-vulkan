from __future__ import annotations

import ctypes
import ctypes.util
import sys


def _has_vulkan_loader() -> bool:
    probe_name = "vulkan-1" if sys.platform == "win32" else "vulkan"
    if ctypes.util.find_library(probe_name) is not None:
        return True

    load_name = "vulkan-1.dll" if sys.platform == "win32" else "libvulkan.so.1"
    loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
    try:
        loader(load_name)
        return True
    except OSError:
        return False


def _has_ncnn_vulkan_runtime() -> bool:
    try:
        from lada.extensions.vulkan.basicvsrpp_ncnn_runtime import (
            get_ncnn_gpu_count,
            import_ncnn_module,
            ncnn_has_lada_basicvsrpp_clip_runner,
            ncnn_has_lada_custom_layer,
            ncnn_has_lada_gridsample_layer,
            ncnn_has_lada_vulkan_net_runner,
            ncnn_has_lada_yolo_attention_layer,
            ncnn_has_lada_yolo_seg_postprocess_vulkan_layer,
        )

        ncnn = import_ncnn_module()
        return (
            get_ncnn_gpu_count(ncnn) > 0
            and ncnn_has_lada_custom_layer(ncnn)
            and ncnn_has_lada_gridsample_layer(ncnn)
            and ncnn_has_lada_yolo_attention_layer(ncnn)
            and ncnn_has_lada_yolo_seg_postprocess_vulkan_layer(ncnn)
            and ncnn_has_lada_vulkan_net_runner(ncnn)
            and ncnn_has_lada_basicvsrpp_clip_runner(ncnn)
        )
    except Exception:
        return False


def get_vulkan_compute_targets() -> list[object]:
    from lada.compute_targets import ComputeTarget

    has_loader = _has_vulkan_loader()
    has_ncnn_runtime = _has_ncnn_vulkan_runtime()
    loader_note = "" if has_loader else " Vulkan loader not detected on this machine."
    ncnn_note = (
        ""
        if has_ncnn_runtime
        else (
            " Build the local ncnn Vulkan runtime with Lada custom operators "
            "(for example via 'scripts/build_ncnn_vulkan_runtime.ps1')."
        )
    )
    return [
        ComputeTarget(
            id="vulkan:0",
            description="Vulkan",
            runtime="vulkan",
            available=has_loader and has_ncnn_runtime,
            torch_device=None,
            notes=(
                "Vulkan backend backed by the local ncnn runtime and Lada custom operators. "
                "Mosaic detection runs through fused YOLO segmentation postprocess on GPU, and "
                "BasicVSR++ restoration runs through the modular 5-frame / recurrent Vulkan graph. "
                f"DeepMosaics is not supported.{loader_note}{ncnn_note}"
            ),
            experimental=True,
        )
    ]


def default_vulkan_fp16_enabled(_target_id: str) -> bool:
    # The local NCNN Vulkan path is designed around fp16 artifacts; callers can still opt out
    # with `--no-fp16` when they need conservative compatibility.
    return True


def configure_vulkan_device_info(show: bool | None = None) -> bool | None:
    from lada.extensions.vulkan.basicvsrpp_ncnn_runtime import (
        configure_ncnn_vulkan_device_info,
    )

    return configure_ncnn_vulkan_device_info(show=show)
