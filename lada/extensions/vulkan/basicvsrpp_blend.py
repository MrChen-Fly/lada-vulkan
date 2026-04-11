from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from lada.utils import ImageTensor

if TYPE_CHECKING:
    from lada.extensions.vulkan.basicvsrpp_restorer import (
        NcnnVulkanBasicvsrppMosaicRestorer,
    )


def blend_patch(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    frame_roi: ImageTensor,
    clip_img: ImageTensor,
    clip_mask: ImageTensor,
) -> ImageTensor:
    if not restorer.runtime_features.use_native_blend_patch:
        raise RuntimeError("Vulkan blend patch runtime is unavailable.")

    if restorer.runtime_features.use_native_blend_patch_inplace:
        with restorer.profiler.measure("vulkan_blend_patch_s"):
            restorer.ncnn.blend_patch_gpu_inplace(frame_roi, clip_img, clip_mask)
        return frame_roi

    frame_roi_np = np.ascontiguousarray(frame_roi.cpu().numpy())
    clip_img_np = np.ascontiguousarray(clip_img.cpu().numpy())
    clip_mask_np = np.ascontiguousarray(clip_mask.cpu().numpy())
    with restorer.profiler.measure("vulkan_blend_patch_s"):
        blended = restorer.ncnn.blend_patch_gpu(
            frame_roi_np,
            clip_img_np,
            clip_mask_np,
        )
    return torch.from_numpy(np.ascontiguousarray(blended))


def blend_patch_padded(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    frame_roi: ImageTensor,
    clip_img: ImageTensor,
    clip_mask: ImageTensor,
    pad_after_resize: tuple[int, int, int, int],
) -> ImageTensor:
    if not restorer.runtime_features.use_native_padded_blend_patch:
        raise RuntimeError("Vulkan padded blend patch runtime is unavailable.")

    pad_top, pad_bottom, pad_left, pad_right = [
        int(value) for value in pad_after_resize
    ]
    if restorer.runtime_features.use_native_padded_blend_patch_inplace:
        with restorer.profiler.measure("vulkan_blend_patch_preprocess_s"):
            restorer.ncnn.blend_patch_gpu_preprocess_inplace(
                frame_roi,
                clip_img,
                clip_mask,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
            )
        return frame_roi

    frame_roi_np = np.ascontiguousarray(frame_roi.cpu().numpy())
    clip_img_np = np.ascontiguousarray(clip_img.cpu().numpy())
    clip_mask_np = np.ascontiguousarray(clip_mask.cpu().numpy())
    with restorer.profiler.measure("vulkan_blend_patch_preprocess_s"):
        blended = restorer.ncnn.blend_patch_gpu_preprocess(
            frame_roi_np,
            clip_img_np,
            clip_mask_np,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        )
    return torch.from_numpy(np.ascontiguousarray(blended))


def blend_patch_padded_batch(
    restorer: "NcnnVulkanBasicvsrppMosaicRestorer",
    frame_rois: list[ImageTensor],
    clip_imgs: list[ImageTensor],
    clip_masks: list[ImageTensor],
    pad_after_resizes: list[tuple[int, int, int, int]],
) -> list[ImageTensor] | None:
    if not restorer.runtime_features.use_native_padded_blend_patch_batch_inplace:
        return None
    if not frame_rois:
        return []

    frame_roi_arrays = [
        np.ascontiguousarray(frame_roi.cpu().numpy()) for frame_roi in frame_rois
    ]
    clip_img_arrays = [
        np.ascontiguousarray(clip_img.cpu().numpy()) for clip_img in clip_imgs
    ]
    clip_mask_arrays = [
        np.ascontiguousarray(clip_mask.cpu().numpy()) for clip_mask in clip_masks
    ]
    paddings = [
        tuple(int(value) for value in pad_after_resize)
        for pad_after_resize in pad_after_resizes
    ]
    with restorer.profiler.measure("vulkan_blend_patch_preprocess_batch_s"):
        restorer.ncnn.blend_patch_gpu_preprocess_inplace_batch(
            frame_roi_arrays,
            clip_img_arrays,
            clip_mask_arrays,
            paddings,
        )
    for frame_roi, blended in zip(frame_rois, frame_roi_arrays, strict=True):
        frame_roi.copy_(torch.from_numpy(np.ascontiguousarray(blended)))
    return frame_rois
