# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lada.utils import Box, Mask, box_utils
from lada.utils import image_utils

def get_box(mask: Mask) -> Box:
    points = cv2.findNonZero(mask)
    return box_utils.convert_from_opencv(cv2.boundingRect(points))

def morph(mask: Mask, iterations=1, operator=cv2.MORPH_DILATE) -> Mask:
    if get_mask_area(mask) < 0.01:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, operator, kernel, iterations=iterations)

def dilate_mask(mask: Mask, dilatation_size=11, iterations=2):
    if iterations == 0:
        return mask
    element = np.ones((dilatation_size, dilatation_size), np.uint8)
    mask_img = cv2.dilate(mask, element, iterations=iterations).reshape(mask.shape)
    return mask_img

def extend_mask(mask: Mask, value) -> Mask:
    # value between 0 and 3 -> higher values mean more extension of mask area. 0 does not change mask at all
    if value == 0:
        return mask

    # Dilations are slow when using huge kernels (which we would need for high-res masks). therefore we downscale mask to perform morph operations on much smaller pixel space with smaller kernels
    target_size = 256
    extended_mask = image_utils.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    extended_mask = morph(extended_mask, iterations=value, operator=cv2.MORPH_DILATE)
    extended_mask = image_utils.resize(extended_mask, mask.shape[:2], interpolation=cv2.INTER_NEAREST)
    extended_mask = extended_mask.reshape(mask.shape)
    assert mask.shape == extended_mask.shape
    return extended_mask

def clean_mask(mask: Mask, box: Box) -> tuple[Mask, Box]:
    t, l, b, r = box
    # Masks from YOLO prediction extend detection area in some cases. Let's crop
    mask[:t + 1, :, :] = 0
    mask[b:, :, :] = 0
    mask[:, :l + 1, :] = 0
    mask[:, r:, :] = 0

    # Mask from YOLO prediction can sometimes contain additional disconnected (tiny) segments. Keep only the largest
    edited_mask = np.zeros_like(mask, dtype=mask.dtype)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) != 0
    if len(contours) > 1:
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[0]
    largest_contour = contours[0]
    cv_box = cv2.boundingRect(largest_contour)
    box = box_utils.convert_from_opencv(cv_box)
    cv2.drawContours(edited_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
    return edited_mask, box

def get_mask_area(mask: Mask) -> float:
    pixels = cv2.countNonZero(mask)
    return pixels / (mask.shape[0] * mask.shape[1])

def smooth_mask(mask: Mask, kernel_size: int) -> Mask:
    return cv2.medianBlur(mask, kernel_size).reshape(mask.shape)

_BLEND_MASK_BOX_BORDER_RATIO = 0.05


def _resolve_blend_box(mask: torch.Tensor) -> tuple[int, int, int, int, int]:
    nonzero = torch.nonzero(mask > 0, as_tuple=False)
    if nonzero.numel() == 0:
        return 0, 0, 0, 0, 0

    top = int(nonzero[:, 0].min().item())
    bottom = int(nonzero[:, 0].max().item()) + 1
    left = int(nonzero[:, 1].min().item())
    right = int(nonzero[:, 1].max().item()) + 1

    box_height = max(bottom - top, 1)
    box_width = max(right - left, 1)
    border_y = max(1, int(round(box_height * _BLEND_MASK_BOX_BORDER_RATIO)))
    border_x = max(1, int(round(box_width * _BLEND_MASK_BOX_BORDER_RATIO)))

    inner_top = max(0, top - border_y)
    inner_bottom = min(mask.shape[0], bottom + border_y)
    inner_left = max(0, left - border_x)
    inner_right = min(mask.shape[1], right + border_x)

    border_height = (top - inner_top) + (inner_bottom - bottom)
    border_width = (left - inner_left) + (inner_right - right)
    border_size = min(border_height, border_width)
    return inner_top, inner_bottom, inner_left, inner_right, border_size


def _create_blend_mask_cpu(mask: torch.Tensor, blur_size: int, inner_top: int, inner_bottom: int, inner_left: int, inner_right: int) -> torch.Tensor:
    mask_np = (mask.numpy() > 0).astype(np.float32)
    blend = np.zeros(mask.shape, dtype=np.float32)
    blend[inner_top:inner_bottom, inner_left:inner_right] = 1.0
    blend = np.maximum(mask_np, blend)
    if blur_size >= 5:
        blend = cv2.blur(blend, (blur_size, blur_size), borderType=cv2.BORDER_REFLECT)
    return torch.from_numpy(blend).to(dtype=mask.dtype)

def create_blend_mask(crop_mask: torch.Tensor):
    mask = crop_mask.squeeze()
    inner_top, inner_bottom, inner_left, inner_right, border_size = _resolve_blend_box(mask)
    if border_size == 0:
        return torch.zeros_like(mask)
    blur_size = int(border_size)
    if blur_size % 2 == 0:
        blur_size += 1
    mask4 = (mask > 0).to(dtype=mask.dtype)
    if mask.device.type == 'cpu':
        blend = _create_blend_mask_cpu(
            mask,
            blur_size,
            inner_top,
            inner_bottom,
            inner_left,
            inner_right,
        )
        assert blend.shape == mask.shape
        return blend
    blend = torch.zeros_like(mask, dtype=mask.dtype)
    blend[inner_top:inner_bottom, inner_left:inner_right] = 1.0
    blend = torch.maximum(mask4, blend)
    if blur_size >= 5:
        kernel = torch.tensor(1.0 / (blur_size**2), device=blend.device, dtype=blend.dtype).expand(1, blur_size, blur_size)
        blend = image_utils.filter2D(blend.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    assert blend.shape == mask.shape
    return blend

def apply_random_mask_extensions(mask: Mask) -> Mask:
    value = np.random.choice([0, 0, 1, 1, 2])
    return extend_mask(mask, value)

def box_to_mask(box: Box, shape, mask_value: int):
    mask = np.zeros((shape[0], shape[1], 1), np.uint8)
    t, l, b, r = box
    mask[t:b + 1, l:r + 1] = mask_value
    return mask
