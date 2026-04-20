# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
from ultralytics.engine.results import Results as UltralyticsResults


DetectionResult: TypeAlias = UltralyticsResults


def build_native_yolo_result(
    *,
    orig_img,
    names: dict[int, str],
    boxes: np.ndarray,
    masks: np.ndarray,
) -> UltralyticsResults:
    box_tensor = torch.from_numpy(
        np.ascontiguousarray(boxes, dtype=np.float32).reshape(-1, 6)
    )
    mask_tensor = None
    mask_array = np.ascontiguousarray(masks, dtype=np.uint8)
    if mask_array.size > 0:
        if mask_array.ndim == 2:
            mask_array = mask_array[None, ...]
        mask_tensor = torch.from_numpy(mask_array)
    return UltralyticsResults(
        orig_img,
        path="",
        names=names,
        boxes=box_tensor,
        masks=mask_tensor,
    )
