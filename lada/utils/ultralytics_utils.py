# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
import os.path
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import settings
from ultralytics.engine.results import Boxes as UltralyticsBoxes
from ultralytics.engine.results import Masks as UltralyticsMasks
from ultralytics.engine.results import Results as UltralyticsResults
from ultralytics.utils import JSONDict

from lada.utils import Box, Image, Mask, MaskTensor


@dataclass(slots=True)
class NativeYoloBox:
    box: Box
    conf_value: float
    cls_value: float
    id: torch.Tensor | None = None

    @property
    def xyxy(self) -> torch.Tensor:
        t, l, b, r = self.box
        return torch.tensor([[float(l), float(t), float(r), float(b)]], dtype=torch.float32)

    @property
    def conf(self) -> torch.Tensor:
        return torch.tensor([self.conf_value], dtype=torch.float32)

    @property
    def cls(self) -> torch.Tensor:
        return torch.tensor([self.cls_value], dtype=torch.float32)

    @property
    def xywh(self) -> torch.Tensor:
        t, l, b, r = self.box
        width = float(r - l)
        height = float(b - t)
        center_x = float(l) + width * 0.5
        center_y = float(t) + height * 0.5
        return torch.tensor([[center_x, center_y, width, height]], dtype=torch.float32)


@dataclass(slots=True)
class NativeYoloMask:
    data: Mask | MaskTensor


@dataclass(slots=True)
class NativeYoloResult:
    orig_img: Image | torch.Tensor
    orig_shape: tuple[int, int]
    names: dict[int, str]
    boxes: list[NativeYoloBox]
    masks: list[NativeYoloMask]


DetectionResult: TypeAlias = UltralyticsResults | NativeYoloResult

def set_default_settings():
    settings.update(dict(
        runs_dir=os.path.join('.', 'experiments', 'yolo'),
        datasets_dir = os.path.join('.', 'datasets'),
        weights_dir = os.path.join('.', 'model_weights', '3rd_party'),
        tensorboard = True,
    ))

def get_settings() -> JSONDict:
    return settings


def build_native_yolo_result(
    *,
    orig_img: Image | torch.Tensor,
    names: dict[int, str],
    boxes: np.ndarray,
    masks: np.ndarray,
) -> NativeYoloResult:
    native_boxes: list[NativeYoloBox] = []
    native_masks: list[NativeYoloMask] = []
    img_h, img_w = tuple(int(v) for v in orig_img.shape[:2])

    for row, mask in zip(boxes, masks):
        l = int(np.clip(row[0], 0, img_w))
        t = int(np.clip(row[1], 0, img_h))
        r = int(np.clip(row[2], 0, img_w))
        b = int(np.clip(row[3], 0, img_h))
        native_boxes.append(
            NativeYoloBox(
                box=(t, l, b, r),
                conf_value=float(row[4]),
                cls_value=float(row[5]),
            )
        )

        native_masks.append(NativeYoloMask(data=_ensure_mask_image(mask)))

    return NativeYoloResult(
        orig_img=orig_img,
        orig_shape=tuple(orig_img.shape[:2]),
        names=names,
        boxes=native_boxes,
        masks=native_masks,
    )


def _extract_box_xyxy(yolo_box: UltralyticsBoxes | NativeYoloBox) -> torch.Tensor:
    if isinstance(yolo_box, NativeYoloBox):
        return yolo_box.xyxy[0]
    return yolo_box.xyxy[0]


def convert_yolo_box(yolo_box: UltralyticsBoxes | NativeYoloBox, img_shape) -> Box:
    if isinstance(yolo_box, NativeYoloBox):
        return yolo_box.box
    _box = _extract_box_xyxy(yolo_box)
    l = int(torch.clip(_box[0], 0, img_shape[1]).item())
    t = int(torch.clip(_box[1], 0, img_shape[0]).item())
    r = int(torch.clip(_box[2], 0, img_shape[1]).item())
    b = int(torch.clip(_box[3], 0, img_shape[0]).item())
    return t, l, b, r


def convert_yolo_boxes(yolo_box: UltralyticsBoxes | NativeYoloBox, img_shape) -> list[Box]:
    if isinstance(yolo_box, NativeYoloBox):
        return [yolo_box.box]
    _boxes = yolo_box.xyxy
    boxes = []
    for _box in _boxes:
        l = int(torch.clip(_box[0], 0, img_shape[1]).item())
        t = int(torch.clip(_box[1], 0, img_shape[0]).item())
        r = int(torch.clip(_box[2], 0, img_shape[1]).item())
        b = int(torch.clip(_box[3], 0, img_shape[0]).item())
        box = t, l, b, r
        boxes.append(box)
    return boxes

def convert_yolo_conf(yolo_box: UltralyticsBoxes | NativeYoloBox) -> float:
    if isinstance(yolo_box, NativeYoloBox):
        return yolo_box.conf_value
    return yolo_box.conf[0].item()

def scale_and_unpad_image(masks, im0_shape):
    h0, w0 = im0_shape[:2]
    h1, w1, _ = masks.shape
    if h1 == h0 and w1 == w0:
        return masks
    g = min(h1 / h0, w1 / w0)
    pw, ph = (w1 - w0 * g) / 2, (h1 - h0 * g) / 2
    t, l = round(ph - 0.1), round(pw - 0.1)
    b, r = h1 - round(ph + 0.1), w1 - round(pw + 0.1)
    x = masks[t:b, l:r].permute(2, 0, 1).unsqueeze(0).float()
    y = F.interpolate(x, size=(h0, w0), mode='bilinear', align_corners=False)
    return y.squeeze(0).permute(1, 2, 0).round_().clamp_(0, 255).to(masks.dtype)


def _ensure_mask_image(mask: np.ndarray) -> Mask:
    mask_array = mask if mask.flags.c_contiguous and mask.dtype == np.uint8 else np.ascontiguousarray(mask, dtype=np.uint8)
    if mask_array.ndim == 2:
        return mask_array[..., None]
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        return np.ascontiguousarray(mask_array[0][..., None])
    if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
        return mask_array
    raise RuntimeError(f"Unsupported native mask shape {tuple(mask_array.shape)}.")

def convert_yolo_mask_tensor(yolo_mask: UltralyticsMasks | NativeYoloMask, img_shape) -> torch.Tensor:
    mask_data = yolo_mask.data
    if (
        isinstance(yolo_mask, NativeYoloMask)
        and isinstance(mask_data, np.ndarray)
        and mask_data.dtype == np.uint8
        and tuple(mask_data.shape[:2]) == tuple(img_shape[:2])
    ):
        return torch.from_numpy(_ensure_mask_image(mask_data))

    if (
        isinstance(yolo_mask, NativeYoloMask)
        and mask_data.dtype == torch.uint8
        and tuple(mask_data.shape[-2:]) == tuple(img_shape[:2])
    ):
        if mask_data.ndim == 2:
            return mask_data.unsqueeze(-1)
        if mask_data.ndim == 3 and mask_data.shape[0] == 1:
            return mask_data[0].unsqueeze(-1)
        if mask_data.ndim == 3 and mask_data.shape[-1] == 1:
            return mask_data

    mask_img = _to_mask_img_tensor(mask_data)
    if mask_img.ndim == 2:
        mask_img = mask_img.unsqueeze(-1)
    mask_img = scale_and_unpad_image(mask_img, img_shape)
    mask_img = torch.where(mask_img > 127, 255, 0).to(torch.uint8)
    assert mask_img.ndim == 3 and mask_img.shape[2] == 1
    return mask_img


def convert_yolo_mask_image(yolo_mask: UltralyticsMasks | NativeYoloMask, img_shape) -> Mask:
    mask_data = yolo_mask.data
    if isinstance(yolo_mask, NativeYoloMask) and isinstance(mask_data, np.ndarray):
        return _ensure_mask_image(mask_data)

    mask_img = convert_yolo_mask_tensor(yolo_mask, img_shape).cpu().numpy()
    assert mask_img.ndim == 3 and mask_img.shape[2] == 1 and mask_img.dtype == np.uint8
    return np.ascontiguousarray(mask_img)

def _to_mask_img_tensor(masks: torch.Tensor, class_val=0, pixel_val=255) -> torch.Tensor:
    masks_tensor = torch.where(masks != class_val, pixel_val, 0).to(torch.uint8)
    return masks_tensor[0]

def convert_yolo_mask(yolo_mask: UltralyticsMasks, img_shape) -> Mask:
    return convert_yolo_mask_image(yolo_mask, img_shape)

def choose_biggest_detection(
    result: DetectionResult,
    tracking_mode=True,
) -> tuple[UltralyticsBoxes | NativeYoloBox | None, UltralyticsMasks | NativeYoloMask | None]:
    """
    Returns the biggest detection box and mask of a YOLO Results set
    """
    box = None
    mask = None
    yolo_box: UltralyticsBoxes | NativeYoloBox
    yolo_mask: UltralyticsMasks | NativeYoloMask
    for i, yolo_box in enumerate(result.boxes):
        if tracking_mode and yolo_box.id is None:
            continue
        yolo_mask = result.masks[i]
        if box is None:
            box = yolo_box
            mask = yolo_mask
        else:
            box_dims = box.xywh[0]
            _box_dims = yolo_box.xywh[0]
            box_size = box_dims[2] * box_dims[3]
            _box_size = _box_dims[2] * _box_dims[3]
            if _box_size > box_size:
                box = yolo_box
                mask = yolo_mask
    return box, mask

def _get_unique_pixel_values(mask: Mask) -> list[int]:
    # get unique values except background (0)
    unique_values = np.unique(mask).tolist()
    if 0 in unique_values: unique_values.remove(0)  # remove background class
    return unique_values

def convert_segment_masks_to_yolo_labels(masks_dir, output_dir_segmentation_labels, output_dir_detection_labels, pixel_to_class_mapping):
    """
    pixel_to_class_mapping is a dict providing a mapping from pixel value to class id.
    e.g. if you only have a single class with id 0 and binary masks use pixel value 255 then this would be:
    pixel_to_class_mapping = {255: 0}

    Based of: ultralytics.data.converter.convert_segment_masks_to_yolo_seg
    """

    PRECISION = 6 # Rounding to 6 decimal places

    def get_yolo_box(contour, img_width, img_height) -> tuple[float]:
        yolo_detection_format_data
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w / 2
        center_y = y + h / 2
        yolo_box = center_x / img_width, center_y / img_height, w / img_width, h / img_height
        return [round(num, PRECISION) for num in yolo_box]

    def get_yolo_segment_polygon(contour, img_width, img_height) -> tuple[float]:
        yolo_polygon = []
        for point in contour:
            yolo_polygon.append(round(point[0] / img_width, PRECISION))
            yolo_polygon.append(round(point[1] / img_height, PRECISION))
        return yolo_polygon

    for mask_path in Path(masks_dir).iterdir():
        if mask_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            img_height, img_width = mask.shape

            unique_values = _get_unique_pixel_values(mask)
            yolo_segmentation_format_data = []
            yolo_detection_format_data = []

            for value in unique_values:
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    print(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    continue

                # Create a binary mask for the current class and find contours
                binary_mask_for_current_class = (mask == value).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask_for_current_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                        contour = contour.squeeze()  # Remove single-dimensional entries
                        segmentation_data = [class_index] + get_yolo_segment_polygon(contour, img_width, img_height)
                        yolo_segmentation_format_data.append(segmentation_data)
                        detection_data = [class_index] + get_yolo_box(contour, img_width, img_height)
                        yolo_detection_format_data.append(detection_data)

            # Save Ultralytics YOLO format data to file
            output_path = Path(output_dir_segmentation_labels) / f"{mask_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                for item in yolo_segmentation_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
            output_path = Path(output_dir_detection_labels) / f"{mask_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                for item in yolo_detection_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
