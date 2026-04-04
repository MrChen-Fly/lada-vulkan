# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Tuple

import cv2
import numpy as np
import torch

from lada.utils import Box
from lada.utils import Image, ImageTensor, Mask, MaskTensor, Pad, VideoMetadata
from lada.utils import image_utils
from lada.utils.box_utils import box_overlap
from lada.utils.scene_utils import expand_box_to_target

_CROP_TARGET_EXPANSION_FACTOR = 1.0
_CROP_BORDER_RATIO = 0.06

type SceneMask = Mask | MaskTensor


def _resolve_crop_box(
    box: Box,
    frame_shape: tuple[int, ...],
    size: int,
) -> Box:
    crop_box, _ = expand_box_to_target(
        box,
        frame_shape,
        (size, size),
        max_box_expansion_factor=_CROP_TARGET_EXPANSION_FACTOR,
        border_size=_CROP_BORDER_RATIO,
    )
    return crop_box


def _crop_with_box(
    img: Image | ImageTensor,
    mask: SceneMask,
    crop_box: Box,
) -> tuple[Image | ImageTensor, SceneMask]:
    t, l, b, r = crop_box
    return img[t : b + 1, l : r + 1], mask[t : b + 1, l : r + 1]


def _merge_scene_masks(base: SceneMask, update: SceneMask) -> SceneMask:
    if isinstance(base, torch.Tensor) or isinstance(update, torch.Tensor):
        base_tensor = base if isinstance(base, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(base))
        update_tensor = (
            update.to(device=base_tensor.device)
            if isinstance(update, torch.Tensor)
            else torch.from_numpy(np.ascontiguousarray(update)).to(device=base_tensor.device)
        )
        return torch.maximum(base_tensor, update_tensor)
    return np.maximum(base, update)


@dataclass
class Scene:
    file_path: str
    video_meta_data: VideoMetadata
    crop_size: int = 256
    frames: list[Image | ImageTensor] = field(default_factory=list)
    masks: list[SceneMask] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    crop_boxes: list[Box] = field(default_factory=list)
    resize_reference_shape: tuple[int, int] = (0, 0)
    frame_start: int | None = None
    frame_end: int | None = None
    _index: int = 0

    def __len__(self) -> int:
        return len(self.frames)

    def add_frame(
        self,
        frame_num: int,
        img: Image | ImageTensor,
        mask: SceneMask,
        box: Box,
    ) -> None:
        if self.frame_start is None:
            self.frame_start = frame_num
            self.frame_end = frame_num
        else:
            assert frame_num == self.frame_end + 1
            self.frame_end = frame_num

        self.frames.append(img)
        self.masks.append(mask)
        self.boxes.append(box)
        self.crop_boxes.append(_resolve_crop_box(box, img.shape, self.crop_size))
        self._refresh_resize_reference_shape()

    def merge_mask_box(self, mask: SceneMask, box: Box) -> None:
        assert self.belongs(box)
        current_box = self.boxes[-1]
        t = min(current_box[0], box[0])
        l = min(current_box[1], box[1])
        b = max(current_box[2], box[2])
        r = max(current_box[3], box[3])
        self.boxes[-1] = (t, l, b, r)
        self.masks[-1] = _merge_scene_masks(self.masks[-1], mask)
        self.crop_boxes[-1] = _resolve_crop_box(self.boxes[-1], self.frames[-1].shape, self.crop_size)
        self._refresh_resize_reference_shape()

    def belongs(self, box: Box) -> bool:
        if len(self.boxes) == 0:
            return False
        return box_overlap(self.boxes[-1], box)

    def __iter__(self) -> "Scene":
        self._index = 0
        return self

    def __next__(self) -> tuple[Image | ImageTensor, SceneMask, Box]:
        if self._index >= len(self):
            raise StopIteration
        item = (
            self.frames[self._index],
            self.masks[self._index],
            self.boxes[self._index],
        )
        self._index += 1
        return item

    def pop_prefix(self, frame_count: int) -> "Scene":
        if frame_count <= 0:
            raise ValueError("frame_count must be positive.")
        if frame_count > len(self.frames):
            raise ValueError("frame_count exceeds buffered scene length.")
        if self.frame_start is None:
            raise ValueError("Scene frame range must be defined.")

        prefix = Scene(self.file_path, self.video_meta_data, self.crop_size)
        prefix.frames = self.frames[:frame_count]
        prefix.masks = self.masks[:frame_count]
        prefix.boxes = self.boxes[:frame_count]
        prefix.crop_boxes = self.crop_boxes[:frame_count]
        prefix.frame_start = self.frame_start
        prefix.frame_end = self.frame_start + frame_count - 1
        prefix._refresh_resize_reference_shape()

        del self.frames[:frame_count]
        del self.masks[:frame_count]
        del self.boxes[:frame_count]
        del self.crop_boxes[:frame_count]

        if self.frames:
            self.frame_start = prefix.frame_end + 1
        else:
            self.frame_start = None
            self.frame_end = None
        self._refresh_resize_reference_shape()
        return prefix

    def _refresh_resize_reference_shape(self) -> None:
        if self.crop_boxes:
            self.resize_reference_shape = _compute_max_width_height(self.crop_boxes)
        else:
            self.resize_reference_shape = (0, 0)


def _compute_max_width_height(boxes: list[Box]) -> tuple[int, int]:
    max_width = 0
    max_height = 0
    for t, l, b, r in boxes:
        width = r - l + 1
        height = b - t + 1
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height
    return max_width, max_height


@dataclass
class ClipDescriptor:
    file_path: str
    frame_start: int
    size: int
    pad_mode: str
    id: str | int
    frames: list[Image | ImageTensor]
    masks: list[SceneMask]
    boxes: list[Box]
    crop_boxes: list[Box]
    resize_reference_shape: tuple[int, int]

    @classmethod
    def from_scene(
        cls,
        scene: Scene,
        size: int,
        pad_mode: str,
        clip_id: str | int,
    ) -> "ClipDescriptor":
        if scene.frame_start is None:
            raise ValueError("Scene frame range must be defined.")
        resize_reference_shape = scene.resize_reference_shape
        if resize_reference_shape[0] <= 0 or resize_reference_shape[1] <= 0:
            resize_reference_shape = _compute_max_width_height(scene.crop_boxes)
        return cls(
            file_path=scene.file_path,
            frame_start=scene.frame_start,
            size=size,
            pad_mode=pad_mode,
            id=clip_id,
            frames=list(scene.frames),
            masks=list(scene.masks),
            boxes=list(scene.boxes),
            crop_boxes=list(scene.crop_boxes),
            resize_reference_shape=resize_reference_shape,
        )

    @property
    def frame_end(self) -> int:
        return self.frame_start + len(self.frames) - 1

    def __len__(self) -> int:
        return len(self.frames)

    def validate(self) -> None:
        frame_count = len(self.frames)
        if frame_count == 0:
            raise ValueError("ClipDescriptor requires at least one frame.")
        if frame_count != len(self.masks) or frame_count != len(self.boxes) or frame_count != len(self.crop_boxes):
            raise ValueError("ClipDescriptor tensors must have the same frame count.")
        if self.resize_reference_shape[0] <= 0 or self.resize_reference_shape[1] <= 0:
            raise ValueError("ClipDescriptor resize reference must be positive.")

    def split(self, max_frames: int) -> list["ClipDescriptor"]:
        if max_frames <= 0:
            raise ValueError("max_frames must be a positive integer.")
        if len(self) <= max_frames:
            return [self]

        segments: list[ClipDescriptor] = []
        for segment_index, start_index in enumerate(range(0, len(self.frames), max_frames)):
            end_index = min(start_index + max_frames, len(self.frames))
            segment_crop_boxes = self.crop_boxes[start_index:end_index]
            segments.append(
                ClipDescriptor(
                    file_path=self.file_path,
                    frame_start=self.frame_start + start_index,
                    size=self.size,
                    pad_mode=self.pad_mode,
                    id=f"{self.id}:{segment_index}",
                    frames=self.frames[start_index:end_index],
                    masks=self.masks[start_index:end_index],
                    boxes=self.boxes[start_index:end_index],
                    crop_boxes=segment_crop_boxes,
                    resize_reference_shape=_compute_max_width_height(segment_crop_boxes),
                )
            )
        return segments


@dataclass(frozen=True)
class ClipResizePlan:
    crop_shape: tuple[int, ...]
    resize_shape: tuple[int, int]
    pad_after_resize: Pad


def crop_descriptor_with_profile(
    descriptor: ClipDescriptor,
    profile: dict[str, float] | None = None,
) -> tuple[
    list[Image | ImageTensor],
    list[SceneMask],
    list[Box],
    list[tuple[int, ...]],
]:
    descriptor.validate()

    frames: list[Image | ImageTensor] = []
    masks: list[SceneMask] = []
    boxes: list[Box] = []
    crop_shapes: list[tuple[int, ...]] = []

    started_at = perf_counter()
    for img, mask, crop_box in zip(descriptor.frames, descriptor.masks, descriptor.crop_boxes):
        cropped_img, cropped_mask = _crop_with_box(img, mask, crop_box)
        frames.append(cropped_img)
        masks.append(cropped_mask)
        boxes.append(crop_box)
        crop_shapes.append(cropped_img.shape)

    if profile is not None:
        profile["clip_crop_s"] = profile.get("clip_crop_s", 0.0) + (perf_counter() - started_at)
    return frames, masks, boxes, crop_shapes


def build_clip_resize_plans(
    descriptor: ClipDescriptor,
    crop_shapes: list[tuple[int, ...]],
) -> list["ClipResizePlan"]:
    max_width, max_height = descriptor.resize_reference_shape
    scale_width = descriptor.size / max_width
    scale_height = descriptor.size / max_height
    resize_plans: list[ClipResizePlan] = []
    for crop_shape in crop_shapes:
        resize_shape = (
            int(crop_shape[0] * scale_height),
            int(crop_shape[1] * scale_width),
        )
        resize_height, resize_width = resize_shape
        if (
            resize_height <= 0
            or resize_width <= 0
            or resize_height > descriptor.size
            or resize_width > descriptor.size
        ):
            raise ValueError(
                "Clip resize plan produced an invalid shape "
                f"{resize_shape} for target size {descriptor.size}."
            )
        pad_h = descriptor.size - resize_height
        pad_w = descriptor.size - resize_width
        pad_after_resize = (
            int(np.ceil(pad_h / 2)),
            int(np.floor(pad_h / 2)),
            int(np.ceil(pad_w / 2)),
            int(np.floor(pad_w / 2)),
        )
        resize_plans.append(
            ClipResizePlan(
                crop_shape=crop_shape,
                resize_shape=resize_shape,
                pad_after_resize=pad_after_resize,
            )
        )
    return resize_plans


def materialize_clip_frames_with_profile(
    cropped_frames: list[Image | ImageTensor],
    resize_plans: list[ClipResizePlan],
    *,
    size: int,
    pad_mode: str,
    profile: dict[str, float] | None = None,
) -> tuple[list[Image | ImageTensor], list[Pad]]:
    started_at = perf_counter()
    frames: list[Image | ImageTensor] = []
    pad_after_resizes: list[Pad] = []
    for cropped_img, resize_plan in zip(cropped_frames, resize_plans):
        resized_img = image_utils.resize(
            cropped_img,
            resize_plan.resize_shape,
            interpolation=cv2.INTER_LINEAR,
        )
        padded_img, pad_after_resize = image_utils.pad_image(
            resized_img,
            size,
            size,
            mode=pad_mode,
        )
        frames.append(padded_img)
        pad_after_resizes.append(pad_after_resize)

    if profile is not None:
        profile["clip_resize_pad_s"] = profile.get("clip_resize_pad_s", 0.0) + (
            perf_counter() - started_at
        )
    return frames, pad_after_resizes


def materialize_clip_masks_with_profile(
    cropped_masks: list[SceneMask],
    resize_plans: list[ClipResizePlan],
    *,
    size: int,
    profile: dict[str, float] | None = None,
) -> tuple[list[MaskTensor], list[Pad]]:
    started_at = perf_counter()
    masks: list[MaskTensor] = []
    pad_after_resizes: list[Pad] = []
    for cropped_mask, resize_plan in zip(cropped_masks, resize_plans):
        resized_mask = image_utils.resize(
            cropped_mask,
            resize_plan.resize_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        padded_mask, _ = image_utils.pad_image(
            resized_mask,
            size,
            size,
            mode="zero",
        )
        if not isinstance(padded_mask, torch.Tensor):
            padded_mask = torch.from_numpy(np.ascontiguousarray(padded_mask))
        masks.append(padded_mask)
        pad_after_resizes.append(resize_plan.pad_after_resize)

    if profile is not None:
        profile["clip_resize_pad_s"] = profile.get("clip_resize_pad_s", 0.0) + (
            perf_counter() - started_at
        )
    return masks, pad_after_resizes


@dataclass
class Clip:
    file_path: str
    frame_start: int
    size: int
    pad_mode: str
    id: str | int
    frames: list[Image | ImageTensor]
    masks: list[MaskTensor]
    boxes: list[Box]
    crop_shapes: list[Tuple[int, ...]]
    pad_after_resizes: list[Pad]
    _index: int = 0

    @property
    def frame_end(self) -> int:
        return self.frame_start + len(self.frames) - 1

    @classmethod
    def from_descriptor(cls, descriptor: ClipDescriptor) -> "Clip":
        return cls.from_descriptor_with_profile(descriptor)

    @classmethod
    def from_descriptor_with_profile(
        cls,
        descriptor: ClipDescriptor,
        profile: dict[str, float] | None = None,
    ) -> "Clip":
        frames, masks, boxes, crop_shapes = crop_descriptor_with_profile(descriptor, profile)
        resize_plans = build_clip_resize_plans(descriptor, crop_shapes)
        resize_started_at = perf_counter()
        frames, pad_after_resizes = materialize_clip_frames_with_profile(
            frames,
            resize_plans,
            size=descriptor.size,
            pad_mode=descriptor.pad_mode,
            profile=None,
        )
        masks, _ = materialize_clip_masks_with_profile(
            masks,
            resize_plans,
            size=descriptor.size,
            profile=None,
        )
        if profile is not None:
            profile["clip_resize_pad_s"] = profile.get("clip_resize_pad_s", 0.0) + (
                perf_counter() - resize_started_at
            )

        clip = cls(
            file_path=descriptor.file_path,
            frame_start=descriptor.frame_start,
            size=descriptor.size,
            pad_mode=descriptor.pad_mode,
            id=descriptor.id,
            frames=frames,
            masks=masks,
            boxes=boxes,
            crop_shapes=crop_shapes,
            pad_after_resizes=pad_after_resizes,
        )
        clip._validate()
        return clip

    @classmethod
    def from_processed_data(
        cls,
        *,
        file_path: str,
        frame_start: int,
        size: int,
        pad_mode: str,
        id: str | int,
        frames: list[Image | ImageTensor],
        masks: list[MaskTensor],
        boxes: list[Box],
        crop_shapes: list[Tuple[int, ...]],
        pad_after_resizes: list[Pad],
    ) -> "Clip":
        clip = cls(
            file_path=file_path,
            frame_start=frame_start,
            size=size,
            pad_mode=pad_mode,
            id=id,
            frames=list(frames),
            masks=list(masks),
            boxes=list(boxes),
            crop_shapes=list(crop_shapes),
            pad_after_resizes=list(pad_after_resizes),
        )
        clip._validate()
        return clip

    def _validate(self) -> None:
        frame_count = len(self.frames)
        if frame_count == 0:
            raise ValueError("Clip requires at least one frame.")
        if not (
            frame_count
            == len(self.masks)
            == len(self.boxes)
            == len(self.crop_shapes)
            == len(self.pad_after_resizes)
        ):
            raise ValueError("Clip tensors must have the same frame count.")

    def __len__(self) -> int:
        return len(self.frames)

    def pop(self):
        frame, mask, box = self.frames.pop(0), self.masks.pop(0), self.boxes.pop(0)
        crop_shape = self.crop_shapes.pop(0)
        pad_after_resize = self.pad_after_resizes.pop(0)
        self.frame_start += 1
        return frame, mask, box, crop_shape, pad_after_resize

    def split(self, max_frames: int) -> list["Clip"]:
        if max_frames <= 0:
            raise ValueError("max_frames must be a positive integer.")
        if len(self) <= max_frames:
            return [self]

        segments: list[Clip] = []
        for segment_index, start_index in enumerate(range(0, len(self.frames), max_frames)):
            end_index = min(start_index + max_frames, len(self.frames))
            segments.append(
                Clip.from_processed_data(
                    file_path=self.file_path,
                    frame_start=self.frame_start + start_index,
                    size=self.size,
                    pad_mode=self.pad_mode,
                    id=f"{self.id}:{segment_index}",
                    frames=self.frames[start_index:end_index],
                    masks=self.masks[start_index:end_index],
                    boxes=self.boxes[start_index:end_index],
                    crop_shapes=self.crop_shapes[start_index:end_index],
                    pad_after_resizes=self.pad_after_resizes[start_index:end_index],
                )
            )
        return segments

    def __iter__(self) -> "Clip":
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        item = (
            self.frames[self._index],
            self.masks[self._index],
            self.boxes[self._index],
            self.crop_shapes[self._index],
            self.pad_after_resizes[self._index],
        )
        self._index += 1
        return item

    def __getitem__(self, item):
        return self.frames[item], self.masks[item], self.boxes[item]


type ClipWorkItem = Clip | ClipDescriptor
