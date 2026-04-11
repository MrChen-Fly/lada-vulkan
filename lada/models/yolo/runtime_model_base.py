# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import nms, ops
from ultralytics.utils.checks import check_imgsz

from lada.utils import Image, ImageTensor
from lada.utils.torch_letterbox import PyTorchLetterBox
from lada.models.yolo.runtime_results import DetectionResult


class BaseYolo11SegmentationModel:
    """Shared runtime helpers for Torch and Vulkan YOLO segmentation backends."""

    def __init__(
        self,
        *,
        names,
        args,
        imgsz: int = 640,
        stride: int = 32,
        letterbox_auto: bool = True,
        torch_device: torch.device | None,
        dtype: torch.dtype,
    ):
        self.stride = stride
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox_auto = letterbox_auto
        self.letterbox: PyTorchLetterBox | LetterBox = LetterBox(
            self.imgsz,
            auto=self.letterbox_auto,
            stride=self.stride,
        )
        self.names = names
        self.args = args
        self.torch_device = torch_device
        self.dtype = dtype
        self.end2end = False
        self._profile_durations: defaultdict[str, float] = defaultdict(float)
        self._profile_counts: defaultdict[str, int] = defaultdict(int)

    @contextmanager
    def _measure_profile(self, bucket: str):
        started_at = perf_counter()
        try:
            yield
        finally:
            self._profile_durations[bucket] += perf_counter() - started_at
            self._profile_counts[bucket] += 1

    def consume_profile(self) -> dict[str, float | int]:
        snapshot: dict[str, float | int] = dict(self._profile_durations)
        for bucket, count in self._profile_counts.items():
            snapshot[f"{bucket}__count"] = count
        self._profile_durations.clear()
        self._profile_counts.clear()
        return snapshot

    @staticmethod
    def build_segmentation_args(
        yolo_model: YOLO,
        device: str | torch.device | None,
        fp16: bool,
        **kwargs,
    ) -> Any:
        custom = {
            "conf": 0.25,
            "batch": 1,
            "save": False,
            "mode": "predict",
            "device": str(device) if device is not None else "cpu",
            "half": fp16,
        }
        return get_cfg(DEFAULT_CFG, {**yolo_model.overrides, **custom, **kwargs})

    @staticmethod
    def _to_numpy_image(img: Image | ImageTensor) -> np.ndarray:
        if isinstance(img, torch.Tensor):
            return np.ascontiguousarray(img.numpy())
        return np.ascontiguousarray(img)

    def _preprocess_cpu(self, imgs: list[Image | ImageTensor]) -> torch.Tensor:
        with self._measure_profile("detector_preprocess_letterbox_s"):
            im = np.stack([self.letterbox(image=self._to_numpy_image(x)) for x in imgs])
            im = im.transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
        with self._measure_profile("detector_preprocess_tensorize_s"):
            return torch.from_numpy(im)

    def _preprocess_gpu(self, imgs: list[ImageTensor]) -> torch.Tensor:
        with self._measure_profile("detector_preprocess_stack_s"):
            batch = torch.stack(imgs, dim=0)
        with self._measure_profile("detector_preprocess_letterbox_s"):
            return self.letterbox(batch)

    def preprocess(self, imgs: list[Image | ImageTensor]) -> torch.Tensor:
        with self._measure_profile("detector_preprocess_total_s"):
            if not isinstance(imgs[0], torch.Tensor) or imgs[0].device.type == "cpu":
                return self._preprocess_cpu(imgs)

            original_shape = tuple(imgs[0].shape[:2])
            if getattr(self.letterbox, "original_shape", None) != original_shape:
                self.letterbox = PyTorchLetterBox(
                    self.imgsz,
                    original_shape,
                    stride=self.stride,
                )
            return self._preprocess_gpu(imgs)

    def prepare_input(self, imgs: torch.Tensor) -> torch.Tensor:
        with self._measure_profile("detector_prepare_input_s"):
            if self.torch_device is None:
                return imgs.to(dtype=self.dtype).div_(255.0)
            return imgs.to(device=self.torch_device).to(dtype=self.dtype).div_(255.0)

    def inference(self, image_batch: torch.Tensor):
        raise NotImplementedError

    def inference_and_postprocess(
        self,
        imgs: torch.Tensor,
        orig_imgs: list[Image | ImageTensor],
    ) -> list[DetectionResult]:
        with torch.inference_mode():
            input_batch = self.prepare_input(imgs)
            preds = self.inference(input_batch)
            return self.postprocess(preds, input_batch, orig_imgs)

    def postprocess(self, preds, img, orig_imgs: list[Image | ImageTensor]) -> list[Results]:
        protos = preds[0][-1]
        preds = nms.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.names),
            end2end=self.end2end,
        )
        return [
            self.construct_result(pred, img, orig_img, proto)
            for pred, orig_img, proto in zip(preds, orig_imgs, protos)
        ]

    def construct_result(
        self,
        preds: torch.Tensor,
        img: torch.Tensor,
        orig_img: Image | ImageTensor,
        proto: torch.Tensor,
    ) -> Results:
        if not len(preds):
            masks = None
        else:
            masks = ops.process_mask(
                proto,
                preds[:, 6:],
                preds[:, :4],
                img.shape[2:],
                upsample=True,
            )
            preds[:, :4] = ops.scale_boxes(img.shape[2:], preds[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0
            preds, masks = preds[keep], masks[keep]
        return Results(
            orig_img,
            path="",
            names=self.names,
            boxes=preds[:, :6].cpu(),
            masks=masks,
        )

    def release_cached_memory(self) -> None:
        return None
