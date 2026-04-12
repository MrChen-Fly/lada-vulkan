from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from ultralytics.data.augment import LetterBox

from lada.models.yolo.runtime_model_base import BaseYolo11SegmentationModel
from lada.models.yolo.runtime_results import DetectionResult
from lada.utils import Image, ImageTensor

from .artifacts import IreeDetectionArtifacts, resolve_iree_detection_artifacts
from .compiler import ensure_iree_detection_vmfb
from .runtime import configure_vulkan_iree_device_info, probe_iree_vulkan_runtime
from .yolo_support import build_segmentation_args, normalize_runtime_imgsz, resolve_letterbox_output_shape


def _select_prediction_and_proto(outputs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Select `(pred, proto)` tensors from one IREE module invocation result."""
    values = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
    arrays = [np.asarray(value) for value in values]

    pred = next((array for array in arrays if array.ndim == 3), None)
    proto = next((array for array in arrays if array.ndim == 4), None)

    if pred is None:
        pred2 = next((array for array in arrays if array.ndim == 2), None)
        if pred2 is not None:
            pred = pred2[None, ...]

    if proto is None:
        proto3 = next((array for array in arrays if array.ndim == 3), None)
        if proto3 is not None and proto3 is not pred:
            proto = proto3[None, ...]

    if pred is None or proto is None:
        shapes = [tuple(array.shape) for array in arrays]
        raise RuntimeError(f"Unexpected IREE YOLO outputs: {shapes}")

    return np.ascontiguousarray(pred), np.ascontiguousarray(proto)


def _list_runtime_export_names(runtime_module: Any) -> tuple[str, ...]:
    ordered_names: list[str] = []

    def add(name: Any) -> None:
        if not isinstance(name, str):
            return
        normalized = name.strip()
        if (
            not normalized
            or normalized.startswith("_")
            or normalized == "__init"
            or normalized.endswith("$async")
            or normalized in ordered_names
        ):
            return
        ordered_names.append(normalized)

    for name in getattr(runtime_module, "function_names", ()):
        add(name)

    try:
        for name in dir(runtime_module):
            add(name)
    except Exception:
        pass

    try:
        match = re.search(r"\[(.*?)\]", repr(runtime_module))
    except Exception:
        match = None
    if match:
        for raw_name in match.group(1).split(","):
            add(raw_name)

    return tuple(ordered_names)


def _resolve_runtime_entry_function(runtime_module: Any, preferred_name: str) -> tuple[str, Any]:
    candidate_names: list[str] = []

    def add_candidate(name: str | None) -> None:
        normalized = str(name or "").strip()
        if normalized and normalized not in candidate_names:
            candidate_names.append(normalized)

    add_candidate(preferred_name)
    add_candidate("main_graph")
    add_candidate("main")
    for exported_name in _list_runtime_export_names(runtime_module):
        add_candidate(exported_name)

    for candidate_name in candidate_names:
        try:
            entry_function = getattr(runtime_module, candidate_name)
        except AttributeError:
            continue
        if callable(entry_function):
            return candidate_name, entry_function

    raise RuntimeError(
        "Unable to resolve an executable IREE YOLO entry function. "
        f"Requested '{preferred_name}', available exports: {list(_list_runtime_export_names(runtime_module))}."
    )


class IreeVulkanYoloSegmentationModel(BaseYolo11SegmentationModel):
    """Run the YOLO segmentation detector through an IREE Vulkan module."""

    runtime = "vulkan-iree"

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        fp16: bool = False,
        device_index: int = 0,
        **kwargs,
    ):
        self.source_model_path = model_path
        self.device_index = max(int(device_index), 0)
        self.fp16_requested = bool(fp16)
        self.base_imgsz = normalize_runtime_imgsz(imgsz)
        self.active_input_shape = self.base_imgsz
        self.runtime_module = None
        self.runtime_entry_function = None
        self.runtime_entry_function_name = ""
        self.artifacts: IreeDetectionArtifacts | None = None

        artifacts = resolve_iree_detection_artifacts(
            model_path,
            imgsz=self.base_imgsz,
            prefer_fp16=self.fp16_requested,
        )
        args = build_segmentation_args(
            base_overrides=artifacts.base_overrides,
            fp16=artifacts.fp16,
            **kwargs,
        )
        super().__init__(
            names=artifacts.names,
            args=args,
            imgsz=self.base_imgsz,
            letterbox_auto=True,
            torch_device=None,
            dtype=torch.float16 if artifacts.fp16 else torch.float32,
        )
        self._configure_runtime(artifacts)
        self._warmup()

    def _load_runtime_module(self, artifacts: IreeDetectionArtifacts):
        probe = probe_iree_vulkan_runtime()
        if not probe.available:
            raise RuntimeError(probe.error or "IREE Vulkan runtime is unavailable.")
        import iree.runtime as iree_runtime

        if configure_vulkan_iree_device_info():
            # Querying devices here makes the chosen Vulkan path observable without touching mainline code.
            iree_runtime.get_driver("vulkan").query_available_devices()
        return iree_runtime.load_vm_flatbuffer_file(str(artifacts.vmfb_path), driver="vulkan")

    def _configure_runtime(self, artifacts: IreeDetectionArtifacts) -> None:
        self.artifacts = ensure_iree_detection_vmfb(artifacts)
        self.active_input_shape = artifacts.imgsz
        self.runtime_module = self._load_runtime_module(artifacts)
        (
            self.runtime_entry_function_name,
            self.runtime_entry_function,
        ) = _resolve_runtime_entry_function(self.runtime_module, self.artifacts.entry_function)

    def _resolve_runtime_input_shape(
        self,
        imgs: list[Image | ImageTensor],
    ) -> tuple[int, int]:
        shapes = {(int(img.shape[0]), int(img.shape[1])) for img in imgs}
        if not shapes:
            return self.active_input_shape
        if len(shapes) != 1:
            raise RuntimeError(
                f"IREE Vulkan detection expects one uniform frame shape per batch, got {sorted(shapes)}."
            )
        return resolve_letterbox_output_shape(
            next(iter(shapes)),
            target_shape=self.base_imgsz,
            stride=self.stride,
            auto=self.letterbox_auto,
        )

    def _ensure_runtime_for_input_shape(self, input_shape: tuple[int, int]) -> None:
        normalized_shape = normalize_runtime_imgsz(input_shape)
        if normalized_shape == self.active_input_shape and self.runtime_module is not None:
            return
        artifacts = resolve_iree_detection_artifacts(
            self.source_model_path,
            imgsz=normalized_shape,
            prefer_fp16=self.fp16_requested,
        )
        self._configure_runtime(artifacts)

    def preprocess(self, imgs: list[Image | ImageTensor]) -> torch.Tensor:
        with self._measure_profile("detector_preprocess_total_s"):
            input_shape = self._resolve_runtime_input_shape(imgs)
            self._ensure_runtime_for_input_shape(input_shape)
            previous_imgsz = self.imgsz
            previous_letterbox = self.letterbox
            try:
                self.imgsz = input_shape
                self.letterbox = LetterBox(
                    self.imgsz,
                    auto=self.letterbox_auto,
                    stride=self.stride,
                )
                with self._measure_profile("detector_preprocess_letterbox_s"):
                    shared_preprocessed = self._preprocess_cpu(imgs)
                return self.prepare_input(shared_preprocessed)
            finally:
                self.imgsz = previous_imgsz
                self.letterbox = previous_letterbox

    def _run_module(self, image_batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if self.runtime_module is None or self.artifacts is None or self.runtime_entry_function is None:
            raise RuntimeError("IREE YOLO runtime module is not configured.")
        outputs = self.runtime_entry_function(
            np.ascontiguousarray(image_batch.detach().cpu().numpy())
        )
        return _select_prediction_and_proto(outputs)

    def _warmup(self) -> None:
        warmup_h, warmup_w = self.active_input_shape
        dummy = np.zeros((warmup_h, warmup_w, 3), dtype=np.uint8)
        prepared = self.preprocess([dummy])
        self._run_module(prepared)

    def inference(self, image_batch: torch.Tensor):
        pred_batch, proto_batch = self._run_module(image_batch)
        return [(torch.from_numpy(pred_batch), torch.from_numpy(proto_batch)), {}]

    def inference_and_postprocess(
        self,
        imgs: torch.Tensor,
        orig_imgs: list[Image | ImageTensor],
    ) -> list[DetectionResult]:
        with torch.inference_mode():
            input_batch = imgs if isinstance(imgs, torch.Tensor) else self.prepare_input(imgs)
            pred_batch, proto_batch = self._run_module(input_batch)
            raw_results = self.postprocess(
                [(torch.from_numpy(pred_batch), torch.from_numpy(proto_batch)), {}],
                input_batch,
                orig_imgs,
            )
            return list(raw_results)
