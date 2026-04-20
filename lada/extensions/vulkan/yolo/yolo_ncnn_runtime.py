from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, YAML

from lada.extensions.runtime_registry import UnsupportedComputeTargetError
from lada.extensions.vulkan.ncnn import (
    audit_ncnn_vulkan_support,
    import_ncnn_module,
    ncnn_has_lada_yolo_attention_layer,
    ncnn_has_lada_yolo_seg_postprocess_layer,
    register_lada_custom_layers,
    set_ncnn_vulkan_device,
)
from .yolo_runtime_support import (
    get_legacy_precision_artifact_dir,
    get_precision_artifact_dir,
    normalize_runtime_imgsz,
    resolve_letterbox_output_shape,
)
from .yolo_runtime_model_base import BaseYolo11SegmentationModel
from .yolo_runtime_results import (
    DetectionResult,
    build_native_yolo_result,
)
from lada.utils import Image, ImageTensor

logger = logging.getLogger(__name__)

DEFAULT_YOLO_MAX_NMS = 30000


@dataclass(frozen=True)
class NcnnDetectionArtifacts:
    """Describe one exported NCNN detection artifact bundle."""

    model_dir: Path
    param_path: Path
    bin_path: Path
    metadata: dict[str, Any]
    fp16: bool


def _normalize_imgsz(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Return a canonical `(height, width)` image size tuple."""
    try:
        return normalize_runtime_imgsz(imgsz)
    except ValueError as exc:
        raise UnsupportedComputeTargetError(str(exc)) from exc


def _coerce_names(value: Any) -> dict[int, str]:
    """Normalize class names loaded from Ultralytics metadata."""
    if isinstance(value, Mapping):
        return {int(key): str(name) for key, name in value.items()}
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    raise UnsupportedComputeTargetError("NCNN detection metadata is missing class names.")


def _artifact_paths(model_dir: Path) -> tuple[Path, Path]:
    """Return the NCNN param/bin files inside an artifact directory."""
    return model_dir / "model.ncnn.param", model_dir / "model.ncnn.bin"


def _patched_param_path(model_dir: Path) -> Path:
    """Return the runtime-only patched NCNN param path."""
    return model_dir / "model.lada_vulkan.param"


def _postprocess_param_path(model_dir: Path) -> Path:
    """Return the generated YOLO postprocess subnet param path."""
    return model_dir / "model.yolo_seg_postprocess.param"


def _postprocess_bin_path(model_dir: Path) -> Path:
    """Return the generated YOLO postprocess subnet bin path."""
    return model_dir / "model.yolo_seg_postprocess.bin"


def _select_runtime_output_names(output_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Keep only public NCNN outputs and drop orphaned internal blobs from patched graphs."""
    public_outputs = tuple(sorted(name for name in output_names if name.startswith("out")))
    return public_outputs or tuple(sorted(output_names))


def _precision_artifact_dir(
    model_path: Path,
    *,
    fp16: bool,
    imgsz: int | tuple[int, int] | list[int],
) -> Path:
    """Return Lada's precision-specific NCNN artifact directory."""
    return get_precision_artifact_dir(model_path, fp16=fp16, imgsz=imgsz)


def _load_artifact_metadata(model_dir: Path) -> dict[str, Any] | None:
    """Load `metadata.yaml` from an NCNN export directory when available."""
    metadata_path = model_dir / "metadata.yaml"
    if not metadata_path.exists():
        return None

    try:
        metadata = YAML.load(metadata_path)
    except Exception:
        return None
    return metadata if isinstance(metadata, Mapping) else None


def _artifacts_match(model_dir: Path, *, imgsz: tuple[int, int], fp16: bool) -> bool:
    """Return whether an export directory matches the requested runtime settings."""
    param_path, bin_path = _artifact_paths(model_dir)
    metadata = _load_artifact_metadata(model_dir)
    if metadata is None or not param_path.exists() or not bin_path.exists():
        return False

    metadata_imgsz = tuple(int(value) for value in metadata.get("imgsz", []))
    metadata_half = bool(metadata.get("args", {}).get("half", False))
    metadata_task = metadata.get("task")
    return metadata_task == "segment" and metadata_imgsz == imgsz and metadata_half == fp16


def _build_artifacts(model_dir: Path, *, fp16: bool) -> NcnnDetectionArtifacts:
    """Create the artifact descriptor from an existing NCNN export directory."""
    param_path, bin_path = _artifact_paths(model_dir)
    metadata = _load_artifact_metadata(model_dir)
    if metadata is None or not param_path.exists() or not bin_path.exists():
        raise UnsupportedComputeTargetError(
            f"NCNN detection artifacts are incomplete in '{model_dir}'."
        )
    return NcnnDetectionArtifacts(
        model_dir=model_dir,
        param_path=param_path,
        bin_path=bin_path,
        metadata=dict(metadata),
        fp16=fp16,
    )


@dataclass(frozen=True)
class _ParsedLayerLine:
    """Describe one line inside an NCNN `.param` graph."""

    layer_type: str
    layer_name: str
    input_count: int
    output_count: int
    bottoms: tuple[str, ...]
    tops: tuple[str, ...]
    params: tuple[str, ...]


def _parse_param_layer_line(line: str) -> _ParsedLayerLine:
    """Parse one NCNN layer line."""
    parts = line.split()
    if len(parts) < 4:
        raise UnsupportedComputeTargetError(f"Malformed NCNN layer line: '{line}'.")

    input_count = int(parts[2])
    output_count = int(parts[3])
    bottoms = tuple(parts[4 : 4 + input_count])
    tops_start = 4 + input_count
    tops = tuple(parts[tops_start : tops_start + output_count])
    params = tuple(parts[tops_start + output_count :])
    return _ParsedLayerLine(
        layer_type=parts[0],
        layer_name=parts[1],
        input_count=input_count,
        output_count=output_count,
        bottoms=bottoms,
        tops=tops,
        params=params,
    )


def _format_param_layer_line(
    layer_type: str,
    layer_name: str,
    bottoms: tuple[str, ...],
    tops: tuple[str, ...],
    params: tuple[str, ...] = (),
) -> str:
    """Serialize one NCNN layer line."""
    tokens = [
        layer_type,
        layer_name,
        str(len(bottoms)),
        str(len(tops)),
        *bottoms,
        *tops,
        *params,
    ]
    return " ".join(tokens)


def _build_attention_replacement_lines(
    permute_line: _ParsedLayerLine,
    qk_line: _ParsedLayerLine,
    scale_line: _ParsedLayerLine,
    softmax_line: _ParsedLayerLine,
    value_line: _ParsedLayerLine,
) -> tuple[str, str, str, str, str]:
    """Create replacement lines for the exported YOLO attention block."""
    query_blob = permute_line.bottoms[0]
    key_blob = qk_line.bottoms[1]
    value_blob = value_line.bottoms[0]

    return (
        _format_param_layer_line(
            "Noop",
            permute_line.layer_name,
            (query_blob,),
            permute_line.tops,
        ),
        _format_param_layer_line(
            "Noop",
            qk_line.layer_name,
            (permute_line.tops[0],),
            qk_line.tops,
        ),
        _format_param_layer_line(
            "Noop",
            scale_line.layer_name,
            (qk_line.tops[0],),
            scale_line.tops,
        ),
        _format_param_layer_line(
            "Noop",
            softmax_line.layer_name,
            (scale_line.tops[0],),
            softmax_line.tops,
        ),
        _format_param_layer_line(
            "lada.YoloAttention",
            value_line.layer_name,
            (query_blob, key_blob, value_blob),
            value_line.tops,
        ),
    )


def _patch_attention_subgraph_text(param_text: str) -> str:
    """Patch the exported attention subgraph to use `lada.YoloAttention`."""
    lines = param_text.splitlines()
    if len(lines) < 3:
        return param_text

    graph_lines = lines[2:]
    changed = False
    for index in range(len(graph_lines) - 4):
        try:
            permute_line = _parse_param_layer_line(graph_lines[index])
            qk_line = _parse_param_layer_line(graph_lines[index + 1])
            scale_line = _parse_param_layer_line(graph_lines[index + 2])
            softmax_line = _parse_param_layer_line(graph_lines[index + 3])
            value_line = _parse_param_layer_line(graph_lines[index + 4])
        except Exception:
            continue

        if permute_line.layer_type != "Permute" or qk_line.layer_type != "MatMul":
            continue
        if scale_line.layer_type != "BinaryOp" or softmax_line.layer_type != "Softmax":
            continue
        if value_line.layer_type != "MatMul":
            continue
        if qk_line.input_count != 2 or value_line.input_count != 2:
            continue
        if permute_line.output_count != 1 or qk_line.output_count != 1:
            continue
        if scale_line.output_count != 1 or softmax_line.output_count != 1:
            continue
        if value_line.output_count != 1:
            continue
        if qk_line.bottoms[0] != permute_line.tops[0]:
            continue
        if scale_line.bottoms[0] != qk_line.tops[0]:
            continue
        if softmax_line.bottoms[0] != scale_line.tops[0]:
            continue
        if value_line.bottoms[1] != softmax_line.tops[0]:
            continue
        if "0=1" not in value_line.params:
            continue

        replacement_lines = _build_attention_replacement_lines(
            permute_line,
            qk_line,
            scale_line,
            softmax_line,
            value_line,
        )
        graph_lines[index : index + 5] = list(replacement_lines)
        changed = True

    if not changed:
        return param_text
    return "\n".join([*lines[:2], *graph_lines]) + "\n"


def _prepare_runtime_param_path(
    artifacts: NcnnDetectionArtifacts,
    *,
    ncnn_module: Any,
) -> Path:
    """Return the param path that should be loaded for the live runtime."""
    if not ncnn_has_lada_yolo_attention_layer(ncnn_module):
        return artifacts.param_path

    original_text = artifacts.param_path.read_text(encoding="utf-8")
    patched_text = _patch_attention_subgraph_text(original_text)
    if patched_text == original_text:
        return artifacts.param_path

    patched_path = _patched_param_path(artifacts.model_dir)
    if not patched_path.exists() or patched_path.read_text(encoding="utf-8") != patched_text:
        patched_path.write_text(patched_text, encoding="utf-8")
    return patched_path


def _build_postprocess_param_text(*, max_det: int, num_classes: int) -> str:
    """Build the standalone NCNN param graph for `lada.YoloSegPostprocess`."""
    lines = [
        "7767517",
        "4 6",
        "Input                    pred                     0 1 pred",
        "Input                    proto                    0 1 proto",
        "Input                    config                   0 1 config",
        (
            "lada.YoloSegPostprocess  yolo_seg_postprocess     3 3 "
            f"pred proto config boxes selected count 0={int(max_det)} 1={int(num_classes)}"
        ),
    ]
    return "\n".join(lines) + "\n"


def _prepare_postprocess_artifacts(
    artifacts: NcnnDetectionArtifacts,
    *,
    ncnn_module: Any,
    max_det: int,
    num_classes: int,
) -> tuple[Path, Path] | None:
    """Materialize the standalone YOLO postprocess subnet when the runtime exposes the layer."""
    if not ncnn_has_lada_yolo_seg_postprocess_layer(ncnn_module):
        return None

    param_path = _postprocess_param_path(artifacts.model_dir)
    bin_path = _postprocess_bin_path(artifacts.model_dir)
    param_text = _build_postprocess_param_text(max_det=max_det, num_classes=num_classes)
    if not param_path.exists() or param_path.read_text(encoding="utf-8") != param_text:
        param_path.write_text(param_text, encoding="utf-8")
    if not bin_path.exists():
        bin_path.write_bytes(b"")
    return param_path, bin_path


def _resolve_existing_artifacts(
    model_path: Path,
    *,
    imgsz: tuple[int, int],
    fp16: bool,
) -> NcnnDetectionArtifacts | None:
    """Return the precision-specific artifact bundle when it already exists."""
    candidate_dirs = (
        _precision_artifact_dir(model_path, fp16=fp16, imgsz=imgsz),
        get_legacy_precision_artifact_dir(model_path, fp16=fp16),
    )
    for candidate_dir in candidate_dirs:
        if _artifacts_match(candidate_dir, imgsz=imgsz, fp16=fp16):
            return _build_artifacts(candidate_dir, fp16=fp16)
    return None


def _move_export_dir(source_dir: Path, target_dir: Path) -> Path:
    """Move an Ultralytics export directory into Lada's precision-specific cache path."""
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))
    return target_dir


def _export_ncnn_artifacts(
    model_path: Path,
    *,
    imgsz: tuple[int, int],
    fp16: bool,
) -> NcnnDetectionArtifacts:
    """Export one `.pt` segmentation model into NCNN artifacts."""
    target_dir = _precision_artifact_dir(model_path, fp16=fp16, imgsz=imgsz)

    yolo_model = YOLO(str(model_path))
    if yolo_model.task != "segment":
        raise UnsupportedComputeTargetError(
            f"NCNN Vulkan detection requires a segmentation model, got task '{yolo_model.task}'."
        )

    result_dir = Path(
        yolo_model.export(
            format="ncnn",
            imgsz=list(imgsz),
            half=fp16,
            device="cpu",
        )
    )
    if result_dir.resolve() != target_dir.resolve():
        result_dir = _move_export_dir(result_dir, target_dir)
    return _build_artifacts(result_dir, fp16=fp16)


def resolve_ncnn_detection_artifacts(
    model_path: str,
    *,
    imgsz: int | tuple[int, int] | list[int],
    prefer_fp16: bool,
) -> NcnnDetectionArtifacts:
    """Resolve or export NCNN artifacts for one detection model."""
    requested_imgsz = _normalize_imgsz(imgsz)
    path = Path(model_path)
    if not path.exists():
        raise UnsupportedComputeTargetError(f"Detection model '{model_path}' does not exist.")

    if path.is_dir():
        metadata = _load_artifact_metadata(path)
        if metadata is None:
            raise UnsupportedComputeTargetError(
                f"NCNN artifact directory '{path}' is missing metadata.yaml."
            )
        metadata_imgsz = tuple(int(value) for value in metadata.get("imgsz", []))
        if metadata.get("task") != "segment":
            raise UnsupportedComputeTargetError(
                f"NCNN artifact directory '{path}' is not a segmentation model."
            )
        if metadata_imgsz != requested_imgsz:
            raise UnsupportedComputeTargetError(
                f"NCNN artifact '{path}' was exported for imgsz={metadata_imgsz}, "
                f"but the pipeline requested imgsz={requested_imgsz}."
            )
        return _build_artifacts(path, fp16=bool(metadata.get("args", {}).get("half", False)))

    if path.suffix.lower() != ".pt":
        raise UnsupportedComputeTargetError(
            "NCNN Vulkan detection expects a `.pt` model or a pre-exported `_ncnn_model` directory."
        )

    fp16 = bool(prefer_fp16)
    existing = _resolve_existing_artifacts(path, imgsz=requested_imgsz, fp16=fp16)
    if existing is not None:
        return existing
    try:
        return _export_ncnn_artifacts(path, imgsz=requested_imgsz, fp16=fp16)
    except Exception as exc:
        raise UnsupportedComputeTargetError(
            f"Failed to prepare NCNN detection artifacts for '{model_path}' with fp16={fp16}."
        ) from exc


def _build_segmentation_args(
    *,
    model_path: str,
    fp16: bool,
    **kwargs,
) -> Any:
    """Build Ultralytics prediction args for the NCNN detection runtime."""
    base_overrides: dict[str, Any] = {}
    path = Path(model_path)
    if path.suffix.lower() == ".pt":
        yolo_model = YOLO(model_path)
        if yolo_model.task != "segment":
            raise UnsupportedComputeTargetError(
                f"NCNN Vulkan detection requires a segmentation model, got task '{yolo_model.task}'."
            )
        base_overrides = dict(yolo_model.overrides)

    custom = {
        "conf": 0.25,
        "batch": 1,
        "save": False,
        "mode": "predict",
        "device": "cpu",
        "half": fp16,
    }
    return get_cfg(DEFAULT_CFG, {**base_overrides, **custom, **kwargs})


def _resolve_yolo_max_nms(value: Any) -> int:
    """Resolve the Ultralytics `max_nms` limit, defaulting to the Torch path default."""
    if value is None:
        return DEFAULT_YOLO_MAX_NMS
    return max(int(value), 0)


class NcnnVulkanYoloSegmentationModel(BaseYolo11SegmentationModel):
    """Run the YOLO segmentation detection model through NCNN Vulkan."""

    runtime = "vulkan"

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
        self.ncnn = import_ncnn_module()
        self.fp16_requested = bool(fp16)
        self.gpu_runner = None
        self.base_imgsz = _normalize_imgsz(imgsz)
        self.active_input_shape = self.base_imgsz

        artifacts = resolve_ncnn_detection_artifacts(
            model_path,
            imgsz=self.base_imgsz,
            prefer_fp16=self.fp16_requested,
        )
        names = _coerce_names(artifacts.metadata.get("names"))
        args = _build_segmentation_args(model_path=model_path, fp16=artifacts.fp16, **kwargs)
        self.max_nms = _resolve_yolo_max_nms(getattr(args, "max_nms", kwargs.get("max_nms")))
        super().__init__(
            names=names,
            args=args,
            imgsz=self.base_imgsz,
            letterbox_auto=True,
            torch_device=None,
            dtype=torch.float32,
        )
        self._configure_runtime(artifacts)
        self._warmup(self.active_input_shape)

    def _configure_runtime(self, artifacts: NcnnDetectionArtifacts) -> None:
        """Bind one artifact bundle to the live NCNN runtime."""
        self.artifacts = artifacts
        self.active_input_shape = _normalize_imgsz(artifacts.metadata.get("imgsz", self.base_imgsz))
        self.runtime_param_path = _prepare_runtime_param_path(artifacts, ncnn_module=self.ncnn)
        postprocess_artifacts = _prepare_postprocess_artifacts(
            artifacts,
            ncnn_module=self.ncnn,
            max_det=int(self.args.max_det),
            num_classes=len(self.names),
        )
        if postprocess_artifacts is None:
            self.postprocess_param_path = None
            self.postprocess_bin_path = None
            self.postprocess_net = None
            self.postprocess_runner = None
        else:
            self.postprocess_param_path, self.postprocess_bin_path = postprocess_artifacts
        self.layer_audit = audit_ncnn_vulkan_support(self.runtime_param_path, ncnn_module=self.ncnn)
        self.net = self._load_net(
            param_path=self.runtime_param_path,
            bin_path=artifacts.bin_path,
            fp16=artifacts.fp16,
        )
        runner_type = getattr(self.ncnn, "LadaVulkanNetRunner", None)
        self.gpu_runner = runner_type(self.net) if callable(runner_type) else None
        if self.postprocess_param_path is not None and self.postprocess_bin_path is not None:
            self.postprocess_net = self._load_net(
                param_path=self.postprocess_param_path,
                bin_path=self.postprocess_bin_path,
                fp16=artifacts.fp16,
            )
            self.postprocess_runner = runner_type(self.postprocess_net) if callable(runner_type) else None
        else:
            self.postprocess_net = None
            self.postprocess_runner = None
        self.input_name = next(iter(self.net.input_names()))
        self.output_names = _select_runtime_output_names(tuple(self.net.output_names()))
        self.effective_fp16 = artifacts.fp16

    def _resolve_runtime_input_shape(
        self,
        imgs: list[Image | ImageTensor],
    ) -> tuple[int, int]:
        """Resolve the active detector input shape for one batch."""
        shapes = {
            (int(img.shape[0]), int(img.shape[1]))
            for img in imgs
        }
        if not shapes:
            return self.active_input_shape
        if len(shapes) != 1:
            raise RuntimeError(
                f"NCNN Vulkan detection expects one uniform frame shape per batch, got {sorted(shapes)}."
            )
        return resolve_letterbox_output_shape(
            next(iter(shapes)),
            target_shape=self.base_imgsz,
            stride=self.stride,
            auto=self.letterbox_auto,
        )

    def _ensure_runtime_for_input_shape(self, input_shape: tuple[int, int]) -> None:
        """Switch to a shape-matched NCNN artifact bundle when the rect input shape changes."""
        normalized_shape = _normalize_imgsz(input_shape)
        if normalized_shape == self.active_input_shape:
            return
        artifacts = resolve_ncnn_detection_artifacts(
            self.source_model_path,
            imgsz=normalized_shape,
            prefer_fp16=self.fp16_requested,
        )
        self._configure_runtime(artifacts)
        self._warmup(self.active_input_shape)

    def _load_net(self, *, param_path: Path, bin_path: Path, fp16: bool):
        """Create and initialize one NCNN Vulkan net."""
        net = self.ncnn.Net()
        if hasattr(net, "set_vulkan_device"):
            set_ncnn_vulkan_device(
                net,
                self.device_index,
                ncnn_module=self.ncnn,
            )
        net.opt.use_vulkan_compute = True
        net.opt.use_fp16_storage = fp16
        net.opt.use_fp16_packed = fp16
        net.opt.use_fp16_arithmetic = fp16
        register_lada_custom_layers(net, ncnn_module=self.ncnn)

        if net.load_param(str(param_path)) != 0:
            raise RuntimeError(f"Failed to load ncnn param '{param_path}'.")
        if net.load_model(str(bin_path)) != 0:
            raise RuntimeError(f"Failed to load ncnn model '{bin_path}'.")
        return net

    @staticmethod
    def _to_numpy_image(image: Image | ImageTensor) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            if image.device.type != "cpu":
                raise RuntimeError("NCNN Vulkan detection preprocess expects CPU frames from the decoder.")
            return np.ascontiguousarray(image.numpy())
        return np.ascontiguousarray(image)

    def _letterbox_to_ncnn_mat(
        self,
        image: Image | ImageTensor,
        *,
        input_shape: tuple[int, int] | None = None,
    ):
        """Convert one CPU BGR frame into a native NCNN CHW fp32 input tensor."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(
                f"NCNN Vulkan detection preprocess expects HWC uint8 images, got {tuple(image.shape)}."
            )

        target_h, target_w = _normalize_imgsz(input_shape or self.active_input_shape)
        src_h, src_w = int(image.shape[0]), int(image.shape[1])
        ratio = min(target_h / src_h, target_w / src_w)
        resized_w = int(round(src_w * ratio))
        resized_h = int(round(src_h * ratio))

        pad_w = target_w - resized_w
        pad_h = target_h - resized_h
        pad_left = round(pad_w / 2 - 0.1)
        pad_right = round(pad_w / 2 + 0.1)
        pad_top = round(pad_h / 2 - 0.1)
        pad_bottom = round(pad_h / 2 + 0.1)

        image_np = self._to_numpy_image(image)
        mat = self.ncnn.Mat.from_pixels_resize(
            image_np,
            self.ncnn.Mat.PixelType.PIXEL_BGR,
            src_w,
            src_h,
            resized_w,
            resized_h,
        )
        if any(pad > 0 for pad in (pad_top, pad_bottom, pad_left, pad_right)):
            mat = self.ncnn.copy_make_border(
                mat,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                0,
                114.0,
            )
        mat.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
        return mat

    def _letterbox_to_vulkan_tensor(
        self,
        image: Image | ImageTensor,
        *,
        input_shape: tuple[int, int] | None = None,
    ):
        """Convert one CPU BGR frame into a native Vulkan tensor for the detector input."""
        if self.gpu_runner is None or not hasattr(self.gpu_runner, "preprocess_bgr_u8"):
            raise RuntimeError("NCNN Vulkan detection preprocess runner is unavailable.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(
                f"NCNN Vulkan detection preprocess expects HWC uint8 images, got {tuple(image.shape)}."
            )

        image_np = self._to_numpy_image(image)
        target_h, target_w = _normalize_imgsz(input_shape or self.active_input_shape)
        return self.gpu_runner.preprocess_bgr_u8(
            image_np,
            [int(target_h), int(target_w)],
        )

    def preprocess(self, imgs: list[Image | ImageTensor]) -> list[Any]:
        """Preprocess one batch through native NCNN image operators instead of Python LetterBox."""
        with self._measure_profile("detector_preprocess_total_s"):
            input_shape = self._resolve_runtime_input_shape(imgs)
            self._ensure_runtime_for_input_shape(input_shape)
            with self._measure_profile("detector_preprocess_letterbox_s"):
                return [
                    self._letterbox_to_vulkan_tensor(image, input_shape=input_shape)
                    for image in imgs
                ]

    def _to_ncnn_input_mat(self, input_frame: Any):
        """Normalize one prepared input into an NCNN Mat."""
        tensor_type = getattr(self.ncnn, "LadaVulkanTensor", None)
        if tensor_type is not None and isinstance(input_frame, tensor_type):
            return input_frame
        if isinstance(input_frame, self.ncnn.Mat):
            return input_frame
        return self.ncnn.Mat(
            np.ascontiguousarray(input_frame.detach().cpu().numpy().astype(np.float32, copy=False))
        )

    def _run_raw_outputs_to_cpu(self, input_frame: Any) -> tuple[np.ndarray, np.ndarray]:
        """Run the detector and return raw `(pred, proto)` tensors through the native runner."""
        if self.gpu_runner is None or not hasattr(self.gpu_runner, "run_many_to_cpu"):
            raise RuntimeError("NCNN Vulkan detection requires the local multi-output Vulkan runner.")

        outputs = self.gpu_runner.run_many_to_cpu(
            {self.input_name: self._to_ncnn_input_mat(input_frame)},
            list(self.output_names),
        )
        pred = next((np.asarray(value) for value in outputs.values() if np.asarray(value).ndim == 2), None)
        proto = next((np.asarray(value) for value in outputs.values() if np.asarray(value).ndim == 3), None)
        if pred is None or proto is None:
            output_shapes = {name: np.asarray(value).shape for name, value in outputs.items()}
            raise RuntimeError(f"Unexpected NCNN segmentation outputs: {output_shapes}")
        return pred, proto

    def _run_fused_subnet(
        self,
        input_frame: Any,
        orig_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Run the detector and fused Vulkan postprocess subnet for one frame."""
        if self.gpu_runner is None or self.postprocess_runner is None:
            raise RuntimeError("NCNN Vulkan detection requires the fused YOLO postprocess subnet.")
        if not hasattr(self.gpu_runner, "run_yolo_segmentation_subnet"):
            raise RuntimeError("NCNN Vulkan detection runtime is missing the fused subnet runner.")

        common_args = (
            {self.input_name: self._to_ncnn_input_mat(input_frame)},
            list(self.output_names),
            self.postprocess_runner,
            [int(self.active_input_shape[0]), int(self.active_input_shape[1])],
            [int(orig_shape[0]), int(orig_shape[1])],
            float(self.args.conf),
            float(self.args.iou),
            int(self.args.max_det),
            len(self.names),
            bool(self.args.agnostic_nms),
        )
        try:
            processed = self.gpu_runner.run_yolo_segmentation_subnet(
                *common_args,
                list(self.args.classes) if self.args.classes is not None else [],
                int(self.max_nms),
            )
        except TypeError as exc:
            if (
                "incompatible function arguments" not in str(exc)
                and "unexpected keyword argument" not in str(exc)
            ):
                raise
            processed = self.gpu_runner.run_yolo_segmentation_subnet(*common_args)
        return {
            "boxes": np.asarray(processed["boxes"], dtype=np.float32),
            "masks": np.asarray(processed["masks"], dtype=np.uint8),
        }

    def _run_native_fused_subnet_batch(
        self,
        input_batch: list[Any],
        orig_shapes: list[tuple[int, int]],
    ) -> list[dict[str, np.ndarray]]:
        """Dispatch one batch through the native Vulkan fused-subnet batch binding."""
        if len(input_batch) != len(orig_shapes):
            raise RuntimeError("YOLO segmentation batch requires one original shape per input frame.")
        if self.gpu_runner is None or self.postprocess_runner is None:
            raise RuntimeError("NCNN Vulkan detection requires the fused YOLO postprocess subnet.")
        if not hasattr(self.gpu_runner, "run_yolo_segmentation_subnet_batch"):
            raise RuntimeError("NCNN Vulkan detection runtime is missing the fused subnet batch runner.")

        common_args = (
            [self._to_ncnn_input_mat(input_frame) for input_frame in input_batch],
            self.input_name,
            list(self.output_names),
            self.postprocess_runner,
            [int(self.active_input_shape[0]), int(self.active_input_shape[1])],
            [[int(orig_shape[0]), int(orig_shape[1])] for orig_shape in orig_shapes],
            float(self.args.conf),
            float(self.args.iou),
            int(self.args.max_det),
            len(self.names),
            bool(self.args.agnostic_nms),
        )
        try:
            processed_batch = self.gpu_runner.run_yolo_segmentation_subnet_batch(
                *common_args,
                list(self.args.classes) if self.args.classes is not None else [],
                int(self.max_nms),
            )
        except TypeError as exc:
            if (
                "incompatible function arguments" not in str(exc)
                and "unexpected keyword argument" not in str(exc)
            ):
                raise
            processed_batch = self.gpu_runner.run_yolo_segmentation_subnet_batch(
                *common_args
            )
        return [
            {
                "boxes": np.asarray(processed["boxes"], dtype=np.float32),
                "masks": np.asarray(processed["masks"], dtype=np.uint8),
            }
            for processed in processed_batch
        ]

    def _run_fused_subnet_batch(
        self,
        input_batch: list[Any],
        orig_shapes: list[tuple[int, int]],
    ) -> list[dict[str, np.ndarray]]:
        """Run one detector batch through the fused Vulkan postprocess subnet."""
        if len(input_batch) != len(orig_shapes):
            raise RuntimeError("YOLO segmentation batch requires one original shape per input frame.")
        if hasattr(self.gpu_runner, "run_yolo_segmentation_subnet_batch"):
            return self._run_native_fused_subnet_batch(input_batch, orig_shapes)
        return [
            self._run_fused_subnet(input_frame, orig_shape)
            for input_frame, orig_shape in zip(input_batch, orig_shapes)
        ]

    def _warmup(self, input_shape: tuple[int, int] | None = None) -> None:
        """Verify that the selected NCNN artifacts can complete the fused Vulkan path."""
        warmup_shape = _normalize_imgsz(input_shape or self.active_input_shape)
        dummy = torch.zeros((3, *warmup_shape), dtype=torch.float32)
        self._run_raw_outputs_to_cpu(dummy)
        self._run_fused_subnet(dummy, warmup_shape)
        if hasattr(self.gpu_runner, "run_yolo_segmentation_subnet_batch"):
            self._run_fused_subnet_batch([dummy], [warmup_shape])

    def inference(self, image_batch: torch.Tensor):
        """Run one single-frame batch through NCNN and return raw `(pred, proto)` outputs."""
        if image_batch.ndim != 4 or image_batch.shape[0] != 1:
            raise RuntimeError(
                f"NCNN Vulkan detection expects a single-frame batch, got {tuple(image_batch.shape)}."
            )
        pred, proto = self._run_raw_outputs_to_cpu(image_batch[0])
        return [(torch.from_numpy(pred).unsqueeze(0), torch.from_numpy(proto).unsqueeze(0)), {}]

    def inference_and_postprocess(
        self,
        imgs: Any,
        orig_imgs: list[Image | ImageTensor],
    ) -> list[DetectionResult]:
        """Run batched detection through the fused NCNN Vulkan detector/postprocess path."""
        with torch.inference_mode():
            with self._measure_profile("detector_inference_total_s"):
                input_batch = imgs if isinstance(imgs, list) else self.prepare_input(imgs)
                orig_shapes = [
                    (int(orig_img.shape[0]), int(orig_img.shape[1]))
                    for orig_img in orig_imgs
                ]
                with self._measure_profile("detector_subnet_fused_s"):
                    processed_batch = self._run_fused_subnet_batch(input_batch, orig_shapes)
                results: list[DetectionResult] = []
                for processed, orig_img in zip(processed_batch, orig_imgs):
                    with self._measure_profile("detector_result_build_s"):
                        results.append(
                            build_native_yolo_result(
                                orig_img=orig_img,
                                names=self.names,
                                boxes=processed["boxes"],
                                masks=processed["masks"],
                            )
                        )
                return results
