from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import yaml
from ultralytics import YOLO

from lada.compute_targets import UnsupportedComputeTargetError

from .yolo_support import (
    coerce_names,
    get_iree_precision_artifact_dir,
    normalize_runtime_imgsz,
)


@dataclass(frozen=True)
class IreeDetectionArtifacts:
    """Describe one exported IREE detection artifact bundle."""

    model_dir: Path
    onnx_path: Path
    mlir_path: Path
    vmfb_path: Path
    metadata_path: Path
    names: dict[int, str]
    base_overrides: dict[str, Any]
    imgsz: tuple[int, int]
    fp16: bool
    entry_function: str = "main"


_ENTRY_FUNCTION_PATTERN = re.compile(r"func\.func\s+@([^\s(]+)\(")


def _artifact_paths(model_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        model_dir / "model.onnx",
        model_dir / "model.mlir",
        model_dir / "model.vmfb",
        model_dir / "metadata.yaml",
    )


def _load_artifact_metadata(model_dir: Path) -> dict[str, Any] | None:
    metadata_path = model_dir / "metadata.yaml"
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle) or {}
    except Exception:
        return None
    return metadata if isinstance(metadata, Mapping) else None


def _dump_artifact_metadata(metadata_path: Path, metadata: Mapping[str, Any]) -> None:
    metadata_path.write_text(
        yaml.safe_dump(dict(metadata), sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def discover_iree_entry_function_from_mlir(mlir_path: Path) -> str | None:
    if not mlir_path.exists():
        return None
    try:
        mlir_text = mlir_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = _ENTRY_FUNCTION_PATTERN.search(mlir_text)
    if not match:
        return None
    return match.group(1)


def resolve_iree_entry_function(model_dir: Path, metadata: Mapping[str, Any] | None = None) -> str:
    resolved_metadata = metadata if metadata is not None else _load_artifact_metadata(model_dir) or {}
    mlir_entry = discover_iree_entry_function_from_mlir(model_dir / "model.mlir")
    configured_entry = str(resolved_metadata.get("entry_function", "")).strip()
    if mlir_entry:
        return mlir_entry
    if configured_entry:
        return configured_entry
    return "main"


def update_iree_artifact_entry_function(model_dir: Path, entry_function: str) -> None:
    metadata = _load_artifact_metadata(model_dir)
    if metadata is None:
        return
    normalized_entry = str(entry_function).strip()
    if not normalized_entry or metadata.get("entry_function") == normalized_entry:
        return
    updated_metadata = dict(metadata)
    updated_metadata["entry_function"] = normalized_entry
    _dump_artifact_metadata(model_dir / "metadata.yaml", updated_metadata)


def _build_artifacts(model_dir: Path) -> IreeDetectionArtifacts:
    onnx_path, mlir_path, vmfb_path, metadata_path = _artifact_paths(model_dir)
    metadata = _load_artifact_metadata(model_dir)
    if metadata is None:
        raise UnsupportedComputeTargetError(
            f"IREE artifact directory '{model_dir}' is missing metadata.yaml."
        )
    if not onnx_path.exists():
        raise UnsupportedComputeTargetError(
            f"IREE detection artifacts are incomplete in '{model_dir}': missing model.onnx."
        )
    imgsz = tuple(int(value) for value in metadata.get("imgsz", []))
    if len(imgsz) != 2:
        raise UnsupportedComputeTargetError(
            f"IREE detection metadata in '{model_dir}' is missing imgsz."
        )
    return IreeDetectionArtifacts(
        model_dir=model_dir,
        onnx_path=onnx_path,
        mlir_path=mlir_path,
        vmfb_path=vmfb_path,
        metadata_path=metadata_path,
        names=coerce_names(metadata.get("names")),
        base_overrides=dict(metadata.get("base_overrides", {})),
        imgsz=(int(imgsz[0]), int(imgsz[1])),
        fp16=bool(metadata.get("fp16", False)),
        entry_function=resolve_iree_entry_function(model_dir, metadata),
    )


def _artifacts_match(model_dir: Path, *, imgsz: tuple[int, int], fp16: bool) -> bool:
    metadata = _load_artifact_metadata(model_dir)
    if metadata is None:
        return False
    metadata_imgsz = tuple(int(value) for value in metadata.get("imgsz", []))
    return (
        metadata.get("task") == "segment"
        and metadata_imgsz == imgsz
        and bool(metadata.get("fp16", False)) == fp16
        and (model_dir / "model.onnx").exists()
    )


def _resolve_existing_artifacts(
    model_path: Path,
    *,
    imgsz: tuple[int, int],
    fp16: bool,
) -> IreeDetectionArtifacts | None:
    candidate_dir = get_iree_precision_artifact_dir(model_path, fp16=fp16, imgsz=imgsz)
    if _artifacts_match(candidate_dir, imgsz=imgsz, fp16=fp16):
        return _build_artifacts(candidate_dir)
    return None


def _write_metadata(
    *,
    model_dir: Path,
    imgsz: tuple[int, int],
    fp16: bool,
    names: dict[int, str],
    base_overrides: Mapping[str, Any],
) -> None:
    _, _, _, metadata_path = _artifact_paths(model_dir)
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "task": "segment",
                "imgsz": list(imgsz),
                "fp16": bool(fp16),
                "names": {int(key): str(value) for key, value in names.items()},
                "base_overrides": dict(base_overrides),
                "entry_function": "main",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )


def _stage_exported_onnx(onnx_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "model.onnx"
    if onnx_path.resolve() != target_path.resolve():
        shutil.copy2(onnx_path, target_path)
    return target_path


def _export_iree_onnx_artifacts(
    model_path: Path,
    *,
    imgsz: tuple[int, int],
    fp16: bool,
) -> IreeDetectionArtifacts:
    target_dir = get_iree_precision_artifact_dir(model_path, fp16=fp16, imgsz=imgsz)
    yolo_model = YOLO(str(model_path))
    if yolo_model.task != "segment":
        raise UnsupportedComputeTargetError(
            f"IREE Vulkan detection requires a segmentation model, got task '{yolo_model.task}'."
        )

    exported_onnx = Path(
        yolo_model.export(
            format="onnx",
            imgsz=list(imgsz),
            half=fp16,
            simplify=False,
            dynamic=False,
            device="cpu",
        )
    )
    _stage_exported_onnx(exported_onnx, target_dir)
    _write_metadata(
        model_dir=target_dir,
        imgsz=imgsz,
        fp16=fp16,
        names={int(key): str(value) for key, value in dict(yolo_model.names).items()},
        base_overrides=dict(yolo_model.overrides),
    )
    return _build_artifacts(target_dir)


def resolve_iree_detection_artifacts(
    model_path: str,
    *,
    imgsz: int | tuple[int, int] | list[int],
    prefer_fp16: bool,
) -> IreeDetectionArtifacts:
    """Resolve or export IREE-ready ONNX artifacts for one YOLO segmentation model."""
    requested_imgsz = normalize_runtime_imgsz(imgsz)
    path = Path(model_path)
    if not path.exists():
        raise UnsupportedComputeTargetError(f"Detection model '{model_path}' does not exist.")

    if path.is_dir():
        metadata = _load_artifact_metadata(path)
        if metadata is None:
            raise UnsupportedComputeTargetError(
                f"IREE artifact directory '{path}' is missing metadata.yaml."
            )
        metadata_imgsz = tuple(int(value) for value in metadata.get("imgsz", []))
        if metadata.get("task") != "segment":
            raise UnsupportedComputeTargetError(
                f"IREE artifact directory '{path}' is not a segmentation model."
            )
        if metadata_imgsz != requested_imgsz:
            raise UnsupportedComputeTargetError(
                f"IREE artifact '{path}' was prepared for imgsz={metadata_imgsz}, "
                f"but the pipeline requested imgsz={requested_imgsz}."
            )
        return _build_artifacts(path)

    if path.suffix.lower() != ".pt":
        raise UnsupportedComputeTargetError(
            "IREE Vulkan detection expects a `.pt` model or a pre-exported IREE artifact directory."
        )

    fp16 = bool(prefer_fp16)
    existing = _resolve_existing_artifacts(path, imgsz=requested_imgsz, fp16=fp16)
    if existing is not None:
        return existing
    try:
        return _export_iree_onnx_artifacts(path, imgsz=requested_imgsz, fp16=fp16)
    except Exception as exc:
        raise UnsupportedComputeTargetError(
            f"Failed to prepare IREE detection artifacts for '{model_path}' with fp16={fp16}."
        ) from exc
