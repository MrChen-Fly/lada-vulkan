from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from lada.compute_targets import UnsupportedComputeTargetError

from .artifacts import (
    IreeDetectionArtifacts,
    discover_iree_entry_function_from_mlir,
    update_iree_artifact_entry_function,
)


def _tool_candidates(name: str) -> tuple[str, ...]:
    exe_name = f"{name}.exe" if sys.platform == "win32" else name
    scripts_dir = Path(sys.executable).resolve().parent
    return (str(scripts_dir / exe_name), exe_name, name)


def _resolve_iree_tool(name: str) -> str:
    for candidate in _tool_candidates(name):
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise UnsupportedComputeTargetError(
        f"Unable to locate the IREE tool '{name}'. Ensure the current .venv has the compiler tools installed."
    )


def _run_iree_tool(argv: list[str]) -> None:
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode == 0:
        return
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    details = stderr or stdout or f"exit code {completed.returncode}"
    raise UnsupportedComputeTargetError(
        f"IREE tool invocation failed: {' '.join(argv)}. Details: {details}"
    )


def _needs_refresh(output_path: Path, *input_paths: Path) -> bool:
    if not output_path.exists():
        return True
    output_mtime = output_path.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > output_mtime for path in input_paths)


def ensure_iree_detection_vmfb(artifacts: IreeDetectionArtifacts) -> IreeDetectionArtifacts:
    """Materialize the MLIR + VMFB files for one IREE detection artifact bundle."""
    artifacts.model_dir.mkdir(parents=True, exist_ok=True)
    if _needs_refresh(artifacts.mlir_path, artifacts.onnx_path):
        _run_iree_tool(
            [
                _resolve_iree_tool("iree-import-onnx"),
                str(artifacts.onnx_path),
                "-o",
                str(artifacts.mlir_path),
            ]
        )
    if _needs_refresh(artifacts.vmfb_path, artifacts.mlir_path):
        _run_iree_tool(
            [
                _resolve_iree_tool("iree-compile"),
                str(artifacts.mlir_path),
                "--iree-hal-target-backends=vulkan-spirv",
                "-o",
                str(artifacts.vmfb_path),
            ]
        )
    entry_function = discover_iree_entry_function_from_mlir(artifacts.mlir_path)
    if entry_function:
        update_iree_artifact_entry_function(artifacts.model_dir, entry_function)
        if entry_function != artifacts.entry_function:
            return replace(artifacts, entry_function=entry_function)
    return artifacts
