# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from .device import configure_ncnn_vulkan_device_info

_REPO_ROOT = Path(__file__).resolve().parents[4]
_KNOWN_LOCAL_RUNTIME_RELATIVE_PATHS = (
    (
        "native",
        "vulkan",
        "ncnn_runtime",
        "build",
        "local_runtime",
    ),
    (
        "native",
        "ncnn_vulkan_runtime",
        "build",
        "local_runtime",
    ),
)


def _looks_like_local_ncnn_runtime_dir(candidate: Path) -> bool:
    package_dir = candidate / "ncnn"
    if package_dir.is_dir() and (package_dir / "__init__.py").exists():
        return True

    for pattern in ("ncnn*.pyd", "ncnn*.so"):
        if any(candidate.glob(pattern)):
            return True

    return False


def _iter_local_ncnn_runtime_dirs() -> list[Path]:
    candidates: list[Path] = []

    runtime_env = os.environ.get("LADA_LOCAL_NCNN_RUNTIME_DIR")
    if runtime_env:
        candidates.append(Path(runtime_env))

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        for relative_path in _KNOWN_LOCAL_RUNTIME_RELATIVE_PATHS:
            candidates.append(Path(meipass).joinpath(*relative_path))

    executable_dir = Path(sys.executable).resolve().parent
    for relative_path in _KNOWN_LOCAL_RUNTIME_RELATIVE_PATHS:
        candidates.append(executable_dir / "_internal" / Path(*relative_path))
        candidates.append(executable_dir / Path(*relative_path))
        candidates.append(_REPO_ROOT.joinpath(*relative_path))

    ordered_existing: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen or not candidate.exists():
            continue
        if not _looks_like_local_ncnn_runtime_dir(candidate):
            continue
        seen.add(resolved)
        ordered_existing.append(candidate)
    return ordered_existing


def _get_local_ncnn_runtime_dir() -> Path:
    runtime_dirs = _iter_local_ncnn_runtime_dirs()
    if runtime_dirs:
        return runtime_dirs[0]
    return _REPO_ROOT.joinpath(*_KNOWN_LOCAL_RUNTIME_RELATIVE_PATHS[0])


def import_ncnn_module(*, prefer_local_runtime: bool = True) -> Any:
    """Import the ncnn Python module, preferring the locally built runtime when available."""
    configure_ncnn_vulkan_device_info()
    if prefer_local_runtime and "ncnn" not in sys.modules:
        for local_runtime_dir in reversed(_iter_local_ncnn_runtime_dirs()):
            runtime_dir_str = str(local_runtime_dir)
            if runtime_dir_str not in sys.path:
                sys.path.insert(0, runtime_dir_str)

    return importlib.import_module("ncnn")
