# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any

_VULKAN_DEVICE_INFO_ENV = "LADA_NCNN_SHOW_VULKAN_DEVICE_INFO"
_ENV_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})
_NCNN_GPU_INSTANCE_READY = False


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _ENV_FALSE_VALUES


def configure_ncnn_vulkan_device_info(*, show: bool | None = None) -> bool:
    """Configure whether the local ncnn runtime may print Vulkan device info."""
    if show is None:
        os.environ.setdefault(_VULKAN_DEVICE_INFO_ENV, "0")
    else:
        os.environ[_VULKAN_DEVICE_INFO_ENV] = "1" if show else "0"
    return _env_flag_enabled(_VULKAN_DEVICE_INFO_ENV)


@contextmanager
def _maybe_silence_vulkan_device_info():
    if configure_ncnn_vulkan_device_info():
        yield
        return

    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return

    try:
        saved_stderr_fd = os.dup(stderr_fd)
    except OSError:
        yield
        return

    try:
        sys.stderr.flush()
    except (AttributeError, OSError, ValueError):
        pass

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


def ensure_ncnn_gpu_instance(ncnn_module: Any) -> None:
    """Initialize the Vulkan instance once, silencing one-time device info by default."""
    global _NCNN_GPU_INSTANCE_READY
    if _NCNN_GPU_INSTANCE_READY:
        return

    create_gpu_instance = getattr(ncnn_module, "create_gpu_instance", None)
    if not callable(create_gpu_instance):
        return

    with _maybe_silence_vulkan_device_info():
        result = create_gpu_instance()
    if isinstance(result, int) and result < 0:
        return
    _NCNN_GPU_INSTANCE_READY = True


def get_ncnn_gpu_count(ncnn_module: Any) -> int:
    """Return the number of visible Vulkan devices after ensuring the runtime is ready."""
    ensure_ncnn_gpu_instance(ncnn_module)
    return int(ncnn_module.get_gpu_count())


def set_ncnn_vulkan_device(net: Any, device_index: int, *, ncnn_module: Any) -> None:
    """Bind one ncnn.Net to a Vulkan device without printing device info by default."""
    ensure_ncnn_gpu_instance(ncnn_module)
    net.set_vulkan_device(int(device_index))
