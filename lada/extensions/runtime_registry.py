from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

ComputeTargetProvider = Callable[[], list[Any]]
DetectionModelBuilder = Callable[..., Any]
RestorationModelBuilder = Callable[..., tuple[Any, str]]
DefaultFp16Resolver = Callable[[str], bool]
RuntimeDeviceInfoConfigurer = Callable[[bool | None], bool | None]


@dataclass(frozen=True)
class RuntimeExtension:
    runtime: str
    get_compute_targets: ComputeTargetProvider | None = None
    build_detection_model: DetectionModelBuilder | None = None
    build_restoration_model: RestorationModelBuilder | None = None
    default_fp16_enabled: DefaultFp16Resolver | None = None
    configure_device_info: RuntimeDeviceInfoConfigurer | None = None


_RUNTIME_EXTENSIONS: dict[str, RuntimeExtension] = {}
_BOOTSTRAPPED = False


def register_runtime_extension(extension: RuntimeExtension) -> None:
    _RUNTIME_EXTENSIONS[extension.runtime.lower()] = extension


def _bootstrap_runtime_extensions() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    from .vulkan import register_extension

    register_extension()
    _BOOTSTRAPPED = True


def get_runtime_extension(runtime: str | None) -> RuntimeExtension | None:
    if not runtime:
        return None
    _bootstrap_runtime_extensions()
    return _RUNTIME_EXTENSIONS.get(runtime.lower())


def iter_runtime_extensions() -> tuple[RuntimeExtension, ...]:
    _bootstrap_runtime_extensions()
    return tuple(_RUNTIME_EXTENSIONS.values())


def replace_runtime_extensions_for_tests(
    extensions: list[RuntimeExtension],
) -> None:
    """Replace the registry with a deterministic set of extensions for tests."""
    global _BOOTSTRAPPED
    _RUNTIME_EXTENSIONS.clear()
    for extension in extensions:
        register_runtime_extension(extension)
    _BOOTSTRAPPED = True


def reset_runtime_extensions_for_tests() -> None:
    """Restore the lazy bootstrap behavior after test-only overrides."""
    global _BOOTSTRAPPED
    _RUNTIME_EXTENSIONS.clear()
    _BOOTSTRAPPED = False
