from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTENSION_ROOT = _REPO_ROOT / "lada" / "extensions" / "vulkan"
_FORBIDDEN_PATTERNS = (
    "lada.models.basicvsrpp.vulkan_",
    "lada.models.yolo.ncnn_vulkan",
    "lada.restorationpipeline.basicvsrpp_vulkan_",
    "lada.restorationpipeline.clip_resize_semantics",
    "lada.restorationpipeline.runtime_options",
    "lada.restorationpipeline.runtime_profiling",
    "lada.extensions.vulkan.basicvsrpp_",
    "lada.extensions.vulkan.yolo_",
    "lada.extensions.vulkan.basicvsrpp_ncnn_runtime",
    "lada.extensions.vulkan.clip_resize_semantics",
    "lada.extensions.vulkan.runtime_options",
    "lada.extensions.vulkan.runtime_profiling",
)


def test_vulkan_extension_sources_do_not_reference_legacy_runtime_paths() -> None:
    offenders: list[str] = []

    for path in sorted(_EXTENSION_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in source:
                offenders.append(f"{path.relative_to(_REPO_ROOT)} contains {pattern!r}")

    assert offenders == []
