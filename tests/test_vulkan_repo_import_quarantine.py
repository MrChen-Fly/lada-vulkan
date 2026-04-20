from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "lada"
_VULKAN_EXTENSION_ROOT = _PACKAGE_ROOT / "extensions" / "vulkan"
_BASICVSRPP_ROOT = _PACKAGE_ROOT / "models" / "basicvsrpp"
_YOLO_ROOT = _PACKAGE_ROOT / "models" / "yolo"
_RESTORATIONPIPELINE_ROOT = _PACKAGE_ROOT / "restorationpipeline"
_FORBIDDEN_TOP_LEVEL_FILES = (
    "compute_targets.py",
)
_FORBIDDEN_PREFIXES = (
    "lada.models.basicvsrpp.ncnn_vulkan",
    "lada.models.basicvsrpp.vulkan_",
    "lada.models.yolo.ncnn_vulkan",
    "lada.restorationpipeline.basicvsrpp_vulkan_",
    "lada.extensions.vulkan.basicvsrpp_",
    "lada.extensions.vulkan.yolo_",
    "lada.extensions.vulkan.basicvsrpp_ncnn_runtime",
    "lada.extensions.vulkan.clip_resize_semantics",
    "lada.extensions.vulkan.runtime_options",
    "lada.extensions.vulkan.runtime_profiling",
)
_FORBIDDEN_BASICVSRPP_FILES = (
    "ncnn_vulkan.py",
    "vulkan_param_patch.py",
    "vulkan_runtime.py",
    "vulkan_runtime_align_chains.py",
    "vulkan_runtime_chains.py",
    "vulkan_runtime_clip.py",
    "vulkan_runtime_propagate_chains.py",
    "vulkan_runtime_propagate_with_flow_chains.py",
)
_FORBIDDEN_YOLO_FILES = (
    "ncnn_vulkan.py",
    "ncnn_vulkan_runtime_support.py",
    "runtime_model_base.py",
    "runtime_results.py",
)
_FORBIDDEN_RESTORATIONPIPELINE_FILES = (
    "basicvsrpp_vulkan_artifacts.py",
    "basicvsrpp_vulkan_blend.py",
    "basicvsrpp_vulkan_common.py",
    "basicvsrpp_vulkan_cpu_extractor.py",
    "basicvsrpp_vulkan_export.py",
    "basicvsrpp_vulkan_io.py",
    "basicvsrpp_vulkan_recurrent_runtime.py",
    "basicvsrpp_vulkan_restorer.py",
    "basicvsrpp_vulkan_restore_paths.py",
    "basicvsrpp_vulkan_runtime_bootstrap.py",
    "basicvsrpp_vulkan_runtime_support.py",
    "clip_resize_semantics.py",
    "runtime_options.py",
    "runtime_profiling.py",
)
_FORBIDDEN_VULKAN_TOP_LEVEL_FILES = (
    "basicvsrpp_artifacts.py",
    "basicvsrpp_blend.py",
    "basicvsrpp_common.py",
    "basicvsrpp_cpu_extractor.py",
    "basicvsrpp_export.py",
    "basicvsrpp_io.py",
    "basicvsrpp_ncnn_runtime.py",
    "basicvsrpp_recurrent_runtime.py",
    "basicvsrpp_restorer.py",
    "basicvsrpp_restore_paths.py",
    "basicvsrpp_runtime_bootstrap.py",
    "basicvsrpp_runtime_support.py",
    "basicvsrpp_vulkan_param_patch.py",
    "basicvsrpp_vulkan_runtime.py",
    "basicvsrpp_vulkan_runtime_align_chains.py",
    "basicvsrpp_vulkan_runtime_chains.py",
    "basicvsrpp_vulkan_runtime_clip.py",
    "basicvsrpp_vulkan_runtime_core.py",
    "basicvsrpp_vulkan_runtime_heads.py",
    "basicvsrpp_vulkan_runtime_propagate_chains.py",
    "basicvsrpp_vulkan_runtime_propagate_with_flow_chains.py",
    "clip_resize_semantics.py",
    "runtime_options.py",
    "runtime_profiling.py",
    "yolo_ncnn_runtime.py",
    "yolo_runtime_model_base.py",
    "yolo_runtime_results.py",
    "yolo_runtime_support.py",
)
_REQUIRED_VULKAN_FEATURE_FILES = (
    "basicvsrpp/__init__.py",
    "basicvsrpp/basicvsrpp_artifacts.py",
    "basicvsrpp/basicvsrpp_common.py",
    "basicvsrpp/basicvsrpp_restorer.py",
    "basicvsrpp/clip_resize_semantics.py",
    "basicvsrpp/runtime_options.py",
    "basicvsrpp/runtime_profiling.py",
    "yolo/__init__.py",
    "yolo/yolo_ncnn_runtime.py",
    "yolo/yolo_runtime_model_base.py",
    "yolo/yolo_runtime_results.py",
    "yolo/yolo_runtime_support.py",
    "ncnn/__init__.py",
    "ncnn/audit.py",
    "ncnn/capabilities.py",
    "ncnn/device.py",
    "ncnn/loader.py",
    "ncnn/runners.py",
    "ncnn_runtime.py",
)
_FORBIDDEN_NATIVE_RUNTIME_PATHS = (
    _REPO_ROOT / "native" / "ncnn_vulkan_runtime",
    _REPO_ROOT / "native" / "vulkan" / "ncnn_runtime" / "build-codex",
)
_REQUIRED_NATIVE_RUNTIME_FILES = (
    _REPO_ROOT / "native" / "vulkan" / "ncnn_runtime" / "CMakeLists.txt",
    _REPO_ROOT
    / "native"
    / "vulkan"
    / "ncnn_runtime"
    / "cmake"
    / "patch_ncnn_shared_vulkan_benchmark.cmake",
    _REPO_ROOT
    / "native"
    / "vulkan"
    / "ncnn_runtime"
    / "cmake"
    / "prepare_local_pyncnn_main.cmake",
    _REPO_ROOT
    / "native"
    / "vulkan"
    / "ncnn_runtime"
    / "src"
    / "python_local_runtime_bindings.cpp",
    _REPO_ROOT
    / "native"
    / "vulkan"
    / "ncnn_runtime"
    / "src"
    / "torchvision_deform_conv2d_layer.cpp",
    _REPO_ROOT
    / "native"
    / "vulkan"
    / "ncnn_runtime"
    / "src"
    / "lada_gridsample_layer.cpp",
    _REPO_ROOT
    / "native"
    / "vulkan"
    / "ncnn_runtime"
    / "src"
    / "lada_yolo_attention_layer.cpp",
    _REPO_ROOT
    / "native"
    / "vulkan"
    / "ncnn_runtime"
    / "src"
    / "lada_yolo_seg_postprocess_layer.cpp",
    _REPO_ROOT
    / "native"
    / "vulkan"
    / "ncnn_runtime"
    / "src"
    / "basicvsrpp_clip_runner.cpp",
    _REPO_ROOT
    / "native"
    / "vulkan"
    / "ncnn_runtime"
    / "src"
    / "vulkan_blend_runtime.cpp",
    _REPO_ROOT / "scripts" / "build_ncnn_vulkan_runtime.ps1",
)
_NATIVE_OP_PROFILE_HEADER = (
    _REPO_ROOT
    / "native"
    / "vulkan"
    / "ncnn_runtime"
    / "src"
    / "native_op_profile.h"
)


def _iter_import_targets(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets.append(node.module)
    return targets


def test_repo_production_code_keeps_legacy_vulkan_paths_quarantined() -> None:
    offenders: list[str] = []

    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for target in _iter_import_targets(path):
            if target.startswith(_FORBIDDEN_PREFIXES):
                offenders.append(f"{path.relative_to(_REPO_ROOT)} imports {target}")

    assert offenders == []


def test_basicvsrpp_model_package_does_not_keep_legacy_vulkan_files() -> None:
    offenders = [
        str((_BASICVSRPP_ROOT / file_name).relative_to(_REPO_ROOT))
        for file_name in _FORBIDDEN_BASICVSRPP_FILES
        if (_BASICVSRPP_ROOT / file_name).exists()
    ]

    assert offenders == []


def test_yolo_model_package_does_not_keep_legacy_vulkan_files() -> None:
    offenders = [
        str((_YOLO_ROOT / file_name).relative_to(_REPO_ROOT))
        for file_name in _FORBIDDEN_YOLO_FILES
        if (_YOLO_ROOT / file_name).exists()
    ]

    assert offenders == []


def test_restorationpipeline_package_does_not_keep_legacy_vulkan_files() -> None:
    offenders = [
        str((_RESTORATIONPIPELINE_ROOT / file_name).relative_to(_REPO_ROOT))
        for file_name in _FORBIDDEN_RESTORATIONPIPELINE_FILES
        if (_RESTORATIONPIPELINE_ROOT / file_name).exists()
    ]

    assert offenders == []


def test_repo_top_level_does_not_keep_extension_only_target_helpers() -> None:
    offenders = [
        str((_PACKAGE_ROOT / file_name).relative_to(_REPO_ROOT))
        for file_name in _FORBIDDEN_TOP_LEVEL_FILES
        if (_PACKAGE_ROOT / file_name).exists()
    ]

    assert offenders == []


def test_vulkan_extension_top_level_keeps_feature_modules_in_subpackages() -> None:
    offenders = [
        str((_VULKAN_EXTENSION_ROOT / file_name).relative_to(_REPO_ROOT))
        for file_name in _FORBIDDEN_VULKAN_TOP_LEVEL_FILES
        if (_VULKAN_EXTENSION_ROOT / file_name).exists()
    ]

    assert offenders == []


def test_vulkan_extension_feature_subpackages_exist() -> None:
    missing = [
        str((_VULKAN_EXTENSION_ROOT / relative_path).relative_to(_REPO_ROOT))
        for relative_path in _REQUIRED_VULKAN_FEATURE_FILES
        if not (_VULKAN_EXTENSION_ROOT / relative_path).exists()
    ]

    assert missing == []


def test_native_runtime_layout_keeps_single_repo_build_drop() -> None:
    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in _FORBIDDEN_NATIVE_RUNTIME_PATHS
        if path.exists()
    ]

    assert offenders == []


def test_native_runtime_source_tree_is_present() -> None:
    missing = [
        str(path.relative_to(_REPO_ROOT))
        for path in _REQUIRED_NATIVE_RUNTIME_FILES
        if not path.exists()
    ]

    assert missing == []


def test_native_runtime_build_script_stays_in_powershell() -> None:
    script_path = _REPO_ROOT / "scripts" / "build_ncnn_vulkan_runtime.ps1"
    source = script_path.read_text(encoding="utf-8")

    assert "cmd /c" not in source
    assert "cmd.exe" not in source


def test_native_runtime_gpu_profile_header_does_not_fall_back_to_void_pointers() -> None:
    source = _NATIVE_OP_PROFILE_HEADER.read_text(encoding="utf-8")

    assert "ScopedNativeOpGpuTimestampQuery(NativeOpKind kind, void* cmd = nullptr)" not in source
    assert "inline void finalize_native_op_gpu_profile(void* cmd, const void* vkdev)" not in source
