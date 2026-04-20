from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import lada.extensions.vulkan.ncnn as ncnn_runtime
from lada.extensions.vulkan import runtime


def test_vulkan_target_uses_runtime_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "probe_ncnn_vulkan_runtime",
        lambda: SimpleNamespace(
            available=True,
            devices=("NCNN Vulkan GPU",),
            runtime_importable=True,
            vulkan_loader_available=True,
            custom_runtime_supported=True,
            error=None,
        ),
    )

    target = runtime.get_vulkan_target()

    assert target.id == "vulkan:0"
    assert target.runtime == "vulkan"
    assert target.available is True
    assert target.torch_device is None
    assert target.description == "Vulkan (NCNN) - NCNN Vulkan GPU"


def test_vulkan_device_helpers_normalize_public_entry() -> None:
    assert runtime.normalize_vulkan_device("vulkan:0") == "vulkan:0"
    assert runtime.normalize_vulkan_device("vulkan") == "vulkan:0"
    assert runtime.is_vulkan_device("vulkan") is True
    assert runtime.is_vulkan_device("vulkan:0") is True
    assert runtime.is_vulkan_device("cuda:0") is False


def test_vulkan_compute_targets_expose_single_public_entry(monkeypatch) -> None:
    target = runtime.ComputeTarget(
        id="vulkan:0",
        description="Vulkan (NCNN)",
        runtime="vulkan",
        available=False,
        torch_device=None,
    )
    monkeypatch.setattr(runtime, "get_vulkan_target", lambda: target)

    assert runtime.get_vulkan_compute_targets() == [target]


def test_vulkan_probe_reports_official_wheel_when_local_runtime_dirs_are_missing(
    monkeypatch,
) -> None:
    fake_ncnn_module = SimpleNamespace(
        __file__=r"D:\Code\github\lada-vulkan\.venv\Lib\site-packages\ncnn\__init__.py"
    )

    monkeypatch.setattr(runtime, "_has_vulkan_loader", lambda: True)
    monkeypatch.setattr(runtime, "import_ncnn_module", lambda: fake_ncnn_module)
    monkeypatch.setattr(runtime, "_iter_local_ncnn_runtime_dirs", lambda: [])
    monkeypatch.setattr(runtime, "get_ncnn_gpu_count", lambda _module: 1)
    monkeypatch.setattr(runtime, "_resolve_device_name", lambda _module, _index: "NCNN GPU")
    monkeypatch.setattr(runtime, "ncnn_has_lada_custom_layer", lambda _module: False)
    monkeypatch.setattr(runtime, "ncnn_has_lada_gridsample_layer", lambda _module: False)
    monkeypatch.setattr(runtime, "ncnn_has_lada_yolo_attention_layer", lambda _module: False)
    monkeypatch.setattr(
        runtime,
        "ncnn_has_lada_yolo_seg_postprocess_vulkan_layer",
        lambda _module: False,
    )
    monkeypatch.setattr(runtime, "ncnn_has_lada_vulkan_net_runner", lambda _module: False)
    monkeypatch.setattr(
        runtime,
        "ncnn_has_lada_basicvsrpp_clip_runner",
        lambda _module: False,
    )

    probe = runtime.probe_ncnn_vulkan_runtime()

    assert probe.available is False
    assert probe.runtime_importable is True
    assert probe.devices == ("NCNN GPU",)
    assert "Imported ncnn from" in str(probe.error)
    assert "site-packages\\ncnn\\__init__.py" in str(probe.error)
    assert "No local ncnn runtime directories were found." in str(probe.error)


def test_local_runtime_dir_scan_ignores_placeholder_dirs(
    monkeypatch, tmp_path: Path
) -> None:
    placeholder_dir = tmp_path / "local_runtime"
    placeholder_dir.mkdir(parents=True)

    monkeypatch.setenv("LADA_LOCAL_NCNN_RUNTIME_DIR", str(placeholder_dir))

    detected_dirs = ncnn_runtime._iter_local_ncnn_runtime_dirs()
    assert placeholder_dir not in detected_dirs

    package_dir = placeholder_dir / "ncnn"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("# test runtime\n", encoding="utf-8")

    detected_dirs = ncnn_runtime._iter_local_ncnn_runtime_dirs()
    assert detected_dirs[0] == placeholder_dir
