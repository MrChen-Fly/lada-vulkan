from __future__ import annotations

import numpy as np
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lada import ModelFiles


def _require_real_vulkan_smoke_inputs() -> tuple[str, str]:
    import lada.extensions.vulkan.runtime as vulkan_runtime

    probe = vulkan_runtime.probe_ncnn_vulkan_runtime()
    if not probe.available:
        pytest.skip(f"real Vulkan smoke requires an available local runtime: {probe}")

    detection_model = ModelFiles.get_detection_model_by_name("v4-fast")
    if detection_model is None:
        pytest.skip("real Vulkan smoke requires model_weights/lada_mosaic_detection_model_v4_fast.pt")

    restoration_model = ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2")
    if restoration_model is None:
        pytest.skip(
            "real Vulkan smoke requires model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"
        )

    return detection_model.path, restoration_model.path


def test_vulkan_runtime_keeps_native_clip_runner_available_but_disabled_by_default(
    monkeypatch,
) -> None:
    from lada.extensions.vulkan.basicvsrpp import basicvsrpp_runtime_bootstrap as bootstrap

    module_names = set(bootstrap._RECURRENT_RUNTIME_MODULES) | set(
        bootstrap._RECURRENT_CLIP_RUNNER_MODULES
    )
    module_names.add("quarter_downsample")
    fake_artifacts = {
        name: SimpleNamespace(param_path=f"{name}.param", bin_path=f"{name}.bin")
        for name in module_names
    }
    fake_ncnn = SimpleNamespace()
    fake_clip_runner = SimpleNamespace(supports_resized_bgr_u8_input=True)

    monkeypatch.setattr(
        bootstrap,
        "ensure_ncnn_basicvsrpp_modular_artifacts",
        lambda *args, **kwargs: fake_artifacts,
    )
    monkeypatch.setattr(
        bootstrap,
        "NcnnVulkanModuleRunner",
        lambda *args, **kwargs: SimpleNamespace(
            ncnn=fake_ncnn,
            gpu_runner=object(),
            layer_audit={},
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "summarize_ncnn_vulkan_audits",
        lambda audits: {"module_count": len(audits)},
    )
    monkeypatch.setattr(
        bootstrap,
        "ncnn_has_lada_basicvsrpp_clip_runner",
        lambda _ncnn: True,
    )
    monkeypatch.setattr(
        bootstrap,
        "NcnnVulkanBasicvsrppClipRunner",
        lambda *args, **kwargs: fake_clip_runner,
    )

    restorer = SimpleNamespace(
        fp16=False,
        frame_count=5,
        num_threads=1,
    )

    bootstrap.initialize_modular_runtime(
        restorer,
        "checkpoint.pth",
        config_path=None,
        artifacts_dir=None,
        frame_shape=(256, 256),
    )

    assert restorer.native_clip_runner is fake_clip_runner
    assert restorer.runtime_features.use_native_clip_runner is False
    assert restorer.runtime_features.supports_descriptor_restore is True


def test_load_models_keeps_torch_path_semantics(monkeypatch) -> None:
    import lada.restorationpipeline as restorationpipeline

    calls: dict[str, object] = {}

    class FakeDetectionModel:
        def __init__(self, model_path, device, **kwargs):
            calls["detection"] = (model_path, device, kwargs)

    class FakeRestorationModel:
        def __init__(self, model, device, fp16):
            calls["restoration"] = (model, device, fp16)

    fake_inference = ModuleType("lada.models.basicvsrpp.inference")

    def fake_load_model(config_path, model_path, device, fp16):
        calls["load_model"] = (config_path, model_path, device, fp16)
        return "torch-restoration-net"

    fake_inference.load_model = fake_load_model

    fake_restorer_module = ModuleType(
        "lada.restorationpipeline.basicvsrpp_mosaic_restorer"
    )
    fake_restorer_module.BasicvsrppMosaicRestorer = FakeRestorationModel

    monkeypatch.setitem(sys.modules, fake_inference.__name__, fake_inference)
    monkeypatch.setitem(sys.modules, fake_restorer_module.__name__, fake_restorer_module)
    monkeypatch.setattr(
        restorationpipeline,
        "Yolo11SegmentationModel",
        FakeDetectionModel,
    )

    detection_model, restoration_model, pad_mode = restorationpipeline.load_models(
        torch.device("cuda:0"),
        "basicvsrpp-v1.2",
        "restoration.pth",
        "restoration.py",
        "detection.pt",
        True,
        False,
    )

    assert isinstance(detection_model, FakeDetectionModel)
    assert isinstance(restoration_model, FakeRestorationModel)
    assert calls["load_model"] == (
        "restoration.py",
        "restoration.pth",
        torch.device("cuda:0"),
        True,
    )
    assert calls["detection"] == (
        "detection.pt",
        torch.device("cuda:0"),
        {"classes": None, "conf": 0.15, "fp16": True},
    )
    assert pad_mode == "zero"


def test_load_models_routes_vulkan_to_extension_builders(monkeypatch) -> None:
    import lada.restorationpipeline as restorationpipeline
    import lada.extensions.vulkan.pipeline as vulkan_pipeline
    from lada.extensions.vulkan.privateuse1 import (
        bootstrap_vulkan_privateuse1_backend,
        get_vulkan_privateuse1_device,
    )

    calls: dict[str, object] = {}

    def build_detection_model(**kwargs):
        calls["detection"] = kwargs
        return "vulkan-detection-model"

    def build_restoration_model(**kwargs):
        calls["restoration"] = kwargs
        return "vulkan-restoration-model", "zero"

    monkeypatch.setattr(vulkan_pipeline, "build_vulkan_detection_model", build_detection_model)
    monkeypatch.setattr(vulkan_pipeline, "build_vulkan_restoration_model", build_restoration_model)
    monkeypatch.setattr(
        vulkan_pipeline.ModelFiles,
        "get_detection_model_by_path",
        lambda _path: SimpleNamespace(name="v4-fast"),
    )
    bootstrap_vulkan_privateuse1_backend()

    detection_model, restoration_model, pad_mode = restorationpipeline.load_models(
        get_vulkan_privateuse1_device(),
        "basicvsrpp-v1.2",
        "restoration.pth",
        "restoration.py",
        "detection.pt",
        False,
        True,
    )

    assert detection_model == "vulkan-detection-model"
    assert restoration_model == "vulkan-restoration-model"
    assert pad_mode == "zero"
    assert calls["detection"] == {
        "model_path": "detection.pt",
        "compute_target_id": "vulkan:0",
        "classes": [0],
        "conf": 0.15,
        "fp16": False,
    }
    assert calls["restoration"] == {
        "model_name": "basicvsrpp-v1.2",
        "model_path": "restoration.pth",
        "config_path": "restoration.py",
        "compute_target_id": "vulkan:0",
        "torch_device": torch.device("cpu"),
        "fp16": False,
    }


def test_load_models_runs_real_vulkan_runtime_smoke_when_assets_exist() -> None:
    import lada.restorationpipeline as restorationpipeline
    from lada.extensions.vulkan.privateuse1 import (
        bootstrap_vulkan_privateuse1_backend,
        get_vulkan_privateuse1_device,
    )

    detection_model_path, restoration_model_path = _require_real_vulkan_smoke_inputs()
    bootstrap_vulkan_privateuse1_backend()

    detector, restorer, pad_mode = restorationpipeline.load_models(
        get_vulkan_privateuse1_device(),
        "basicvsrpp-v1.2",
        restoration_model_path,
        None,
        detection_model_path,
        False,
        False,
    )

    frames = [np.zeros((640, 640, 3), dtype=np.uint8)]
    prepared_frames = detector.preprocess(frames)
    detection_results = detector.inference_and_postprocess(prepared_frames, frames)

    clip = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(5)]
    restored_frames = restorer.restore(clip)

    assert pad_mode == "zero"
    assert len(detection_results) == 1
    assert tuple(detection_results[0].boxes.data.shape) == (0, 6)
    assert len(restored_frames) == 5
    assert tuple(restored_frames[0].shape) == (256, 256, 3)
    assert restored_frames[0].dtype == torch.uint8
