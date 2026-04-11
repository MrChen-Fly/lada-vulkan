import unittest
from unittest import mock

from lada.compute_targets import (
    ComputeTarget,
    configure_compute_target_device_info,
    default_fp16_enabled_for_compute_target,
    get_compute_targets,
)
from lada.extensions.runtime_registry import (
    RuntimeExtension,
    replace_runtime_extensions_for_tests,
    reset_runtime_extensions_for_tests,
)
from lada.models.yolo.detection_backends import build_mosaic_detection_model
from lada.restorationpipeline.restoration_backends import (
    build_mosaic_restoration_model,
)


def _fake_target(runtime: str = "fake") -> ComputeTarget:
    return ComputeTarget(
        id=f"{runtime}:0",
        description=f"{runtime.title()} Runtime",
        runtime=runtime,
        available=True,
        torch_device=None,
        notes="provided by test extension",
        experimental=True,
    )


class RuntimeExtensionTests(unittest.TestCase):
    def tearDown(self):
        reset_runtime_extensions_for_tests()

    def test_compute_targets_include_registered_extension_target(self):
        replace_runtime_extensions_for_tests(
            [
                RuntimeExtension(
                    runtime="fake",
                    get_compute_targets=lambda: [_fake_target()],
                    default_fp16_enabled=lambda target_id: target_id == "fake:0",
                )
            ]
        )

        target_ids = [target.id for target in get_compute_targets(include_experimental=True)]

        self.assertIn("fake:0", target_ids)
        self.assertTrue(default_fp16_enabled_for_compute_target("fake:0"))

    def test_detection_backend_dispatches_to_extension_builder(self):
        sentinel = object()
        builder = mock.Mock(return_value=sentinel)
        replace_runtime_extensions_for_tests(
            [
                RuntimeExtension(
                    runtime="fake",
                    get_compute_targets=lambda: [_fake_target()],
                    build_detection_model=builder,
                )
            ]
        )

        result = build_mosaic_detection_model(
            "weights.pt",
            "fake:0",
            imgsz=512,
            fp16=True,
            conf=0.25,
        )

        self.assertIs(result, sentinel)
        builder.assert_called_once_with(
            model_path="weights.pt",
            compute_target_id="fake:0",
            imgsz=512,
            fp16=True,
            conf=0.25,
        )

    def test_restoration_backend_dispatches_to_extension_builder(self):
        sentinel_model = object()
        builder = mock.Mock(return_value=(sentinel_model, "zero"))
        replace_runtime_extensions_for_tests(
            [
                RuntimeExtension(
                    runtime="fake",
                    get_compute_targets=lambda: [_fake_target()],
                    build_restoration_model=builder,
                )
            ]
        )

        model, pad_mode = build_mosaic_restoration_model(
            "basicvsrpp-v1.2",
            "checkpoint.pth",
            None,
            "fake:0",
            torch_device=None,
            fp16=False,
            artifacts_dir="artifacts",
        )

        self.assertIs(model, sentinel_model)
        self.assertEqual(pad_mode, "zero")
        builder.assert_called_once_with(
            model_name="basicvsrpp-v1.2",
            model_path="checkpoint.pth",
            config_path=None,
            compute_target_id="fake:0",
            torch_device=None,
            fp16=False,
            artifacts_dir="artifacts",
        )

    def test_device_info_hook_uses_runtime_hint_without_probing_targets(self):
        configure_hook = mock.Mock(return_value=True)
        target_probe = mock.Mock(side_effect=AssertionError("compute target probe should not run"))
        replace_runtime_extensions_for_tests(
            [
                RuntimeExtension(
                    runtime="fake",
                    get_compute_targets=target_probe,
                    configure_device_info=configure_hook,
                )
            ]
        )

        result = configure_compute_target_device_info("fake:0", show=False)

        self.assertTrue(result)
        configure_hook.assert_called_once_with(False)
        target_probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
