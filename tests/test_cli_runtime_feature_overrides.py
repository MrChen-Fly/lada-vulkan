import unittest

from lada.cli.runtime_feature_overrides import (
    apply_runtime_feature_overrides,
    parse_runtime_feature_overrides,
)
from lada.restorationpipeline.runtime_options import RestorationRuntimeFeatures


class _FakeRuntimeModel:
    def __init__(self) -> None:
        self.runtime_features = RestorationRuntimeFeatures(
            use_native_blob_bridge=False,
            use_native_recurrent_runtime=True,
        )


class RuntimeFeatureOverridesTest(unittest.TestCase):
    def test_parse_runtime_feature_overrides_accepts_implicit_true(self) -> None:
        overrides = parse_runtime_feature_overrides(
            ["use_native_blob_bridge", "use_native_frame_preprocess_batch=no"],
        )

        self.assertEqual(
            overrides,
            {
                "use_native_blob_bridge": True,
                "use_native_frame_preprocess_batch": False,
            },
        )

    def test_parse_runtime_feature_overrides_rejects_unknown_feature(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Unknown runtime feature override"):
            parse_runtime_feature_overrides(["not_a_real_feature=true"])

    def test_apply_runtime_feature_overrides_updates_runtime_features(self) -> None:
        model = _FakeRuntimeModel()

        updated = apply_runtime_feature_overrides(
            model,
            {
                "use_native_blob_bridge": True,
                "use_native_frame_preprocess": True,
            },
        )

        self.assertTrue(updated.use_native_blob_bridge)
        self.assertTrue(updated.use_native_frame_preprocess)
        self.assertTrue(model.runtime_features.use_native_recurrent_runtime)

    def test_apply_runtime_feature_overrides_requires_mutable_runtime_features(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not expose mutable runtime features"):
            apply_runtime_feature_overrides(object(), {"use_native_blob_bridge": True})


if __name__ == "__main__":
    unittest.main()
