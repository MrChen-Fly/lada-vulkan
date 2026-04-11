import unittest
from unittest import mock

import numpy as np
import torch

from lada.parity import _run_detection_parity
from lada.parity_report import build_probe
from lada.models.yolo.runtime_results import build_native_yolo_result


class BuildProbeTests(unittest.TestCase):
    def test_reports_same_shape_diff(self):
        probe = build_probe(
            "example",
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 2.5], [2.0, 4.0]], dtype=np.float32),
        )

        self.assertEqual(probe["name"], "example")
        self.assertEqual(probe["reference"]["shape"], [2, 2])
        self.assertEqual(probe["candidate"]["shape"], [2, 2])
        self.assertTrue(probe["diff"]["shape_match"])
        self.assertEqual(probe["diff"]["max_abs_diff"], 1.0)
        self.assertEqual(probe["diff"]["mean_abs_diff"], 0.375)

    def test_reports_shape_mismatch(self):
        probe = build_probe(
            "mismatch",
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        )

        self.assertFalse(probe["diff"]["shape_match"])
        self.assertEqual(probe["diff"]["reference_shape"], [2, 2])
        self.assertEqual(probe["diff"]["candidate_shape"], [1, 2])


class DetectionParityTests(unittest.TestCase):
    def test_reference_detection_parity_reuses_first_raw_outputs(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        class FakeReferenceModel:
            def __init__(self):
                self.pred = torch.tensor(
                    [[[1.0, 2.0], [3.0, 4.0]]],
                    dtype=torch.float32,
                )
                self.proto = torch.tensor(
                    [[[[5.0, 6.0], [7.0, 8.0]]]],
                    dtype=torch.float32,
                )
                self.names = {0: "mosaic"}
                self.postprocess_called = False

            def preprocess(self, imgs):
                return torch.ones((1, 3, 4, 4), dtype=torch.float32)

            def prepare_input(self, imgs):
                return imgs

            def inference(self, imgs):
                return [(self.pred, self.proto), {}]

            def postprocess(self, preds, img, orig_imgs):
                self.postprocess_called = True
                return [
                    build_native_yolo_result(
                        orig_img=orig_imgs[0],
                        names=self.names,
                        boxes=np.zeros((0, 6), dtype=np.float32),
                        masks=np.zeros((0, *orig_imgs[0].shape[:2]), dtype=np.uint8),
                    )
                ]

            def inference_and_postprocess(self, imgs, orig_imgs):
                raise AssertionError("reference parity should not rerun inference_and_postprocess")

            def release_cached_memory(self):
                return None

        class FakeCandidateModel:
            def preprocess(self, imgs):
                return [np.ones((3, 4, 4), dtype=np.float32)]

            def _run_raw_outputs_to_cpu(self, input_frame):
                return (
                    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
                )

            def inference_and_postprocess(self, imgs, orig_imgs):
                return [
                    build_native_yolo_result(
                        orig_img=orig_imgs[0],
                        names={0: "mosaic"},
                        boxes=np.zeros((0, 6), dtype=np.float32),
                        masks=np.zeros((0, *orig_imgs[0].shape[:2]), dtype=np.uint8),
                    )
                ]

            def release_cached_memory(self):
                return None

        reference_model = FakeReferenceModel()
        candidate_model = FakeCandidateModel()
        with (
            mock.patch("lada.parity.resolve_torch_device", return_value=torch.device("cpu")),
            mock.patch(
                "lada.parity.build_mosaic_detection_model",
                side_effect=[reference_model, candidate_model],
            ),
        ):
            report = _run_detection_parity(
                frame,
                detection_model_path="dummy.pt",
                reference_device_id="cpu",
                fp16=False,
            )

        self.assertTrue(reference_model.postprocess_called)
        probes = {probe["name"]: probe for probe in report["probes"]}
        self.assertEqual(0.0, probes["detector/raw_outputs/pred"]["diff"]["max_abs_diff"])
        self.assertEqual(0.0, probes["detector/raw_outputs/proto"]["diff"]["max_abs_diff"])


if __name__ == "__main__":
    unittest.main()
