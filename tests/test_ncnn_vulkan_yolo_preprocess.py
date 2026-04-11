from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from lada.models.basicvsrpp.ncnn_vulkan import import_ncnn_module
from lada.models.yolo.ncnn_vulkan import NcnnVulkanYoloSegmentationModel


class NcnnVulkanYoloPreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ncnn = import_ncnn_module()
        if not getattr(cls.ncnn, "has_lada_vulkan_net_runner", False):
            raise unittest.SkipTest("Local ncnn runtime does not expose LadaVulkanNetRunner.")

        cls.model_path = Path("model_weights/lada_mosaic_detection_model_v4_fast.pt")
        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Missing YOLO model artifact: {cls.model_path}")

        cls.model = NcnnVulkanYoloSegmentationModel(
            str(cls.model_path),
            imgsz=640,
            fp16=True,
            device_index=0,
        )

    @staticmethod
    def _build_frame(height: int = 72, width: int = 128) -> np.ndarray:
        y_coords, x_coords = np.indices((height, width), dtype=np.uint16)
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[..., 0] = ((x_coords * 5 + y_coords * 3) % 256).astype(np.uint8)
        frame[..., 1] = ((x_coords * 7 + 19) % 256).astype(np.uint8)
        frame[..., 2] = ((y_coords * 11 + 53) % 256).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def _run_raw(self, prepared_input) -> tuple[np.ndarray, np.ndarray]:
        outputs = self.model.gpu_runner.run_many_to_cpu(
            {self.model.input_name: prepared_input},
            list(self.model.output_names),
        )
        pred = np.asarray(outputs[self.model.output_names[0]], dtype=np.float32)
        proto = np.asarray(outputs[self.model.output_names[1]], dtype=np.float32)
        return pred, proto

    def test_fp16_native_preprocess_matches_uploaded_cpu_path(self) -> None:
        frame = self._build_frame()
        input_shape = self.model._resolve_runtime_input_shape([frame])
        self.model._ensure_runtime_for_input_shape(input_shape)

        cpu_mat = self.model._letterbox_to_ncnn_mat(frame, input_shape=input_shape)
        uploaded = self.model.gpu_runner.upload_cpu_mat(cpu_mat)
        native = self.model._letterbox_to_vulkan_tensor(frame, input_shape=input_shape)

        uploaded_np = np.asarray(uploaded.download(), dtype=np.float32)
        native_np = np.asarray(native.download(), dtype=np.float32)
        np.testing.assert_allclose(native_np, uploaded_np, rtol=0.0, atol=0.0)

        pred_uploaded, proto_uploaded = self._run_raw(uploaded)
        pred_native, proto_native = self._run_raw(native)
        np.testing.assert_allclose(pred_native, pred_uploaded, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(proto_native, proto_uploaded, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
