import unittest
from unittest.mock import patch

from lada.utils import video_utils


class EncodingPresetTests(unittest.TestCase):
    def tearDown(self) -> None:
        video_utils.get_encoding_presets.cache_clear()
        video_utils.is_intel_qsv_encoding_available.cache_clear()
        video_utils.is_nvidia_cuda_encoding_available.cache_clear()
        video_utils.is_apple_videotoolbox_encoding_available.cache_clear()
    def test_should_not_expose_amd_presets_even_when_amf_encoders_are_present(self) -> None:
        available_encoders = [
            video_utils.Encoder(
                name="h264_amf",
                long_name="AMD AMF H.264 Encoder",
                hardware_encoder=True,
                hardware_devices={"amf"},
            ),
            video_utils.Encoder(
                name="hevc_amf",
                long_name="AMD AMF HEVC encoder",
                hardware_encoder=True,
                hardware_devices={"amf"},
            ),
        ]

        with patch(
            "lada.utils.video_utils.get_video_encoder_codecs",
            return_value=available_encoders,
        ):
            preset_names = {preset.name for preset in video_utils.get_encoding_presets()}

        self.assertNotIn("h264-amd-gpu-fast", preset_names)
        self.assertNotIn("hevc-amd-gpu-balanced", preset_names)
        self.assertNotIn("hevc-amd-gpu-hq", preset_names)

    def test_should_fallback_to_cpu_preset_when_no_supported_gpu_encoder_is_available(self) -> None:
        with (
            patch("lada.utils.os_utils.has_nvidia_gpu", return_value=False),
            patch("lada.utils.video_utils.is_nvidia_cuda_encoding_available", return_value=False),
            patch("lada.utils.video_utils.is_apple_videotoolbox_encoding_available", return_value=False),
            patch("lada.utils.os_utils.has_intel_arc_gpu", return_value=False),
            patch("lada.utils.video_utils.is_intel_qsv_encoding_available", return_value=False),
        ):
            preset_name = video_utils.get_default_preset_name()

        self.assertEqual(preset_name, "h264-cpu-fast")

    def test_should_choose_apple_preset_before_cpu_when_videotoolbox_is_available(self) -> None:
        with (
            patch("lada.utils.os_utils.has_nvidia_gpu", return_value=False),
            patch("lada.utils.video_utils.is_nvidia_cuda_encoding_available", return_value=False),
            patch("lada.utils.video_utils.is_apple_videotoolbox_encoding_available", return_value=True),
            patch("lada.utils.os_utils.has_intel_arc_gpu", return_value=False),
            patch("lada.utils.video_utils.is_intel_qsv_encoding_available", return_value=False),
        ):
            preset_name = video_utils.get_default_preset_name()

        self.assertEqual(preset_name, "hevc-apple-gpu-balanced")

    def test_should_choose_intel_preset_before_cpu_when_qsv_is_available(self) -> None:
        with (
            patch("lada.utils.os_utils.has_nvidia_gpu", return_value=False),
            patch("lada.utils.video_utils.is_nvidia_cuda_encoding_available", return_value=False),
            patch("lada.utils.video_utils.is_apple_videotoolbox_encoding_available", return_value=False),
            patch("lada.utils.os_utils.has_intel_arc_gpu", return_value=True),
            patch("lada.utils.video_utils.is_intel_qsv_encoding_available", return_value=True),
        ):
            preset_name = video_utils.get_default_preset_name()

        self.assertEqual(preset_name, "hevc-intel-gpu-hq")


if __name__ == "__main__":
    unittest.main()
