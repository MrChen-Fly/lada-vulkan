from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


INPUT_PATH = Path(".codex_tmp/aoz_0.webm")
DETECTION_MODEL_PATH = Path("model_weights/lada_mosaic_detection_model_v4_fast.pt")
RESTORATION_MODEL_PATH = Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")
START_FRAME = 54
FRAME_COUNT = 5


def _run_aoz_pipeline_parity_subprocess() -> dict[str, object]:
    script = textwrap.dedent(
        f"""
        import gc
        import json

        from lada.models.basicvsrpp.inference import load_model
        from lada.parity import load_video_clip, resolve_reference_torch_device
        from lada.parity_restoration_core import build_reference_basicvsrpp_modules
        from lada.parity_restoration_pipeline import build_restoration_pipeline_probes
        from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
        from lada.restorationpipeline.basicvsrpp_vulkan_restorer import NcnnVulkanBasicvsrppMosaicRestorer

        input_path = {str(INPUT_PATH)!r}
        detection_model_path = {str(DETECTION_MODEL_PATH)!r}
        restoration_model_path = {str(RESTORATION_MODEL_PATH)!r}
        start_frame = {START_FRAME}
        frame_count = {FRAME_COUNT}

        def summarize_prefix(probes, prefix):
            count = 0
            max_mean_abs_diff = 0.0
            max_max_abs_diff = 0.0
            for probe in probes:
                if not isinstance(probe, dict):
                    continue
                if not str(probe.get("name", "")).startswith(prefix):
                    continue
                diff = probe.get("diff", {{}})
                if not isinstance(diff, dict):
                    continue
                if not bool(diff.get("shape_match", False)):
                    raise AssertionError(f"Shape mismatch for parity probe: {{probe.get('name')}}")
                count += 1
                max_mean_abs_diff = max(max_mean_abs_diff, float(diff.get("mean_abs_diff", 0.0)))
                max_max_abs_diff = max(max_max_abs_diff, float(diff.get("max_abs_diff", 0.0)))
            return {{
                "count": count,
                "max_mean_abs_diff": max_mean_abs_diff,
                "max_max_abs_diff": max_max_abs_diff,
            }}

        _, reference_device = resolve_reference_torch_device("cpu")
        frames = load_video_clip(input_path, start_frame=start_frame, frame_count=frame_count)
        reference_model = load_model(None, restoration_model_path, reference_device, fp16=False)
        reference_modules = build_reference_basicvsrpp_modules(reference_model, reference_device)
        reference_restorer = BasicvsrppMosaicRestorer(reference_model, reference_device, fp16=False)
        candidate_restorer = NcnnVulkanBasicvsrppMosaicRestorer(
            restoration_model_path,
            config_path=None,
            fp16=False,
            artifacts_dir=None,
        )
        try:
            pipeline_result = build_restoration_pipeline_probes(
                frames,
                detection_model_path=detection_model_path,
                reference_device_id="cpu",
                reference_device=reference_device,
                reference_modules=reference_modules,
                reference_restorer=reference_restorer,
                candidate_restorer=candidate_restorer,
                fp16=False,
            )
        finally:
            reference_restorer.release_cached_memory()
            candidate_restorer.release_cached_memory()
            reference_modules.clear()
            del reference_model
            del reference_modules
            del reference_restorer
            del candidate_restorer
            gc.collect()

        print(json.dumps({{
            "errors": list(pipeline_result["errors"]),
            "descriptor_summary": summarize_prefix(
                list(pipeline_result["probes"]),
                "descriptor/restored_patch/",
            ),
            "blend_summary": summarize_prefix(
                list(pipeline_result["probes"]),
                "blend/final_frame/",
            ),
        }}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "AOZ pipeline parity subprocess failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


class BasicvsrppVulkanAozPipelineParityTests(unittest.TestCase):
    def test_aoz_pipeline_restore_and_blend_match_reference_thresholds(self) -> None:
        if not INPUT_PATH.exists():
            raise unittest.SkipTest(f"Missing AOZ sample clip: {INPUT_PATH}")
        if not DETECTION_MODEL_PATH.exists():
            raise unittest.SkipTest(
                f"Missing detection model artifact: {DETECTION_MODEL_PATH}"
            )
        if not RESTORATION_MODEL_PATH.exists():
            raise unittest.SkipTest(
                f"Missing restoration model artifact: {RESTORATION_MODEL_PATH}"
            )

        # Run the AOZ parity pass in a fresh Python process. The native Vulkan
        # runtime can become unstable after other heavy in-process native trace
        # tests, while the same AOZ pass is stable in isolation.
        result = _run_aoz_pipeline_parity_subprocess()
        self.assertEqual([], result["errors"])

        descriptor_summary = result["descriptor_summary"]
        self.assertEqual(FRAME_COUNT, int(descriptor_summary["count"]))
        self.assertLessEqual(float(descriptor_summary["max_mean_abs_diff"]), 0.05)
        self.assertLessEqual(float(descriptor_summary["max_max_abs_diff"]), 4.0)

        blend_summary = result["blend_summary"]
        self.assertEqual(FRAME_COUNT, int(blend_summary["count"]))
        self.assertLessEqual(float(blend_summary["max_mean_abs_diff"]), 0.003)
        self.assertLessEqual(float(blend_summary["max_max_abs_diff"]), 3.0)


if __name__ == "__main__":
    unittest.main()
