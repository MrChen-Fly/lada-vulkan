# Detection With NCNN Vulkan

## Overview

Lada now uses the local `ncnn + Vulkan` runtime for the `vulkan:0` mosaic detection path.
The previous OpenCV Vulkan DNN backend has been removed from the detection runtime.

Current runtime flow:

- Input detection model: `.pt`
- On-demand export: Ultralytics `format="ncnn"`
- Runtime: local `ncnn` Python module with `use_vulkan_compute`
- Preprocess: native `LadaVulkanNetRunner.preprocess_bgr_u8(...)` Vulkan path for resize + pad + normalize; batched detector inputs now prefer `preprocess_bgr_u8_batch(...)` to share one Vulkan submit per batch
- Precision: prefers `fp16` when enabled, otherwise uses `fp32`
- Postprocess: standalone `lada.YoloSegPostprocess` subnet executed through the native Vulkan runner
- Output contract: `DetectionResult`, compatible with the existing scene assembly path

## Artifact Layout

For a detection model such as `model_weights/lada_mosaic_detection_model_v4_fast.pt`, Lada now manages precision-specific NCNN artifacts:

- `model_weights/lada_mosaic_detection_model_v4_fast.fp16_ncnn_model`
- `model_weights/lada_mosaic_detection_model_v4_fast.fp32_ncnn_model`

This avoids the old problem where one generic `_ncnn_model` directory could be silently reused with the wrong precision.

Each artifact directory contains:

- `model.ncnn.param`
- `model.ncnn.bin`
- `metadata.yaml`

## FP16 Policy

Behavior is now:

- `--fp16` enabled:
  Lada prefers the `fp16` NCNN artifact and Vulkan runtime flags.
- `--no-fp16`:
  Lada uses the `fp32` NCNN artifact.
- If the requested precision cannot export or warm up:
  Lada now fails fast instead of silently downgrading to another precision.

On the current AMD Radeon 880M test machine, `fp16` export and runtime warmup both succeed.

## Current Limits

The detection model is now running through `ncnn + Vulkan`, but it is still not a pure-GPU end-to-end path.

Known remaining limits:

- Decoder frames still arrive on CPU before being uploaded into the Vulkan preprocess path.
- CPU-side scene assembly is still used after detection.
- The media IO cleanup now keeps decoded frames as numpy until they are actually
  needed for blend / restore work, so the pipeline no longer tensorizes every
  decoded frame eagerly on the decode thread.
- First-run export and warmup are still setup work on CPU.
- The exported `v4-fast` segmentation graph now audits clean on compute layers:
  - unsupported Vulkan compute layers: none

That means the detector compute graph and fused postprocess graph are now Vulkan-covered; the remaining CPU work is mainly decode/setup/scene-assembly glue.

## Reproduce

Run the full CLI on `main.webm` and write a timing report:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m lada.cli.main `
  --input "D:/Code/github/lada/resources/main.webm" `
  --output "D:/Code/github/lada/.helloagents/tmp/main_vulkan_gpuutil_yoloprep.mp4" `
  --device "vulkan:0" `
  --timing-report-path "D:/Code/github/lada/.helloagents/tmp/main_vulkan_gpuutil_yoloprep_timing.json"
```

## Latest Measured Result

Latest `main.webm` measurement on the current branch (`main_vulkan_mediaio_pushdown_timing.json`):

- CLI total: `96.768563s`
- File total: `76.921395s`
- Decode (`frame_decode_s`): `19.053603s`
- Detection inference (`frame_inference_s`): `26.783908s`
- Detection preprocess (`detector_preprocess_total_s`): `0.963524s`
- Detection postprocess / scene-side handoff (`frame_detector_postprocess_s`): `30.625312s`

Compared with the earlier single-frame Vulkan preprocess branch (`main_vulkan_gpuutil_yoloprep_timing.json`):

- Detection preprocess: `1.542998s -> 1.043074s`
- Detection-side postprocess / scene handoff is still dominated by CPU scene assembly, not by detector compute layers

The detector compute graph is already Vulkan-covered. The largest remaining non-GPU boundary around detection is now the CPU-side decode and scene/clip assembly around the detector outputs.
