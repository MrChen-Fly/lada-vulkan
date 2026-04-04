# Vulkan Media IO Notes

## Scope

This note records the current media IO boundary around the `vulkan:0` pipeline.
The main detector and restoration compute graphs already run through the local
`ncnn + Vulkan` runtime. The remaining end-to-end CPU boundaries are mostly:

- video decode into CPU frames
- scene/clip assembly around detector outputs
- final FFmpeg/PyAV handoff for encode + mux

## What Changed

### 1. Decode no longer tensorizes every frame eagerly

`VideoReader.frames()` now supports:

```python
VideoReader(...).frames(output_format="numpy")
```

That keeps decoded frames as contiguous CPU `HWC uint8` numpy arrays instead of
immediately wrapping them with `torch.from_numpy(...)`.

### 2. Detection consumes decoded numpy frames directly

`MosaicDetector` now reads:

- decoded frames as numpy arrays
- detector preprocess through the native Vulkan `preprocess_bgr_u8_batch(...)`
- tensor conversion only when a frame actually needs to enter blend / restore

This does not remove CPU decode, but it removes one unconditional decode-side
bridge on the main thread.

### 3. Writer no longer does a CPU BGR->RGB conversion first

`VideoWriter.write(..., bgr2rgb=True)` still accepts BGR frames from the pipeline,
but it now passes them to PyAV as `bgr24` input instead of first calling:

```python
cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

PyAV / FFmpeg handles the input pixel format conversion when the encoder expects
`yuv420p`, `nv12`, or another codec-specific format.

### 4. AMD machines now default to AMF GPU encode

`lada.utils.video_utils` now probes AMD AMF availability and exposes AMD GPU
presets through the existing preset system.

On this machine the default preset now resolves to:

```text
hevc-amd-gpu-balanced -> hevc_amf -usage transcoding -quality balanced
```

This is intentional. The current machine can run AMF successfully, while the
experimental Vulkan encode path still fails during `hwupload`.

## Current Boundary Map

```text
PyAV decode (CPU)
  -> contiguous BGR uint8 numpy frame
  -> Vulkan preprocess/upload
  -> detection / restoration / blend on Vulkan
  -> CPU frame handed to FFmpeg/PyAV as bgr24
  -> AMF GPU encode + mux
```

## Latest `main.webm` Result

Command:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m lada.cli.main `
  --input "resources/main.webm" `
  --output ".helloagents/tmp/main_vulkan_dual_chain_streaming_v7.mp4" `
  --device "vulkan:0" `
  --timing-report-path ".helloagents/tmp/main_vulkan_dual_chain_streaming_v7_timing.json"
```

Artifacts:

- output video: `.helloagents/tmp/main_vulkan_dual_chain_streaming_v7.mp4`
- timing report: `.helloagents/tmp/main_vulkan_dual_chain_streaming_v7_timing.json`
- stdout log: `.helloagents/tmp/main_vulkan_dual_chain_streaming_v7_stdout.log`

Validation:

- format: `mp4`
- video: `hevc 1280x720 yuv420p`
- audio: `vorbis`
- encoder: `hevc_amf -usage transcoding -quality balanced`

Measured result:

- `command_total_s = 93.033082`
- `file_total_s = 76.578318`
- `frame_detector_postprocess_s = 28.068898`
- `clip_preprocess_s = 4.030366`
- `prepared_clip_queue_get_s = 10.992434`
- `clip_restore_s = 64.233813`
- `restored_clip_queue_get_s = 47.815228`
- `frame_blend_s = 27.958011`
- `frame_blend_vulkan_s = 23.596455`
- `clips_emitted = 3`

Compared with the earlier AMF-default baseline:

- `command_total_s`: `96.174126 -> 93.033082`
- `file_total_s`: `77.677998 -> 76.578318`
- `frame_detector_postprocess_s`: `30.450568 -> 28.068898`
- `restored_clip_queue_get_s`: `53.598414 -> 47.815228`
- detector emission: `2 clips -> 3 clips`

Conclusion:

- the AMF output path remains stable and the generated mp4 is valid
- descriptor + preprocess-worker pushdown is now faster than the previous AMF baseline
- the remaining dominant cost is still Vulkan detector / restore contention, not final encode

## Streaming Pushdown Note

The current best result came from keeping the lazy descriptor path, but changing
the default detector emission cadence to roughly two restore chunks per clip
when the restoration backend exposes `stream_restore_chunk_size`.

In practice on `main.webm` this means:

- detector emission moves from `2` long descriptors to `3` medium descriptors
- the preprocess worker can hand prepared clips to the restore worker earlier
- the frame worker spends less time blocked on `restored_clip_queue_get_s`
- end-to-end time drops without reintroducing the old eager clip build on the detector thread

## Rejected Shared-GPU Queue Throttling

A follow-up experiment tried to reduce single-GPU contention by statically
throttling the detector side:

- shrinking detector feeder / inference queue depth
- shrinking detector-driven backlog indirectly through queue pressure
- reordering detector `clip finalize` ahead of `frame_detection_queue.put(...)`

Those changes were tested on `main.webm` and all produced valid mp4 outputs,
but every run regressed wall time versus the current v7 baseline:

- `v8`: `command_total_s = 108.239513`
- `v9`: `command_total_s = 102.476946`
- `v10`: `command_total_s = 105.574370`

The practical conclusion is:

- static detector queue throttling is not an effective fix for the current
  shared-GPU contention
- detector emit-order reordering also regresses the pipeline
- the retained baseline stays `main_vulkan_dual_chain_streaming_v7` for now
- the next optimization target should move to compute-side submit granularity,
  not more Python queue-depth tuning

## Compute-Side Submit Granularity Follow-Up

A compute-side follow-up pushed two more native batching changes:

- detector fused subnet now supports batch submission through
  `run_yolo_segmentation_subnet_batch(...)`
- `BasicVsrppClipRunner` no longer downloads every `output_frame` with one
  submit per frame; it now batch-downloads output frames in bounded chunks

The first unbounded version of `output_frame` batch download was not viable on
`main.webm`: it hit repeated `vkAllocateMemory failed -2` allocation failures
and left the CLI waiting on a crashed worker thread. The retained variant keeps
the batched path but limits output downloads to smaller chunks so the recurrent
runtime does not exhaust Vulkan memory.

Validation command:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m lada.cli.main `
  --input "resources/main.webm" `
  --output ".helloagents/tmp/main_vulkan_compute_submit_v12.mp4" `
  --device "vulkan:0" `
  --timing-report-path ".helloagents/tmp/main_vulkan_compute_submit_v12_timing.json"
```

Artifacts:

- output video: `.helloagents/tmp/main_vulkan_compute_submit_v12.mp4`
- timing report: `.helloagents/tmp/main_vulkan_compute_submit_v12_timing.json`
- stdout log: `.helloagents/tmp/main_vulkan_compute_submit_v12_stdout.log`

Validation:

- format: `mp4`
- video: `hevc 1280x720 yuv420p`
- audio: `vorbis`
- encoder: `hevc_amf -usage transcoding -quality balanced`

Measured result:

- `command_total_s = 97.285877`
- `file_total_s = 77.682049`
- `frame_detector_postprocess_s = 28.017962`
- `clip_preprocess_s = 6.085537`
- `clip_restore_s = 63.529906`
- `restored_clip_queue_get_s = 50.786253`
- `frame_blend_s = 25.831658`
- `clips_emitted = 3`

Compared with `v7`:

- detector fused subnet batching is roughly neutral on wall time
- native recurrent restore compute improves slightly (`64.233813 -> 63.529906`)
- blend cost improves (`27.958011 -> 25.831658`)
- overall file wall time still regresses (`76.578318 -> 77.682049`) because
  queue wait shifted upward again (`clip_preprocess_s` and
  `restored_clip_queue_get_s`)

Conclusion:

- the chunked batch-download path is stable and fixes the unbounded
  `output_frame` OOM / hang failure
- it is not a new performance baseline
- the retained fastest validated baseline is still
  `main_vulkan_dual_chain_streaming_v7`

## Detection Scene/Clip Pushdown Follow-Up

The next follow-up removed more CPU-side detection/clip bookkeeping overhead:

- native YOLO masks now stay in scene-ready `uint8 HWC` image form instead of
  being wrapped into CPU torch tensors immediately
- `MosaicDetector` now consumes masks through one normalized image path instead
  of maintaining a separate native branch
- `Scene` now caches expanded crop boxes and resize-reference dimensions while
  detections are appended/merged
- `ClipDescriptor` now reuses those cached crop boxes instead of recomputing
  `expand_box_to_target(...)` for every emitted frame
- clip preprocessing now keeps masks on the CPU numpy path until the final
  padded mask is materialized for blend

Validation command:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m lada.cli.main `
  --input "resources/main.webm" `
  --output ".helloagents/tmp/main_detection_pushdown_v13.mp4" `
  --device "vulkan:0" `
  --timing-report-path ".helloagents/tmp/main_detection_pushdown_v13_timing.json"
```

Artifacts:

- output video: `.helloagents/tmp/main_detection_pushdown_v13.mp4`
- timing report: `.helloagents/tmp/main_detection_pushdown_v13_timing.json`
- stdout log: `.helloagents/tmp/main_detection_pushdown_v13_stdout.log`

Validation:

- format: `mp4`
- video: `hevc 1280x720 yuv420p`
- audio: `vorbis`
- encoder: `hevc_amf -usage transcoding -quality balanced`

Measured result:

- `command_total_s = 80.103785`
- `file_total_s = 67.289923`
- `frame_detector_scene_update_s = 0.015067`
- `clip_descriptor_build_s = 0.000054`
- `clip_preprocess_s = 3.792910`
- `clip_restore_s = 55.989540`
- `restored_clip_queue_get_s = 47.164402`
- `frame_blend_s = 19.383206`
- `clips_emitted = 3`

Compared with `v12`:

- detector result build shrinks materially (`0.098912 -> 0.014405`)
- scene update bookkeeping is almost eliminated (`2.853458 -> 0.015067`)
- clip preprocess drops (`6.085537 -> 3.792910`)
- recurrent restore wall time also drops (`63.529906 -> 55.989540`)
- overall file wall time improves strongly (`77.682049 -> 67.289923`)

Updated conclusion:

- this is the current fastest validated `main.webm` Vulkan pipeline result
- the detection-side CPU bookkeeping regression is no longer a dominant cost
- the remaining long pole is still the restore/blend side plus queue backpressure,
  not detector scene assembly itself



## Queue Backpressure Scheduling Follow-Up

A scheduling-only follow-up targeted the two remaining pressure buckets around the
`vulkan:0` `main.webm` path:

- `restored_clip_queue_get_s`
- `frame_detection_queue_put_s`

Validated artifacts:

- `v14`: `.helloagents/tmp/main_queue_backpressure_v14.mp4`
- `v15`: `.helloagents/tmp/main_queue_backpressure_v15.mp4`
- `v16`: `.helloagents/tmp/main_queue_backpressure_v16.mp4`
- `v17`: `.helloagents/tmp/main_queue_backpressure_v17.mp4`
- `v18`: `.helloagents/tmp/main_queue_backpressure_v18.mp4`

Summary:

| Run | command_total_s | file_total_s | restored_clip_queue_get_s | frame_detection_queue_put_s | clip_preprocess_s | clip_restore_s | frame_blend_s |
|-----|----------------:|-------------:|--------------------------:|----------------------------:|------------------:|---------------:|--------------:|
| `v13` baseline | 80.103785 | 67.289923 | 47.164402 | 26.935985 | 3.792910 | 55.989540 | 19.383206 |
| `v14` | 87.222403 | 71.523185 | 44.757524 | 0.003011 | 14.371663 | 64.254176 | 25.776537 |
| `v15` | 88.457059 | 71.066994 | 53.201204 | 0.002448 | 11.746259 | 58.333570 | 16.845453 |
| `v16` | 87.987059 | 70.376432 | 52.519353 | 10.846710 | 10.175263 | 57.829296 | 16.945049 |
| `v17` | 89.586328 | 73.583661 | 57.154719 | 11.141763 | 9.611077 | 60.814837 | 15.536985 |
| `v18` | 89.605737 | 74.318037 | 54.937688 | 23.969644 | 4.520659 | 61.484991 | 18.477372 |

Conclusion:

- aggressive queue drain can collapse detector-side put blocking, but it shifts
  cost into `clip_preprocess_s`, `clip_restore_s`, and shared detector/restore contention
- `v16` is the best scheduling-only variant tested in this round, but it is still
  slower than the retained `v13` baseline on end-to-end wall time
- the fastest validated baseline remains `main_detection_pushdown_v13`
- the next useful optimization target is native/GPU-side clip preprocess or
  chunk-boundary wait removal, not more Python queue choreography

## Why Vulkan Encode Is Still Not The Default

The current machine exposes `h264_vulkan` / `hevc_vulkan` through the external
FFmpeg CLI, but the real upload path is not usable yet. The last probe failed
with errors equivalent to:

- `No memory type found for flags 0x1`
- `Failed to allocate frame to upload to`

So the practical decision is:

- use AMF as the stable GPU encode path now
- keep Vulkan encode as a later driver/runtime investigation item

## What Is Still Not GPU-Native

The following pieces still break the pure-GPU chain:

- PyAV decode still materializes CPU frames
- scene bookkeeping and clip assembly are still CPU-side orchestration
- FFmpeg/PyAV still receives CPU frames before the AMF hardware encoder takes over

## Next Downshift Targets

The next realistic media IO pushdown targets are:

1. introduce a native decode-upload bridge so decoded frames can move into the
   Vulkan preprocess path without the Python frame object boundary
2. move more scene/clip bookkeeping away from per-frame Python object churn
3. evaluate whether the local runtime should own a narrower encoder handoff so
   the FFmpeg/PyAV boundary becomes mostly queue drain work

## Pitfalls

- Do not use plain `python`; use `.venv\Scripts\python.exe`
- PowerShell may report a non-zero native command result when stderr contains
  runtime banner output, even if the generated mp4 and timing report are valid
- A lower `video_writer_write_s` does not mean encode disappeared; it only means
  the writer path is asynchronous and the main thread now pays enqueue cost
- On this machine, working GPU encode means `AMF`, not `Vulkan encode`
