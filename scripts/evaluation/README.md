# Evaluation Scripts

This directory contains manual evaluation helpers for developers.

The `compare-restoration-devices.py` script is an on-demand regression tool.
It is not wired into CI, pre-commit hooks, or an automatic "rerun after every
Vulkan change" workflow.

What it does:

- renders the same input video(s) on two compute targets
- writes per-device output media plus per-device JSON reports
- compares the rendered outputs with `ffprobe`, `ffmpeg` PSNR/SSIM, and
  PyAV/Numpy frame-difference statistics

Typical usage:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluation/compare-restoration-devices.py `
  --input "D:/Code/github/lada-vulkan/pic_bak/dungeon/character/AOZ/中条鈴華" `
  --output-dir "D:/Code/github/lada-vulkan/.helloagents/tmp/regression/zhongtiaolinghua/manual" `
  --baseline-device "cpu" `
  --candidate-device "vulkan:0"
```

If you already have fresh rendered outputs and only want the comparison stage:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluation/compare-restoration-devices.py `
  --input "D:/Code/github/lada-vulkan/pic_bak/dungeon/character/AOZ/中条鈴華" `
  --output-dir "D:/Code/github/lada-vulkan/.helloagents/tmp/regression/zhongtiaolinghua/manual" `
  --baseline-device "cpu" `
  --candidate-device "vulkan:0" `
  --reuse-existing-outputs
```

Generated artifacts stay under the chosen `--output-dir`, for example:

- `0-cpu.mp4`
- `0-cpu.json`
- `0-vulkan-0.mp4`
- `0-vulkan-0.json`
- `cpu-vs-vulkan-0.json`
