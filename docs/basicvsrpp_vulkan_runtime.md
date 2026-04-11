# BasicVSR++ Vulkan Runtime

本文说明项目中 `BasicVSR++ + ncnn + Vulkan` 修复链路的现状、入口、模块分工与运行时边界，重点覆盖当前保留的 `vulkan:0` 路径。

## 1. 总览

当前 Vulkan 修复路径已经收敛为一个明确的计算目标：

- CLI / GUI / `FrameRestorer` 统一通过 `vulkan:0` 选择 Vulkan 修复后端。
- 保留的修复实现是 `5` 帧滑窗路径与 recurrent/native 路径。
- 主要恢复算子尽量下沉到 GPU，但媒体解码、场景组装和部分混合逻辑仍然保留在 CPU。
- Vulkan blend 快捷路径不能再假定与 CPU 基线等价，当前应以 CPU 参考混合逻辑为安全基线。

这个运行时的目标不是复刻 Torch/CUDA 实现，而是在本地 `ncnn` Vulkan runtime 能力范围内，把可稳定下沉到 GPU 的预处理、修复与桥接能力统一到一条可维护链路上。

## 2. 入口与选择

CLI 入口：

```bash
--device vulkan:0
```

主要选择逻辑位于：

- `lada/cli/main.py`
- `lada/compute_targets.py`
- `lada/restorationpipeline/restoration_backends.py`

当前 `vulkan:0` 会作为实验性但可用的 compute target 暴露，前提包括：

- 系统可加载 Vulkan runtime。
- Python 侧可以导入 `ncnn`。
- `ncnn.get_gpu_count() > 0`。
- 本地 runtime 暴露了修复链路所需的关键能力，例如 `deformconv`、`GridSample`、YOLO 相关后处理，以及 BasicVSR++ clip runner。

## 3. 运行时结构

### 3.1 FrameRestorer 层

- `lada/restorationpipeline/frame_restorer.py`
  - 负责 orchestration shell。
  - 对外暴露统一修复 API，并连接 profile / timing 统计。
- `lada/restorationpipeline/frame_restorer_worker.py`
  - 承担 clip preprocess、restore 与 frame restore worker 循环。
  - 管理 `FrameRestorationState`、队列与阶段间状态推进。
- `lada/restorationpipeline/frame_restorer_clip_ops.py`
  - 负责 clip descriptor materialize。
  - 负责 native descriptor restore 与 clip 相关的准备工作。
- `lada/restorationpipeline/frame_restorer_blend.py`
  - 保留 CPU 参考混合实现。
  - 负责 batch blend，以及 native blend bridge 的 CPU fallback。

### 3.2 BasicVSR++ Vulkan 层

- `lada/restorationpipeline/basicvsrpp_vulkan_restorer.py`
  - 作为运行时 orchestration shell。
  - 统一 artifact、runner 与 runtime feature 解析。
- `lada/restorationpipeline/basicvsrpp_vulkan_restore_paths.py`
  - 负责 restore path 分派。
  - 覆盖 Vulkan frame preprocess、fused restore、recurrent modular 与 native clip runner 路径。
- `lada/restorationpipeline/basicvsrpp_vulkan_blend.py`
  - 负责 native blend patch / padded blend / batch blend bridge。
- `lada/restorationpipeline/basicvsrpp_vulkan_io.py`
  - 负责 Vulkan tensor 编组、clip window 输入准备与 GPU bridge。
- `lada/restorationpipeline/basicvsrpp_vulkan_common.py`
  - 负责 `ncnn` / `pnnx` artifact 公共逻辑。
- `lada/restorationpipeline/basicvsrpp_vulkan_export.py`
  - 负责从 PyTorch 导出 Vulkan 所需 artifact。
- `lada/restorationpipeline/basicvsrpp_vulkan_artifacts.py`
  - 负责 artifact 解析与定位。

### 3.3 Native runtime

- `native/ncnn_vulkan_runtime/`
  - 项目自带的 `ncnn` Vulkan runtime。
- `scripts/build_ncnn_vulkan_runtime.ps1`
  - Windows 下的本地 runtime 构建入口。

## 4. Artifact 与运行时能力

### 4.1 Artifact 约束

Vulkan 修复不再依赖通用的旧兼容后端，而是围绕当前保留的 `5f` 与 recurrent artifact 工作：

- 由 PyTorch 模型导出 `pnnx` / `ncnn` artifact。
- 单帧 `1f` compatibility backend 已不再是主线路径。
- 当前运行时默认围绕 `frame_count = 5` 的窗口约束工作。

### 4.2 Runtime feature

`lada/restorationpipeline/runtime_options.py` 统一描述后端能力，当前 Vulkan 路径关注的能力包括：

- GPU blob bridge
- native frame preprocess
- batched native frame preprocess
- fused `restore_clip`
- recurrent modular runtime
- native clip runner
- descriptor restore
- native blend patch / inplace / padded / batch-inplace 变体

这些 feature 决定 `basicvsrpp_vulkan_restore_paths.py` 最终选择哪条修复路径。

### 4.3 Restore path

对外主要暴露两类修复入口：

- `restore(video, max_frames=-1)`
- `restore_cropped_clip_frames(...)`

当前可选路径包括：

- fixed-size fused `restore_clip`
- recurrent modular path
- native clip runner path
- 基于 `5` 帧窗口的 sliding restore path

同时，长视频已经支持按 segment 流式修复，避免整段 clip 完成前长期占用 GPU 与队列资源。

## 5. Blend 设计边界

native runtime 暴露的 blend 能力包括：

- `blend_patch_gpu_inplace`
- `blend_patch_gpu_preprocess_inplace`
- `blend_patch_gpu_preprocess_inplace_batch`

但当前项目不能把这些接口直接等价为“最终安全混合结果”。实际交付仍然遵守以下原则：

- CPU 参考路径负责 `unpad`、`resize`、blend mask 生成与最终 apply。
- 多 ROI、多 mosaic、边界反射与尺寸对齐问题，优先以 `frame_restorer_blend.py` 的 CPU 逻辑校准。
- 若 native blend bridge 与 CPU 参考路径产生偏差，应回落到 CPU 基线，而不是继续扩展 Vulkan blend 快捷路径。

这条规则来自现有回归结论：多 mosaic 场景下，Vulkan blend 快捷路径可能与 CPU 结果不一致，并导致局部黑块。

## 6. GPU 与 CPU 分工

### 6.1 当前已经稳定下沉到 GPU 的部分

- `quarter_downsample`
- `feat_extract`
- `spynet`
- fused `restore_clip` 或 recurrent/native clip runner
- `deformconv`
- `GridSample`
- GPU blob bridge
- Vulkan frame preprocess

### 6.2 仍保留在 CPU 的边界

- `pnnx` 与 artifact 生成
- 媒体 decode
- scene / clip 组装与调度 glue
- 参考混合路径中的部分 resize / mask 逻辑
- numpy / tensor materialization
- PyAV / FFmpeg 侧的视频封装与 mux

因此，当前 Vulkan 路径的性能瓶颈通常不再是核心修复算子本身，而是解码、拼装、桥接和最终媒体输出。

## 7. 当前实现状态

截至目前，Vulkan 修复路径的关键状态如下：

- 统一保留 `vulkan:0` 作为修复 compute target。
- 保留 `5f` / recurrent 主路径，移除了单帧兼容后端在主线中的地位。
- 长 clip 默认按 segment 流式修复。
- CLI 异步写出依赖 `VideoWriter`。
- `runtime_options.py` 统一管理调度参数与运行时 feature。
- blend 仍以 CPU 参考路径为安全基线，native shortcut 需要严格回归覆盖后才能放宽。

## 8. 排障要点

### 8.1 runtime 不可用

优先检查：

- Vulkan loader 是否可用。
- Python 是否能导入 `ncnn`。
- 本地 runtime 是否包含 `LadaVulkanNetRunner`。
- 关键算子如 `deformconv`、`GridSample` 是否可用。
- native clip runner 与 blend patch 接口是否正确导出。

### 8.2 artifact / `frame_count` 不匹配

如果运行时、artifact 与 restore path 的窗口约束不一致，通常会表现为：

- artifact 加载失败
- clip restore 失败
- feature 解析不匹配
- runtime path 选择异常

因此，排障时应优先确认 artifact、`frame_count` 假设与 restore path 是否一致。

### 8.3 blend 异常

如果出现黑块、局部 ROI 失败或多 mosaic 场景结果异常，应优先验证：

- 是否误走了不安全的 native blend shortcut。
- CPU 参考路径与 native 路径的输入尺寸、mask 与边界处理是否一致。
- 当前问题是否应该直接回落到 `frame_restorer_blend.py` 的 CPU 基线。
