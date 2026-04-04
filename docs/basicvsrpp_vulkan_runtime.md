# BasicVSR++ Vulkan Runtime 方案与避坑

本文档说明当前仓库里 `BasicVSR++ + ncnn + Vulkan + 自定义算子` 的正式实现方案、目录边界、运行链路、GPU/CPU 分工和维护注意事项。

结论先写在前面：

- 当前正式的 Vulkan 修复路径是固定 `5` 帧导出图上的 `BasicVSR++` runtime。
- 当前代码不再保留旧的单帧兼容实现；`frame_count != 5` 会直接报错。
- 目标不是“把 PyTorch 原样搬到 Vulkan”，而是把关键子图导出为 `ncnn` 模块，再借助本地 `ncnn` runtime、Vulkan、自定义 `deformconv` 和 `GridSample` 在 GPU 上执行。
- 当前主路径已经整理成“高层 orchestration shell + 低层职责模块”的结构，后续维护应优先沿着模块边界排查问题。

## 1. 目标与非目标

### 1.1 目标

- 在 Vulkan 设备上跑通 `BasicVSR++` 修复链路。
- 避开 `cuda:0` 依赖，让 AMD / iGPU 等非 CUDA 设备也能走 GPU 计算。
- 尽可能把 `deformconv`、`GridSample`、模块间 GPU bridge 和 blend patch 留在 GPU 侧，减少 CPU fallback。
- 让 artifact 缓存、native runtime、restore path、blend bridge 和上层调度的边界足够清晰。

### 1.2 非目标

- 不追求把全部预处理、后处理、媒体读写都搬到 GPU。
- 不保证当前实现支持任意帧长导出图。
- 不保证当前实现可以主动把 GPU 使用率限制到某个百分比，例如 `85%`。

## 2. 当前正式实现

### 2.1 设备与后端

CLI 可以传：

```bash
--device vulkan
```

代码会把它归一化成 `vulkan:0`。

相关入口：

- `lada/cli/main.py`
- `lada/compute_targets.py`
- `lada/restorationpipeline/restoration_backends.py`

### 2.2 正式修复路径

当前 `BasicVSR++` 的 Vulkan 正式入口是：

- `NcnnVulkanBasicvsrppMosaicRestorer(..., frame_count=5, ...)`

`frame_count` 不是“推荐值”，而是正式约束。`lada/restorationpipeline/basicvsrpp_vulkan_restorer.py` 会直接拒绝不是 `5` 的帧数。

## 3. 代码结构

### 3.1 上层修复流水线

- `lada/restorationpipeline/frame_restorer.py`
  - 高层 orchestration shell
  - 队列生命周期、线程启动停止、profile 汇总、公共迭代器 API
- `lada/restorationpipeline/frame_restorer_worker.py`
  - clip preprocess / restore / frame restore 三个 worker 主循环
  - `FrameRestorationState`、frame / clip buffer、marker 协调
- `lada/restorationpipeline/frame_restorer_clip_ops.py`
  - clip descriptor materialize
  - native descriptor restore 输入准备
  - 长 clip 分段辅助与 cache 释放
- `lada/restorationpipeline/frame_restorer_blend.py`
  - 单帧 blend
  - batch blend
  - Vulkan blend bridge 接入

### 3.2 BasicVSR++ Vulkan runtime

- `lada/restorationpipeline/basicvsrpp_vulkan_restorer.py`
  - runtime orchestration shell
  - artifact 加载、runner 初始化、feature 探测、对外 API
- `lada/restorationpipeline/basicvsrpp_vulkan_restore_paths.py`
  - restore path 主体
  - Vulkan frame preprocess、特征准备、fused restore、recurrent modular、native clip runner、固定窗口 restore
- `lada/restorationpipeline/basicvsrpp_vulkan_blend.py`
  - Vulkan blend patch / padded blend / batch blend bridge
- `lada/restorationpipeline/basicvsrpp_vulkan_io.py`
  - 输入字典构造、输出拆包、runtime tensor 形状转换、GPU bridge 错误归类
- `lada/restorationpipeline/basicvsrpp_vulkan_common.py`
  - 共享常量、artifact 路径、`ncnn/pnnx` 导入与 artifact 校验
- `lada/restorationpipeline/basicvsrpp_vulkan_export.py`
  - PyTorch 子模块导出为 `ncnn` artifact
- `lada/restorationpipeline/basicvsrpp_vulkan_artifacts.py`
  - artifact 缓存、复用、失效重建

### 3.3 Native runtime

- `native/ncnn_vulkan_runtime/`
  - 本地 `ncnn` Vulkan runtime 工程
- `scripts/build_ncnn_vulkan_runtime.ps1`
  - Windows 构建入口

## 4. 端到端运行链路

### 4.1 设备探测

`lada/compute_targets.py` 在 `include_experimental=True` 时暴露 `vulkan:0`。可用条件包括：

- 系统存在 Vulkan loader
- Python 可以导入 `ncnn`
- `ncnn.get_gpu_count() > 0`

### 4.2 导入本地 runtime

`lada/models/basicvsrpp/ncnn_vulkan.py` 会优先尝试导入：

- `native/ncnn_vulkan_runtime/build/local_runtime/`

这一步非常关键，因为本地 runtime 才能同时提供：

- `torchvision.deform_conv2d` 自定义层
- `lada.GridSample` 自定义层
- `LadaVulkanNetRunner`
- `LadaVulkanTensor`
- Vulkan blend patch 系列接口

### 4.3 Artifact 导出与加载

首次运行或缓存失效时，Vulkan 路径会：

1. 用 PyTorch 加载 `BasicVSR++` 权重
2. 用 `pnnx` 导出模块化 `ncnn` artifact
3. 对导出的 `param` 做补丁修正
4. 回读并校验 artifact

### 4.4 Runtime feature 探测

`basicvsrpp_vulkan_restorer.py` 初始化时会收束 `runtime_features`，包括：

- GPU blob bridge
- Vulkan frame preprocess
- fused `restore_clip`
- recurrent modular runtime
- native `BasicVsrppClipRunner`
- Vulkan blend patch / preprocess / batch 接口

### 4.5 Restore path 分发

对外只有两个正式入口：

- `restore(video, max_frames=-1)`
- `restore_cropped_clip_frames(...)`

但内部会在 `basicvsrpp_vulkan_restore_paths.py` 里根据 clip 形态与 runtime features 分发到：

- fixed-size fused `restore_clip`
- recurrent modular path
- native clip runner path
- 固定 `5` 帧窗口的 sliding restore path

### 4.6 Blend

如果本地 runtime 暴露了 Vulkan blend 接口，上层会优先走：

- `blend_patch_gpu_inplace`
- `blend_patch_gpu_preprocess_inplace`
- `blend_patch_gpu_preprocess_inplace_batch`

否则才退回到本地 tensor / numpy bridge。

## 5. GPU 与 CPU 的边界

### 5.1 当前明确在 GPU 上的部分

- `quarter_downsample`
- `feat_extract`
- `spynet`
- fused `restore_clip` 或 recurrent/native clip runner 的核心计算
- `deformconv` 自定义层
- `GridSample` 自定义层
- 模块间 GPU blob bridge
- Vulkan frame preprocess
- Vulkan blend patch / preprocess / batch blend

### 5.2 当前仍然明确依赖 CPU 的部分

- 首次 artifact 导出与 `pnnx` 转换
- 媒体 decode、scene / clip 组装、Python 队列调度
- 某些桥接场景下的 numpy / tensor materialization
- 最终 PyAV / FFmpeg 写盘链路中的 CPU-side 交接

要点是：当前主路径已经尽量把推理与 blend 热点留在 GPU，但整条视频流水线不可能完全没有 CPU。

## 6. 当前实现约束

- 正式修复图固定为 `5f`
- 长 clip 默认按 `60` 帧 streaming segment 恢复
- CLI 写盘走异步 `VideoWriter`
- 旧 `1f` 兼容 backend 不再保留
- 缺少自定义层或本地 runtime 能力时，`vulkan:0` 会直接视为环境不满足，而不是重新回退到旧兼容方案

## 7. 常见坑

### 7.1 `ncnn` 能导入，但不是本地 runtime

现象：

- Vulkan backend 看起来可用
- 但没有 `LadaVulkanNetRunner`
- 没有自定义 `deformconv`
- 没有自定义 `GridSample`
- 没有 Vulkan blend patch 系列接口

排查重点：

- `lada/models/basicvsrpp/ncnn_vulkan.py`
- `native/ncnn_vulkan_runtime/build/local_runtime/`

### 7.2 只装了 `ncnn`，没装 `pnnx`

现象：

- 首次构建 artifact 时直接失败

原因：

- `pnnx` 是 artifact 导出阶段的硬依赖

### 7.3 误以为 `frame_count` 可以随便改

当前正式 runtime 是固定 `5` 帧导出图。若以后要做可变帧长，需要重新设计：

- 导出逻辑
- artifact 命名
- clip 调度
- 输出拆包
- runtime path 选择

### 7.4 首次运行很慢，不代表正式推理慢

首次运行慢通常包含：

- artifact 导出
- `pnnx` 转换
- native runtime 冷启动

批量处理多个视频时，这部分冷启动成本会被摊薄。

## 8. 维护建议

- 以固定 `5f Vulkan` 为正式路径，不再重新引入旧兼容分支
- 以本地 `ncnn` runtime 为中心，不把 pip `ncnn` 当成同等能力实现
- 继续沿着“减少 CPU fallback、减少 bridge 开销、保持边界清晰”的方向维护
- 优先看文档、基准脚本和实际运行结果，不靠猜测判断是否真的走了 GPU
- 出现结构问题时，优先沿着 `frame_restorer*` 与 `basicvsrpp_vulkan_*` 的职责边界排查，而不是继续把逻辑堆回大文件
