# Torch / CUDA 源码基线（基于 `D:\Code\github\lada`）

> 这份文档是 `lada-vulkan` 的 **Torch / CUDA 行为基线**。以后凡是修改 `vulkan:0`、检测链、恢复链、descriptor、blend、导出图、算子拆分、数据流拼装，都必须先回到 `D:\Code\github\lada` 的对应源码，再按本文对齐。
>
> **判断优先级：源码 > 本文 > 现有 Vulkan 实现 > 历史结论。** 只要 Vulkan 行为和参考源码有差异，就以参考源码为准继续修。

## 1. 文档目标与适用范围

### 1.1 目标

这份文档解决三个问题：

1. 说明 `D:\Code\github\lada` 里 Torch / CUDA 参考实现的真实调用链。
2. 固化 `lada-vulkan` 后续对齐时必须遵守的数据契约、时序语义和算子语义。
3. 给每一次 Vulkan 修改提供一张“应该对谁负责”的源码地图。

### 1.2 读者

- 维护 `lada-vulkan` 的开发者。
- 需要把 `vulkan:0` 和 `cuda:0` / Torch 路径对齐的实现者。
- 需要判断“当前差异到底是 Vulkan 问题，还是参考链理解错误”的排查者。

### 1.3 适用范围

本文覆盖以下参考链路：

- CLI 入口与设备选择。
- Torch 检测模型装配与预处理 / 后处理。
- `MosaicDetector` 的 scene / clip 生成逻辑。
- `FrameRestorer` 的 worker 编排、clip 恢复、blend 写回、输出编码。
- `BasicVSR++` 的 Torch 参考前向：`compute_flow -> propagate -> upsample`。
- 视频读写、音频合流、编码 preset 这些会影响最终输出一致性的外围逻辑。

本文不展开训练流程、数据集制作细节和 GUI 交互细节；它们不是 `vulkan:0` 语义对齐的核心真值。

## 2. 这份基线应该怎么用

### 2.1 每次改 Vulkan 前先做的事

每次准备修改 `lada-vulkan` 时，必须先完成这三个动作：

1. 找到当前改动在参考仓库里的对应层级。
2. 明确这一层的输入 / 输出 shape、dtype、通道顺序、时序顺序。
3. 明确这一层属于“必须逐像素对齐”还是“允许数值近似但语义一致”。

### 2.2 对齐时的硬规则

- 不能把 CPU fallback 当成 Vulkan 对齐完成的答案。
- 不能因为 Vulkan 图不好导，就改参考语义去适配导出图。
- 不能把“当前 Vulkan 已经能跑通”当成正确；只有和参考源码一致才算正确。
- 不能依赖旧文档、口头结论或之前会话摘要覆盖源码事实。

### 2.3 推荐的对照顺序

当某个 Vulkan 输出异常时，按下面顺序回查：

1. `D:\Code\github\lada\lada\cli\main.py`
2. `D:\Code\github\lada\lada\restorationpipeline\__init__.py`
3. `D:\Code\github\lada\lada\restorationpipeline\mosaic_detector.py`
4. `D:\Code\github\lada\lada\restorationpipeline\frame_restorer.py`
5. `D:\Code\github\lada\lada\restorationpipeline\basicvsrpp_mosaic_restorer.py`
6. `D:\Code\github\lada\lada\models\basicvsrpp\inference.py`
7. `D:\Code\github\lada\lada\models\basicvsrpp\mmagic\basicvsr_plusplus_net.py`
8. `D:\Code\github\lada\lada\utils\image_utils.py` / `mask_utils.py` / `scene_utils.py` / `video_utils.py` / `audio_utils.py`

## 3. 参考源码总表

| 层级 | 参考文件 | 作用 | Vulkan 对齐时关注点 |
| --- | --- | --- | --- |
| CLI 入口 | `D:\Code\github\lada\lada\cli\main.py` | 参数解析、设备选择、单文件处理入口 | `--device`、`--fp16`、编码器选择、输出路径 |
| 模型装配 | `D:\Code\github\lada\lada\restorationpipeline\__init__.py` | 选择 YOLO 与恢复模型、确定 `pad_mode` | `basicvsrpp -> pad_mode='zero'` |
| 检测模型 | `D:\Code\github\lada\lada\models\yolo\yolo11_segmentation_model.py` | Torch YOLO11 分割模型、预处理、推理、后处理 | CPU / GPU 预处理差异、letterbox、mask 还原 |
| 检测结果转换 | `D:\Code\github\lada\lada\utils\ultralytics_utils.py` | YOLO box / mask 转内部格式 | `box` 轴顺序、mask 二值化与 unpad |
| Scene / Clip | `D:\Code\github\lada\lada\restorationpipeline\mosaic_detector.py` | scene 聚合、clip 裁剪、统一缩放 / pad | `belongs()`、`crop_to_box_v3()`、clip 统一尺度 |
| 裁剪规则 | `D:\Code\github\lada\lada\utils\scene_utils.py` | box 扩张与 crop 逻辑 | border、扩张预算、最终 crop box |
| 图像算子 | `D:\Code\github\lada\lada\utils\image_utils.py` | resize / pad / unpad / tensor 转换 | CPU / GPU 路径差异、双线性 / 最近邻 |
| mask 算子 | `D:\Code\github\lada\lada\utils\mask_utils.py` | blend mask、mask 清理与扩张 | `create_blend_mask()` |
| 恢复编排 | `D:\Code\github\lada\lada\restorationpipeline\frame_restorer.py` | detection / restoration / blend / 输出主控 | worker 顺序、clip buffer、ROI blend |
| Torch 恢复包装 | `D:\Code\github\lada\lada\restorationpipeline\basicvsrpp_mosaic_restorer.py` | clip 输入转 BTCHW，再调用 Torch 模型 | HWC BGR uint8 -> BTCHW float |
| 模型加载 | `D:\Code\github\lada\lada\models\basicvsrpp\inference.py` | 构造 `BasicVSRPlusPlusGan`、加载 checkpoint | `is_use_ema=True`、`model.half()` |
| Torch 模型外层 | `D:\Code\github\lada\lada\models\basicvsrpp\basicvsrpp_gan.py` | `BasicVSRPlusPlusGanNet` 与 GAN wrapper | generator / generator_ema 关系 |
| 真正网络 | `D:\Code\github\lada\lada\models\basicvsrpp\mmagic\basicvsr_plusplus_net.py` | `compute_flow`、`propagate`、`upsample` 真正语义 | 这是 Vulkan 子图拆分的最高优先级真值 |
| 视频 IO | `D:\Code\github\lada\lada\utils\video_utils.py` | `VideoReader`、`VideoWriter`、编码 preset | BGR / RGB、PTS 重排、损坏帧复制 |
| 音频合流 | `D:\Code\github\lada\lada\utils\audio_utils.py` | 把临时视频与原音频合流 | `start_pts` 偏移、容器兼容性 |

## 4. 端到端调用图

```text
lada/cli/main.py::main()
  -> 解析 input / output / device / fp16 / encoder / model names
  -> restorationpipeline.__init__.py::load_models()
     -> Yolo11SegmentationModel(...)
     -> BasicvsrppMosaicRestorer(...)
  -> process_video_file(...)
     -> FrameRestorer(...)
     -> FrameRestorer.start()
        -> MosaicDetector.start()
           -> frame feeder worker
           -> frame inference worker
           -> frame detector worker
        -> clip restoration worker
        -> frame restoration worker
     -> VideoWriter.write(...)
     -> audio_utils.combine_audio_video_files(...)
```

这条链路说明一件事：

**参考实现不是“纯模型前向”，而是一整条从解码到贴回再到 mux 的完整 Torch / CUDA 流水线。**

因此后续任何 Vulkan 对齐，都不能只盯 `BasicVSR++`，还要对齐：

- clip 构造方式；
- resize / pad 方式；
- blend 写回方式；
- 视频与音频输出方式。

## 5. 设备、dtype 与 Torch / CUDA 语义

### 5.1 默认设备选择

参考实现的默认设备逻辑在 `D:\Code\github\lada\lada\utils\os_utils.py`：

- `cuda:0` 优先。
- 其次是 `mps`。
- 再其次是 `xpu:0`。
- 最后才是 `cpu`。

对应函数：

- `get_default_torch_device()`
- `gpu_has_fp16_acceleration()`

结论：

- 在参考项目里，“GPU 基线”默认首先就是 `cuda:0`。
- `fp16` 不是任意默认打开；它取决于设备能力。

### 5.2 Torch / CUDA 路径里哪些环节真的在 GPU 上

即使用户传的是 `--device cuda:0`，参考实现也不是“全链路 GPU 常驻”。

真实情况是：

- `VideoReader.frames()` 返回的是 **CPU 上的** `torch.Tensor(H, W, C)`。
- `Scene`、`Clip` 里的 frame / mask 主数据流，默认也还是 CPU tensor。
- YOLO 推理前，会把 batch `.to(device)` 送到 GPU。
- BasicVSR++ 推理前，会把 clip batch `.to(device)` 送到 GPU。
- blend 是否走 GPU，取决于 `_restore_frame()` 里当前 `frame.device.type`。CLI 默认从 `VideoReader` 拿到的是 CPU frame，所以参考 CLI 主路径的 blend 实际上还是 CPU。

这条事实非常关键：

**`cuda:0` 基线的“模型语义”在 GPU 上，但“解码、clip 组织、最终写回”并不天然是 GPU-only。**

所以 `lada-vulkan` 不能把“我把更多东西搬上 GPU 了”当成语义正确的理由；必须以最终行为一致为准。

## 6. CLI 与模型装配基线

### 6.1 CLI 入口

参考入口是 `D:\Code\github\lada\lada\cli\main.py`。

关键参数：

- `--device`
- `--fp16`
- `--mosaic-restoration-model`
- `--mosaic-restoration-config-path`
- `--mosaic-detection-model`
- `--max-clip-length`
- `--encoding-preset` 或 `--encoder` + `--encoder-options`

### 6.2 `process_video_file()` 的真实职责

`process_video_file()` 做的不是“跑模型然后保存”。它实际负责：

1. 读取视频 metadata。
2. 创建 `FrameRestorer`。
3. 创建临时视频输出路径。
4. 启动 `FrameRestorer`。
5. 逐帧从 `FrameRestorer` 取 `(restored_frame, restored_frame_pts)`。
6. 写入 `VideoWriter`。
7. 结束后调用 `combine_audio_video_files()`。

对齐含义：

- 任何 Vulkan CLI 输出对比，都不能只看 patch，还要看写出和 mux 后的最终文件。

### 6.3 `load_models()` 的真实分支

参考实现的装配逻辑在 `D:\Code\github\lada\lada\restorationpipeline\__init__.py`。

对 `basicvsrpp*` 模型名，固定行为是：

- `load_model(...)`
- `BasicvsrppMosaicRestorer(...)`
- `pad_mode = 'zero'`

这意味着：

- 当 Vulkan 走 `BasicVSR++` 路径时，descriptor / clip resize 后的 pad 语义必须优先对齐 `zero`。
- 如果当前 Vulkan 在某些地方用的是 `reflect` 或其他 pad 语义，就已经偏离参考。

## 7. 检测链基线：YOLO11 Segmentation

### 7.1 模型初始化

`D:\Code\github\lada\lada\models\yolo\yolo11_segmentation_model.py`

初始化时会完成：

- `YOLO(model_path)`
- `AutoBackend(...)`
- `check_imgsz(imgsz, stride=32)`
- `self.letterbox` 初始化
- `self.model.warmup(imgsz=(1, 3, *self.imgsz))`
- `self.dtype = torch.float16 if fp16 else torch.float32`

因此 Vulkan 对齐时，检测链不能只参考输出 boxes / masks，还要对齐：

- letterbox 尺寸；
- stride；
- warmup 后真实模型 backend 行为；
- 输入 batch 的 dtype 和归一化时机。

### 7.2 预处理：CPU 输入与 GPU 输入分两条路

`preprocess()` 有两条路径：

#### CPU 输入路径

当 `imgs[0].device.type == 'cpu'`：

- 走 `_preprocess_cpu()`；
- 对每帧调用 `LetterBox`；
- 把 `BHWC` 转成 `BCHW`；
- `np.ascontiguousarray()`；
- `torch.from_numpy(...)`。

#### GPU 输入路径

当输入已经在 GPU：

- 动态构造或复用 `PyTorchLetterBox`；
- 直接在 Torch tensor 上做 letterbox。

对齐含义：

- 参考工程本身已经区分 CPU / GPU preprocess。
- 如果 `lada-vulkan` 改检测前处理，一定要先判断自己是在对齐 CPU 输入语义，还是对齐 GPU 输入语义，而不是混写成第三套规则。

### 7.3 推理与后处理

`inference_and_postprocess()` 的顺序是：

1. `imgs.to(device=self.device)`
2. `.to(dtype=self.dtype)`
3. `.div_(255.0)`
4. `self.model(image_batch)`
5. `postprocess(...)`

这里最重要的是归一化时机：

- 参考代码不是在 `preprocess()` 里除以 `255`；
- 它是在真正送入 backend 前才 `.div_(255.0)`。

如果 Vulkan 路径提前或延后归一化，哪怕最后数值范围也是 `[0, 1]`，也可能改变后续计算精度与 rounding 行为。

### 7.4 box / mask 转内部格式

`D:\Code\github\lada\lada\utils\ultralytics_utils.py` 里有两条关键转换：

- `convert_yolo_box(...)`
- `convert_yolo_mask_tensor(...)`

内部契约是：

- box 统一成 `(top, left, bottom, right)`。
- mask 最终是 `torch.Tensor(H, W, 1)`。
- mask 会经过 `scale_and_unpad_image()` 还原到原图 shape。
- mask 最终阈值化成 `0` / `255`，dtype `torch.uint8`。

对齐含义：

- 后续 `Scene`、`Clip`、blend 全都依赖这个内部 box / mask 契约。
- 这里一旦偏离，后面所有 crop、resize、blend 都会错位。

## 8. Scene / Clip 基线

### 8.1 Scene 的职责

`D:\Code\github\lada\lada\restorationpipeline\mosaic_detector.py::Scene`

`Scene` 是按时间聚合的检测片段，保存：

- `frames`
- `masks`
- `boxes`
- `frame_start`
- `frame_end`

### 8.2 `belongs()` 的真实含义

`Scene.belongs(box)` 不是 tracking id，也不是 IoU 学习器；它只是：

- 拿当前 box 和 `scene.boxes[-1]`；
- 调 `box_overlap(...)`；
- 能接住就认为属于同一 scene。

这条规则很朴素，但它就是参考真值。Vulkan 不能擅自引入更复杂的 tracking 规则来“优化” scene 组织。

### 8.3 同帧 merge 与跨帧 append

当同一帧命中已有 scene：

- 走 `merge_mask_box()`；
- box 取并集；
- mask 取 `torch.maximum(...)`。

当下一帧延续已有 scene：

- 走 `add_frame()`；
- 要求 `frame_num == frame_end + 1`。

这意味着：

- 参考实现默认 scene 是**严格连续帧段**；
- 中间断一帧就不是同一条 scene。

### 8.4 `Clip` 的真实语义：构造时就完成裁剪、缩放与 pad

`Clip` 在 `mosaic_detector.py` 里直接做两件事：

1. 按检测 box 调 `crop_to_box_v3()` 裁出每一帧 patch。
2. 统计整个 clip 的最大宽高，再把整段 clip 统一缩放到同一个参考尺度，最后 pad 到固定 `size`。

这点必须牢牢记住：

**参考实现不是“每帧独立 resize 到 256”；而是“整段 clip 共享同一套缩放基准”。**

### 8.5 `crop_to_box_v3()` 的关键语义

`D:\Code\github\lada\lada\utils\scene_utils.py::crop_to_box_v3()` 的关键规则：

- 输入 box 是 `(top, left, bottom, right)`。
- 若配置了 `border_size`，会先向外扩一圈上下文。
- 计算在 `target_size` 下还能补多少宽 / 高。
- 在不越界且不超过 `max_box_expansion_factor` 预算的前提下，把 box 再向四周扩。
- 最终返回：
  - cropped image
  - cropped mask
  - cropped box
  - scale factor

这段逻辑直接决定 patch 的上下文范围。Vulkan 如果只对齐了最终 `256x256` 输入，而没有对齐 box 扩张规则，也会出现“看起来像对齐了，实际上 patch 内容不同”的问题。

### 8.6 `Clip` 的统一缩放规则

`Clip.__init__()` 里先统计整段 clip 的：

- `max_width`
- `max_height`

然后计算：

- `scale_width = size / max_width`
- `scale_height = size / max_height`

每一帧都用同一对比例去算自己的 `resize_shape`。

对齐含义：

- Vulkan descriptor / materialize 路径必须保留“整 clip 统一参考尺度”的行为。
- 不能把它偷换成“每帧按自身 box 自适应填满目标尺寸”。

### 8.7 `image_utils.resize()` 与 `pad_image()` 的真实分支

`D:\Code\github\lada\lada\utils\image_utils.py`

#### `resize()`

- CPU tensor / numpy 路径：走 OpenCV `cv2.resize`。
- CUDA / XPU / MPS tensor 路径：走 Torch `Resize(..., antialias=False)`。

#### `pad_image()`

- CPU 路径：最终落到 `np.pad(...)`。
- GPU 路径：最终落到 `F.pad(...)` 或 `_torch_pad_reflect(...)`。

这意味着：

- Torch / CUDA 参考链本身就包含 CPU / GPU 两套底层实现。
- 如果 Vulkan 要对齐 `cuda:0`，优先要看当时对应的数据到底落在 CPU 还是 GPU 上，而不是机械地只认函数名。

## 9. `FrameRestorer` 基线

### 9.1 线程与队列结构

`D:\Code\github\lada\lada\restorationpipeline\frame_restorer.py`

主队列有四个：

- `frame_detection_queue`
- `mosaic_clip_queue`
- `restored_clip_queue`
- `frame_restoration_queue`

线程有五段：

- `frame feeder worker`
- `frame inference worker`
- `frame detector worker`
- `clip restoration worker`
- `frame restoration worker`

### 9.2 双解码事实

参考链里视频会被解码两次：

1. `MosaicDetector` 自己开一次 `VideoReader` 读帧做检测。
2. `FrameRestorer._frame_restoration_worker()` 再开一次 `VideoReader` 读帧做最终写回。

所以参考真相不是“检测时读到的 frame 直接传给最终输出”。

这件事会影响：

- frame / pts 同步方式；
- 损坏帧 duplicate 行为；
- 某些 cache / reuse 优化是否仍然符合参考语义。

### 9.3 clip 恢复 worker 的职责

`_clip_restoration_worker()` 很简单：

1. 从 `mosaic_clip_queue` 拿 `Clip`。
2. `_restore_clip(clip)`。
3. 把 clip 放到 `restored_clip_queue`。

它不负责 blend，也不负责输出。

### 9.4 `_restore_clip()` 的真实语义

- 若开启 `mosaic_detection`，仅画检测框。
- 否则调用 `_restore_clip_frames(clip.frames)`。
- 恢复结果会**原位覆盖** `clip.frames[i]`。

这意味着：

- `Clip` 在参考实现里是“可变对象 + 流式消费对象”；
- 恢复后 patch 会直接替换原 patch，而不是额外产生新 descriptor。

### 9.5 `_restore_frame()` 的真实 blend 公式

`_restore_frame()` 做的步骤顺序不能改：

1. `clip.pop()` 取当前帧 patch / mask / box / crop_shape / pad_after_resize。
2. `unpad_image(clip_img, pad_after_resize)`。
3. `unpad_image(clip_mask, pad_after_resize)`。
4. `resize(clip_img, orig_crop_shape[:2])`。
5. `resize(clip_mask, orig_crop_shape[:2], INTER_NEAREST)`。
6. `create_blend_mask(clip_mask.float())`。
7. 把 blend mask 与 patch 写回整帧 ROI。

CPU 路径的公式是：

```text
frame_roi = frame[t:b+1, l:r+1, :]
out = frame_roi + (clip_img - frame_roi) * blend_mask
```

这里 `blend_mask` 会在最后一维 broadcast 到 3 个颜色通道。

如果 Vulkan 写回顺序、unpad / resize 顺序、mask 插值方式、blend 公式任何一个环节不同，最终帧就不可能完全一致。

### 9.6 `create_blend_mask()` 的关键语义

`D:\Code\github\lada\lada\utils\mask_utils.py::create_blend_mask()` 的规则是：

- 先根据 `crop_mask` 建一个内层全 1 矩形。
- 外围 padding 为 0。
- 再和原二值 mask 做 `torch.maximum(...)`。
- 最后用均值核做一次模糊平滑。

这不是简单的高斯 blur 整张 mask，也不是固定 feather 半径。它和 patch 尺寸绑定。

## 10. BasicVSR++ Torch / CUDA 基线

### 10.1 模型容器关系

模型装载文件是 `D:\Code\github\lada\lada\models\basicvsrpp\inference.py`。

关键事实：

- 默认 config 构造的是 `BasicVSRPlusPlusGan`。
- `get_default_gan_inference_config()` 里 `is_use_ema=True`。
- `load_model()` 里会 `MODELS.build(config)`、`load_checkpoint(...)`、`model.to(device).eval()`。
- 如果启用 `fp16`，会 `model.half()`。

`BasicVSRPlusPlusGan` 继承自 `RealBasicVSR`。`RealBasicVSR.forward_tensor()` 的逻辑是：

- 训练或 `is_use_ema=False` 时，用 `self.generator(inputs)`；
- 推理且 `is_use_ema=True` 时，用 `self.generator_ema(inputs)`。

因此在参考推理链里，**真正的生成器语义以 `generator_ema` 为准**。

### 10.2 `BasicvsrppMosaicRestorer.restore()` 的输入输出契约

`D:\Code\github\lada\lada\restorationpipeline\basicvsrpp_mosaic_restorer.py`

输入契约：

- `list[torch.Tensor(H, W, C)]`
- `dtype=torch.uint8`
- 通道顺序是 `BGR`

内部转换：

1. `x.permute(2, 0, 1)` -> `CHW`
2. `torch.stack(..., dim=0)` -> `TCHW`
3. `.to(device).to(dtype).div_(255.0)` -> `float`
4. `.unsqueeze(0)` -> `BTCHW`
5. `self.model(inputs=inference_view)`
6. 输出乘 `255`、`round_()`、`clamp_()`、转回 `uint8`
7. `permute(0, 2, 3, 1)` -> `THWC`

对齐含义：

- Vulkan 的 BasicVSR++ 输入预处理，必须先对齐这个 BGR HWC uint8 -> BTCHW float 流程。
- 不能提前把通道变成 RGB，也不能在不同阶段做不同 rounding。

### 10.3 `BasicVSRPlusPlusNet.forward()` 的完整顺序

`D:\Code\github\lada\lada\models\basicvsrpp\mmagic\basicvsr_plusplus_net.py`

真正顺序是：

1. `lqs_downsample = interpolate(..., scale_factor=0.25, mode='bicubic')`
2. `feat_extract` 提取 `spatial` 特征。
3. `compute_flow(lqs_downsample)` 算前后向光流。
4. 按固定顺序做四条传播分支：
   - `backward_1`
   - `forward_1`
   - `backward_2`
   - `forward_2`
5. `upsample(lqs, feats)` 生成输出。

这就是 `vulkan:0` 拆图、导图、runtime 编排必须对齐的主轴。

### 10.4 `compute_flow()` 的真实语义

- 输入是下采样后的 `BTCHW`。
- `flows_backward = spynet(lqs_t, lqs_{t+1})`
- `flows_forward = spynet(lqs_{t+1}, lqs_t)`

方向定义必须保持一致：

- `backward` 分支用的是“当前到下一帧”的 backward flow。
- `forward` 分支用的是“当前到上一帧”的 forward flow。

如果 Vulkan 把这两个方向互换，后面 branch 名字不变也没有意义。

### 10.5 `SPyNet.forward()` 的真实语义

`SPyNet.forward()` 做三件事：

1. 把输入 resize 到 `32` 的倍数。
2. 跑 `compute_flow()`。
3. 把 flow resize 回原分辨率，并分别按 `w / w_up`、`h / h_up` 缩放 X / Y 分量。

因此 Vulkan 只对齐了“resize 到 32 倍数 + 跑 core SPyNet”还不够；还必须对齐：

- resize 回原尺寸；
- flow X / Y 的值缩放。

### 10.6 `propagate()` 的真实语义

`propagate()` 是整个 BasicVSR++ 对齐最重要的一段。

#### 帧与 flow 索引

- `frame_idx = [0, 1, ..., t]`
- `mapping_idx = [0, 1, ..., len(spatial)-1, len(spatial)-1, ..., 0]`
- backward 分支会把 `frame_idx` 反转。

#### 每一步做什么

对于每个 step：

- 取 `feat_current = feats['spatial'][mapping_idx[idx]]`
- 若 `i == 0`：没有 deform align，直接走 backbone
- 若 `i > 0`：
  - 取 `flow_n1`
  - `cond_n1 = flow_warp(feat_prop, flow_n1)`
  - 初始化 `feat_n2 = 0`、`flow_n2 = 0`、`cond_n2 = 0`
  - 若 `i > 1`：
    - `feat_n2 = feats[module_name][-2]`
    - `flow_n2 = flows[:, flow_idx[i - 1], ...]`
    - `flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1)`
    - `cond_n2 = flow_warp(feat_n2, flow_n2)`
  - `cond = cat([cond_n1, feat_current, cond_n2])`
  - `feat_prop = cat([feat_prop, feat_n2])`
  - `feat_prop = deform_align(module_name)(feat_prop, cond, flow_n1, flow_n2)`
- 然后把当前 step 需要的其他 branch feature 一起拼进 backbone 输入。
- `feat_prop = feat_prop + backbone(module_name)(feat)`
- append 到 `feats[module_name]`

#### 对 Vulkan 的直接约束

Vulkan 的 branch step 设计必须对齐这些细节：

- `i == 0` 与 `i > 0` 是两套不同语义。
- second-order 只在 `i > 1` 时启用。
- `cond` 的拼接顺序固定是：`cond_n1, feat_current, cond_n2`。
- backbone 输入顺序固定是：`feat_current + 其他 branch 当前帧特征 + feat_prop`。
- branch 执行顺序固定是 `backward_1 -> forward_1 -> backward_2 -> forward_2`。

### 10.7 `upsample()` 的真实语义

`upsample()` 不是简单把四条分支特征拼起来做输出，它还依赖 branch list 的消费顺序：

1. 对每个输出帧 `i`：
   - 对所有非 `spatial` 的 branch，执行 `feats[k].pop(0)`；
   - 再在最前面插入 `feats['spatial'][mapping_idx[i]]`。
2. `torch.cat(...)`
3. `reconstruction`
4. `upsample1`
5. `upsample2`
6. `conv_hr`
7. `conv_last`
8. `hr += lqs[:, i, :, :, :]`

这意味着：

- branch 输出 list 的排列方向必须和参考完全一致；
- 不能只让每帧 branch tensor 数值近似一致，还要保证最终消费顺序一致。

## 11. 视频读写与输出文件基线

### 11.1 `VideoReader.frames()`

`D:\Code\github\lada\lada\utils\video_utils.py::VideoReader.frames()` 的关键事实：

- 用 PyAV packet-level demux / decode。
- 每帧转成 `bgr24`。
- 返回 `torch.from_numpy(nd_frame)`。
- 遇到 `InvalidDataError` 时，会复制上一帧，最多连续复制 `10` 次。

对齐含义：

- 参考实现不是“遇到坏帧就跳过”。
- Vulkan 如果在坏帧场景下行为不同，会影响时序和 clip 组织。

### 11.2 `VideoWriter.write()`

- 接收 `torch.Tensor` 或 `np.ndarray`。
- 若 `bgr2rgb=True`，先做 `cv2.cvtColor`。
- 用 FIFO frame queue + min heap 的方式给 frame 与 PTS 配对，避免问题视频的 PTS 乱序。

所以最终文件一致性不只是像素问题，也包括 PTS 排布。

### 11.3 `combine_audio_video_files()`

音频合流逻辑在 `D:\Code\github\lada\lada\utils\audio_utils.py`：

- 先读原视频第一路音频编码。
- 判断输出容器是否兼容该音频编码。
- 若不兼容，只 copy 视频流，音频重编码。
- 若兼容，`-c copy` 直接 mux。
- 若 `start_pts > 0`，通过 `-itsoffset` 给视频流加偏移。
- 原视频无音频时，直接 copy 临时视频。

这部分不会改变恢复 patch 本身，但会影响最终导出文件是否和参考 CLI 行为一致。

## 12. `lada-vulkan` 必须遵守的基线约束

下面这些约束不是建议，而是后续修改的硬门槛。

### 12.1 数据契约约束

- 检测前原始 frame 基线：`torch.Tensor(H, W, C, uint8, BGR)`。
- BasicVSR++ patch 输入基线：`list[torch.Tensor(H, W, C, uint8, BGR)]`。
- 模型前实际输入基线：`BTCHW float`，值域 `[0, 1]`。
- box 基线：`(top, left, bottom, right)`。
- mask 基线：`HWC(1)`、`uint8`、值域 `0/255`。

### 12.2 时序约束

- scene 是连续帧段，不允许中途断帧后继续并入旧 scene。
- clip 内所有帧共享统一缩放参考尺寸。
- `BasicVSR++` branch 顺序固定，不能重排。
- `upsample()` 取 branch 特征时依赖 `pop(0)` 的消费顺序。

### 12.3 图像算子约束

- patch 写回前必须先 `unpad`，再 `resize` 回原 crop 尺寸。
- patch image 用双线性；mask 用最近邻。
- blend mask 语义以 `create_blend_mask()` 为准，不能替换成任意 feather 算法。
- `SPyNet.forward()` 必须包含 resize-back 与 X / Y flow scaling。

### 12.4 架构约束

- Vulkan 可以改变执行位置，但不能改变参考仓库的输入 / 输出契约。
- Vulkan 可以做 fused module，但 fused 前后必须保持与参考源码同一条数据流语义。
- 如果导出框架限制了某个参考行为，应该扩展导出 / runtime，而不是回退修改参考行为。

## 13. 推荐的 Vulkan 对齐映射

后续在 `lada-vulkan` 修改时，建议按下面映射回查：

| `lada-vulkan` 侧 | 参考基线 |
| --- | --- |
| `lada/restorationpipeline/basicvsrpp_vulkan_restore_paths.py` | `D:\Code\github\lada\lada\restorationpipeline\basicvsrpp_mosaic_restorer.py` + `frame_restorer.py` |
| `lada/restorationpipeline/frame_restorer_clip_ops.py` | `D:\Code\github\lada\lada\restorationpipeline\mosaic_detector.py::Clip` |
| `lada/restorationpipeline/frame_restorer_blend.py` | `D:\Code\github\lada\lada\restorationpipeline\frame_restorer.py::_restore_frame` |
| `lada/models/basicvsrpp/vulkan_runtime*.py` | `D:\Code\github\lada\lada\models\basicvsrpp\mmagic\basicvsr_plusplus_net.py` |
| `lada/models/yolo/*vulkan*` | `D:\Code\github\lada\lada\models\yolo\yolo11_segmentation_model.py` + `ultralytics_utils.py` |
| `lada/restorationpipeline/runtime_options.py` | 参考仓库的 CLI / runtime 使用习惯，而不是当前 Vulkan 私有假设 |

## 14. 后续修改的执行要求

从现在开始，`lada-vulkan` 的相关修改应该遵守下面流程：

1. **先定位**：明确改动属于检测、scene / clip、restore、blend、输出中的哪一层。
2. **再对照**：在 `D:\Code\github\lada` 里找到对应函数，确认输入 / 输出 / shape / dtype / 时序。
3. **再实现**：只允许把 Vulkan 实现改成更接近参考，不允许反向改参考理解去迁就 Vulkan。
4. **再验证**：验证时优先看 stage parity，再看最终 ROI parity，最后看 CLI 实样输出。

## 15. 一句话结论

`lada-vulkan` 的未来修改，不应该以“当前 Vulkan 怎么写”为起点，而应该以 `D:\Code\github\lada` 的 Torch / CUDA 源码为起点；`vulkan:0` 只是执行后端，**不是语义真值**。
