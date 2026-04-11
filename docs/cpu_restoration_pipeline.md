# CPU 恢复链路源码梳理（当前 `lada` 源码）

本文按当前 `D:/Code/github/lada` 源码重写，梳理 `--device cpu` 时，从输入文件进入，到输出文件落盘的实际恢复链路。

本文重点针对当前实际对照链路：

- 检测模型：`Torch + Ultralytics YOLO11 segmentation`
- 恢复模型：`Torch BasicVSR++`（`basicvsrpp-*`）
- CLI 入口：`lada/cli/main.py`
- 检测/恢复流水线：`lada/restorationpipeline/__init__.py`、`lada/restorationpipeline/mosaic_detector.py`、`lada/restorationpipeline/frame_restorer.py`

说明：

- 这不是理想设计图，而是当前仓库里的真实调用链。
- 旧版文档中提到的 `batch_runner.py`、`model_loader.py`、`restoration_backends.py`、`clip_units.py`、`frame_restorer_worker.py`、`frame_restorer_blend.py`、`runtime_options.py` 等路径，在当前源码里已不存在或已被内联/重构。
- 本文全部以当前代码实现为准，不再沿用旧文档里的模块拆分方式。
- `deepmosaics-*` 分支仍然存在，但本文只在模型加载和恢复入口处说明，重点仍放在 BasicVSR++ CPU 链路。

## 1. 总调用图

### 1.1 单文件 / 目录入口

```text
lada/cli/main.py::main()
  -> 解析参数 / 解析 device / 解析模型路径 / 解析编码器
  -> lada/restorationpipeline/__init__.py::load_models()
     -> 选择 restoration model（deepmosaics / basicvsrpp）
     -> 构造 Yolo11SegmentationModel
  -> lada/cli/utils.py::setup_input_and_output_paths()
  -> 对每个 input/output 对调用 lada/cli/main.py::process_video_file()
     -> get_video_meta_data()
     -> FrameRestorer(...)
     -> FrameRestorer.start()
        -> MosaicDetector.start()
           -> frame feeder worker
           -> frame inference worker
           -> frame detector worker
        -> clip restoration worker
        -> frame restoration worker
     -> for restored_frame in FrameRestorer: VideoWriter.write(...)
     -> audio_utils.combine_audio_video_files()
     -> 最终输出文件
```

### 1.2 当前实现与旧版文档的关键差异

- 当前没有独立的 `batch_runner.py`；目录输入只是 `main()` 里循环调用 `process_video_file()`。
- 当前没有 `ClipDescriptor -> materialize -> prepared_clip_queue` 这套链路；`MosaicDetector` 直接构造可恢复的 `Clip`。
- 当前 CPU 链路里的原始帧、Scene、Clip、BasicVSR++ 输入，主体上都是 `torch.Tensor(HWC, uint8, BGR)`，不是旧文档描述的 `numpy -> torch` 断层。
- 当前没有独立的 output metadata 构建逻辑，也没有旧文档里的“全视频 passthrough / remux”分支。

## 2. 入口层：参数、输入输出路径、处理范围

### 2.1 CLI 主入口

入口函数：

- `lada/cli/main.py::main()`

这里完成的事情是：

1. 解析参数。
2. 校验 `device`、输入路径、输出路径和编码器参数。
3. 把检测模型名和恢复模型名解析成真实权重路径。
4. 调用 `load_models()` 加载检测模型和恢复模型。
5. 调用 `setup_input_and_output_paths()` 生成输入/输出文件列表。
6. 对每个文件调用 `process_video_file()`。

### 2.2 输入输出路径

路径逻辑在：

- `lada/cli/utils.py::setup_input_and_output_paths()`
- `lada/cli/utils.py::_get_output_file_path()`

当前行为：

- 单文件输入：
  - 未指定 `--output` 时，输出目录默认取输入文件所在目录。
  - 如果 `--output` 是目录，则按 `--output-file-pattern` 生成文件名。
  - 如果 `--output` 是文件，则直接使用。
- 目录输入：
  - 先扫描目录下 MIME 类型为 `video/*` 的文件。
  - 再按 pattern 生成输出路径。
  - 然后 `main()` 逐个调用 `process_video_file()`。

结论：

- 当前目录模式不是“批处理子系统”，也没有状态文件、重试次数、backup 阶段之类的 batch 行为。
- 当前只是“单进程循环多个文件”。

## 3. 模型加载层

### 3.1 统一加载入口

函数：

- `lada/restorationpipeline/__init__.py::load_models()`

当前实现已经把旧文档里的 `model_loader.py` / `restoration_backends.py` 逻辑内联到 `__init__.py` 里。

### 3.2 当前恢复模型分支

`load_models()` 会按模型名前缀分支：

- `deepmosaics*`
  - 调用 `lada.models.deepmosaics.models.loadmodel.video()`
  - 包装为 `DeepmosaicsMosaicRestorer`
  - `pad_mode = 'reflect'`
- `basicvsrpp*`
  - 调用 `lada.models.basicvsrpp.inference.load_model()`
  - 包装为 `BasicvsrppMosaicRestorer`
  - `pad_mode = 'zero'`

检测模型始终通过：

- `lada.models.yolo.yolo11_segmentation_model.Yolo11SegmentationModel`

来构造。

### 3.3 BasicVSR++ CPU 模型加载

函数：

- `lada/models/basicvsrpp/inference.py::load_model()`

实际流程：

1. `register_all_modules()`
2. 无自定义 config 时，使用 `get_default_gan_inference_config()`
3. `MODELS.build(config)`
4. `load_checkpoint(..., map_location='cpu')`
5. `model.to(device).eval()`
6. 若启用 `fp16`，则 `model.half()`

这部分与旧文档描述基本一致，但实现位置与文件路径已经变化。

## 4. `process_video_file()`：单文件恢复主控

函数：

- `lada/cli/main.py::process_video_file()`

按当前代码，单文件处理流程是：

1. `get_video_meta_data(input_path)` 读取输入视频元信息。
2. 构造 `FrameRestorer(...)`。
3. 生成临时视频路径：
   - `os.path.join(temp_dir_path, f"{basename}.tmp{ext}")`
4. 创建输出目录。
5. `frame_restorer.start()`。
6. 用 `VideoWriter(...)` 打开临时视频文件。
7. 遍历 `FrameRestorer` 输出的 `(restored_frame, restored_frame_pts)` 并写入编码器。
8. 成功时调用 `audio_utils.combine_audio_video_files()` 做音频合流。
9. 失败时删除临时视频。

### 4.1 当前实现里不存在的旧逻辑

当前代码里没有：

- `_build_output_metadata()`
- `_is_full_video_passthrough()`
- `copy_or_remux_video_file()`
- `build_temp_video_output_path()`

也就是说：

- 当前不会额外构造 `title/comment/description/software` 之类的输出 metadata。
- 当前没有“完全未检测到 mosaic 时直接 passthrough 原视频”的专门分支。
- 当前成功路径统一走 `combine_audio_video_files()`。

## 5. `FrameRestorer`：CPU 恢复流水线总调度器

类：

- `lada/restorationpipeline/frame_restorer.py::FrameRestorer`

### 5.1 当前的关键队列

当前 `FrameRestorer` 只维护 4 个主队列：

- `frame_detection_queue`
  - 内容：`(frame_num, num_mosaics_detected)`
- `mosaic_clip_queue`
  - 内容：已经构造好的 `Clip`
- `restored_clip_queue`
  - 内容：已经跑完恢复模型的 `Clip`
- `frame_restoration_queue`
  - 内容：`(frame, frame_pts)`

和旧文档不同的是：

- 当前没有 `prepared_clip_queue`
- 当前没有 `ClipDescriptor`
- 当前 `frame_detection_queue` 不携带原始 frame，只携带“当前帧需要几个 clip”这个计数信息

### 5.2 当前的线程结构

`FrameRestorer.start()` 会启动：

1. `MosaicDetector.start()`
   - 内部启动：
     - `frame feeder worker`
     - `frame inference worker`
     - `frame detector worker`
2. `clip restoration worker`
3. `frame restoration worker`

### 5.3 一个很重要的实现事实：当前视频会被解码两次

这是当前实现与旧文档差异最大的点之一。

当前链路里：

- `MosaicDetector` 会自己打开一次 `VideoReader`，读取整段视频做检测。
- `FrameRestorer._frame_restoration_worker()` 也会再打开一次 `VideoReader`，读取整段视频拿完整帧和 PTS 做最终回写。

也就是说：

- 检测链和整帧回写链不是共享同一份 frame buffer。
- 它们是“两路独立解码 + 通过 `frame_num` 和 queue 同步”的结构。

## 6. 检测链：从视频文件读帧，到 scene / clip 发射

类：

- `lada/restorationpipeline/mosaic_detector.py::MosaicDetector`

### 6.1 三段 worker

`MosaicDetector` 当前内部固定分三段：

1. `frame feeder worker`
2. `frame inference worker`
3. `frame detector worker`

### 6.2 原始帧读取：`VideoReader.frames()`

函数：

- `lada/utils/video_utils.py::VideoReader.frames()`

当前真实行为：

1. 用 PyAV packet-level demux/decode 读取视频。
2. 每帧转换成 `bgr24`。
3. `torch.from_numpy(nd_frame)`，直接返回 `torch.Tensor`。
4. 若遇到 `InvalidDataError`：
   - 如果此前有成功帧，则复制上一帧继续产出。
   - 连续坏帧过多则报错终止。

因此当前 CPU 链路的原始帧源头是：

- `torch.Tensor`
- 形状：`HWC`
- 通道顺序：`BGR`
- dtype：`torch.uint8`

这和旧文档里“返回 numpy”已经不同。

### 6.3 CPU 检测预处理

函数：

- `lada/models/yolo/yolo11_segmentation_model.py::Yolo11SegmentationModel.preprocess()`
- `lada/models/yolo/yolo11_segmentation_model.py::_preprocess_cpu()`

当前 CPU 路径逻辑：

1. 输入是 `list[torch.Tensor(HWC)]`。
2. `x.numpy()` 取 CPU tensor 的 numpy 视图。
3. 对每帧做 `LetterBox`。
4. 把 `(N, H, W, C)` 转成 `(N, C, H, W)`。
5. `np.ascontiguousarray()`。
6. `torch.from_numpy(...)` 转回批量 tensor。

也就是说：

- 原始帧的公共数据契约是 `torch.Tensor`。
- 检测预处理内部会短暂转成 numpy 来复用 Ultralytics 的 letterbox 实现。
- 但这不会把上游 Scene / Clip 的主数据流改成 numpy。

### 6.4 检测推理与后处理

函数：

- `lada/models/yolo/yolo11_segmentation_model.py::inference_and_postprocess()`
- `lada/models/yolo/yolo11_segmentation_model.py::postprocess()`

流程：

1. `imgs.to(device=self.device).to(dtype=self.dtype).div_(255.0)`
2. `self.model(image_batch)` 跑 YOLO11 segmentation
3. 做 NMS、mask 还原、box 缩放
4. 返回 `Results` 列表

### 6.5 检测结果转内部 box / mask

函数：

- `lada/utils/ultralytics_utils.py::convert_yolo_box()`
- `lada/utils/ultralytics_utils.py::convert_yolo_mask_tensor()`

转换结果：

- box：`(top, left, bottom, right)`
- mask：`torch.Tensor(H, W, 1)`，值域 `0/255`，dtype `torch.uint8`

注意：

- 当前 mask 也已经是 tensor，不是旧文档里的 numpy mask。
- `Scene` 存的是 `torch frame + torch mask + box tuple`。

## 7. scene 聚合层：把逐帧检测结果组织成时序 clip

### 7.1 Scene 数据结构

类：

- `lada/restorationpipeline/mosaic_detector.py::Scene`

每个 `Scene` 保存：

- `frames`
- `masks`
- `boxes`
- `frame_start`
- `frame_end`

当前没有单独的 `resize_reference_shape`、`crop_boxes`、`ClipDescriptor` 这些结构。

### 7.2 每帧如何归入 scene

函数：

- `lada/restorationpipeline/mosaic_detector.py::_create_or_append_scenes_based_on_prediction_result()`

逻辑：

1. 遍历当前帧的每个检测结果。
2. 先把 YOLO box/mask 转成内部格式。
3. 遍历已有 `Scene`：
   - 如果 `scene.belongs(box)`：
     - 同一帧里已经命中过同一个 scene，则 `merge_mask_box()`
     - 否则说明是跨帧延续，`add_frame()`
4. 没有 scene 能接住时，新建 `Scene`。

当前 `belongs()` 的判断依据是：

- 对比这个检测 box 与 `scene` 最后一帧 box 的 `box_overlap()`
- 当前并没有显式 tracking id

### 7.3 scene 何时发射成 clip

函数：

- `lada/restorationpipeline/mosaic_detector.py::_create_clips_for_completed_scenes()`

发射条件：

- `scene.frame_end < frame_num`，说明 scene 已结束
- 或 `len(scene) >= max_clip_length`
- 或 `EOF`

发射时会直接构造：

- `Clip(completed_scene, clip_size, pad_mode, clip_id)`

因此当前 `mosaic_clip_queue` 里放的是：

- 已经完成 crop / resize / pad 的 `Clip`
- 不是旧文档里的 `ClipDescriptor`

## 8. `Clip`：CPU 恢复前的实际物化对象

类：

- `lada/restorationpipeline/mosaic_detector.py::Clip`

### 8.1 当前 `Clip` 在构造时就完成了裁剪与标准化

`Clip.__init__()` 里直接做了 2 组工作：

1. 按 box 计算 crop
2. 把整个 clip 统一 resize / pad 到固定大小

因此当前不存在旧文档里的：

- `ClipDescriptor -> materialize_clip_work_item()`
- `build_clip_resize_plans()`
- `materialize_clip_frames_with_profile()`
- `materialize_clip_masks_with_profile()`

这些逻辑已经折叠进当前 `Clip` 构造函数里了。

### 8.2 crop 规则

crop 调用的是：

- `lada/utils/scene_utils.py::crop_to_box_v3()`

参数固定为：

- `max_box_expansion_factor = 1.0`
- `border_size = 0.06`
- `target_size = (size, size)`，当前默认 `size = 256`

含义：

- 不是按检测 box 硬裁。
- 会在边界允许范围内向外扩 box，并保留一圈上下文。

### 8.3 resize / pad 规则

`Clip` 当前做法是：

1. 先统计整个 clip 内所有 crop 的最大宽高。
2. 用这组最大宽高算统一缩放比例。
3. 对每一帧 crop：
   - `image_utils.resize(..., INTER_LINEAR)`
   - `image_utils.resize(mask, ..., INTER_NEAREST)`
   - frame 用 `pad_mode` 补边
   - mask 一律 `zero` 补边

这点和旧文档“整段 clip 共享同一套参考尺度”这个大方向是一致的，只是实现不再拆成多个 helper 文件。

### 8.4 当前 `Clip.frames` / `Clip.masks` 的真实类型

这是当前 CPU 链路里最关键的事实之一。

当前：

- `Scene.frames` 来自 `VideoReader.frames()`，本来就是 `torch.Tensor`
- `image_utils.resize()` 在 CPU tensor 路径上会转 numpy 再转回 torch
- `image_utils.pad_image()` 在 CPU tensor 路径上也会转 numpy 再转回 torch

所以最终：

- `Clip.frames` 是 `list[torch.Tensor(HWC, uint8)]`
- `Clip.masks` 是 `list[torch.Tensor(HWC(1), uint8)]`

这意味着：

- 当前 BasicVSR++ CPU 恢复节点的上游输入类型是正确对齐的。
- 旧文档里“clip 还是 numpy，进 BasicVSR++ 前会因 `permute()` 失败”的结论，已经不适用于当前源码。

## 9. CPU 恢复节点

### 9.1 当前 clip 恢复 worker

函数：

- `lada/restorationpipeline/frame_restorer.py::_clip_restoration_worker()`

当前逻辑很直接：

1. 从 `mosaic_clip_queue` 取出 `Clip`
2. 调用 `_restore_clip(clip)`
3. 把恢复后的 `Clip` 推入 `restored_clip_queue`

当前不存在：

- `clip preprocess worker`
- `prepared_clip_queue`
- `descriptor direct restore`
- `stream_restore_chunk_size` 拆段逻辑

### 9.2 `BasicvsrppMosaicRestorer.restore()`

函数：

- `lada/restorationpipeline/basicvsrpp_mosaic_restorer.py::restore()`

当前输入假设与上游实际完全对齐：

- 输入：`list[torch.Tensor(HWC, uint8, BGR)]`
- 内部：
  1. `x.permute(2, 0, 1)` 变成 `TCHW`
  2. `.to(device).to(dtype).div_(255.0).unsqueeze(0)` 变成 `BTCHW`
  3. `self.model(inputs=...)`
  4. 输出再转回 `list[torch.Tensor(HWC, uint8)]`

注意：

- `restore()` 支持 `max_frames` 分块推理参数。
- 但当前 `FrameRestorer` 调用它时没有传 `max_frames`，所以当前 CPU BasicVSR++ 默认是整 clip 一次送入模型。

### 9.3 DeepMosaics 分支

`DeepmosaicsMosaicRestorer.restore()` 当前会：

1. 把每帧 tensor 转成 numpy
2. 调用 `restore_video_frames(...)`
3. 再把结果包回 tensor

所以当前两个恢复分支的输入输出契约是：

- 上游都拿 `Clip.frames` 这组 tensor
- DeepMosaics 在内部短暂转 numpy
- BasicVSR++ 全程吃 tensor

## 10. frame restoration worker：把恢复后的 clip 写回完整帧

函数：

- `lada/restorationpipeline/frame_restorer.py::_frame_restoration_worker()`

### 10.1 当前帧从哪里来

这里会再次打开：

- `VideoReader(self.video_meta_data.video_file)`

然后自己读取：

- `frame`
- `frame_pts`

再从 `frame_detection_queue` 取对应的：

- `detection_frame_num`
- `num_mosaics_detected`

也就是说，frame restoration worker 的同步方式是：

- 一路自己解码视频帧
- 一路从检测线程拿“当前帧需要几个 clip”
- 用 `frame_num` 断言两边严格同步

### 10.2 当前帧需要哪些 clip

辅助逻辑：

- `_clip_buffer_contains_all_cips_needed_for_current_restoration()`

判定规则：

- 当前 frame 需要的 clip 数量 = `num_mosaics_detected`
- clip buffer 里 `frame_start == current_frame_num` 的 clip 数量必须和它相等

够了之后才进入 blending。

### 10.3 blending 的当前真实实现

函数：

- `lada/restorationpipeline/frame_restorer.py::_restore_frame()`

对每个 `frame_start == frame_num` 的 clip：

1. `buffered_clip.pop()` 取出当前帧对应的：
   - `clip_img`
   - `clip_mask`
   - `orig_clip_box`
   - `orig_crop_shape`
   - `pad_after_resize`
2. `image_utils.unpad_image()` 去掉补边
3. `image_utils.resize()` 把 patch 和 mask 拉回原 crop 尺寸
4. `mask_utils.create_blend_mask(clip_mask.float())` 生成软融合 mask
5. 进入 `_blend_cpu()` 或 `_blend_gpu()`

### 10.4 CPU 混合写回

当前 CPU 路径会走 `_blend_cpu()`：

1. `blend_mask.cpu().numpy()`
2. `clip_img.cpu().numpy()`
3. `frame[t:b+1, l:r+1, :].numpy()` 取整帧 ROI 的 numpy 视图
4. 计算：

```text
out = frame_roi + (clip_img - frame_roi) * blend_mask
```

5. 结果写回 ROI

重要结论：

- 当前 full frame 自始至终就是 `torch.Tensor`，不需要旧文档里那种“blending 前先 `torch.from_numpy(...)`”的补救步骤。
- CPU blending 只是临时把 tensor 的底层内存当成 numpy 视图来做 in-place 混合。

### 10.5 `Clip.pop()` 的语义

`Clip.pop()` 当前会：

- 弹出当前头帧的 patch / mask / box / pad 信息
- `frame_start += 1`
- 空 clip 之后由 `_collect_garbage()` 清掉

因此当前 `Clip` 仍然是“流式消费对象”，这一点与旧文档的大方向一致。

## 11. 输出阶段：编码与音频合流

### 11.1 `FrameRestorer.__next__()`

函数：

- `lada/restorationpipeline/frame_restorer.py::__next__()`

当前行为：

- 从 `frame_restoration_queue` 取元素
- 遇到 `EOF_MARKER` 就 `StopIteration`
- 遇到 `STOP_MARKER` 或 `ErrorMarker` 则向上返回异常状态

当前没有旧文档里的 `emit_frame_restoration_output()` 中间层，也没有先转 numpy 再 `copy()` 的独立步骤。

### 11.2 `VideoWriter`

类：

- `lada/utils/video_utils.py::VideoWriter`

关键行为：

1. `write(frame, pts, bgr2rgb=True)` 如果收到的是 tensor，会先转 numpy。
2. 若 `bgr2rgb=True`，则转成 RGB。
3. 用“FIFO frame queue + 最小堆 pts”做轻量重排。
4. `_process_buffer()` 里把帧编码进 PyAV 容器。
5. `release()` 冲刷尾包并关闭容器。

因此当前编码器看到的最终帧是：

- `np.ndarray`
- `HWC`
- `RGB`
- `uint8`

### 11.3 `combine_audio_video_files()`

函数：

- `lada/utils/audio_utils.py::combine_audio_video_files()`

当前逻辑：

1. 检查原视频第一路音频编码。
2. 判断目标容器是否兼容该音频编码。
3. 若不兼容：
   - 只复制视频流，重编码音频流
4. 若兼容：
   - `-c copy` 直接 mux
5. 若 `start_pts > 0`：
   - 用 `-itsoffset` 对齐视频时间轴
6. 若原视频无音频：
   - 直接把临时视频复制为最终输出
7. 最后删除临时视频

注意：

- 当前只处理 `0:a:0` 这一路音频。
- 当前没有专门的 `copy_or_remux_video_file()` 辅助函数。
- 当前没有额外写入输出 metadata。

## 12. 贯穿整条 CPU 链路的数据形态

| 阶段 | 数据对象 | 主要类型 | 形态 |
| --- | --- | --- | --- |
| 视频解码 | 原始帧 | `torch.Tensor` | `HWC/BGR/uint8` |
| 检测预处理后 | `frames_batch` | `torch.Tensor` | `BCHW/uint8` |
| 检测模型输入 | `input_batch` | `torch.Tensor` | `BCHW/float/0..1` |
| 检测结果 mask | scene mask | `torch.Tensor` | `HWC(1)/uint8` |
| `Scene.frames` | 场景帧缓存 | `torch.Tensor` | `HWC/BGR/uint8` |
| `Scene.masks` | 场景 mask 缓存 | `torch.Tensor` | `HWC(1)/uint8` |
| `Clip.frames` | 物化后 patch | `torch.Tensor` | `HWC/BGR/uint8` |
| `Clip.masks` | 物化后 patch mask | `torch.Tensor` | `HWC(1)/uint8` |
| BasicVSR++ 输入 | 恢复模型输入 | `torch.Tensor` | `HWC/BGR/uint8` |
| blending 前 full frame | 当前完整帧 | `torch.Tensor` | `HWC/BGR/uint8` |
| `VideoWriter.write()` 内部 | 编码前帧 | `np.ndarray` | `HWC/RGB/uint8` |
| 临时视频文件 | `*.tmp{ext}` | 文件 | 仅视频流 |
| 最终输出文件 | 输出视频 | 文件 | 视频流 + 原音频（如存在） |

这张表对应的关键事实是：

- 当前 CPU BasicVSR++ 链路前后，主数据流已经对齐到 tensor。
- 旧文档里“`Clip.frames` 还是 numpy，BasicVSR++ 期望 torch”的断层，在当前代码里并不存在。

## 13. 用于 Vulkan 对照时，当前真正应该对齐的基线

如果后续要拿当前 CPU 链路给 Vulkan 版本做对照，最应该逐点对齐的是：

1. `VideoReader.frames()` 的输出契约是否一致。
2. Scene 聚合规则是否一致：`box_overlap()`、同帧 merge、跨帧 append。
3. `crop_to_box_v3()` 的 box 扩展、border 和目标尺寸策略是否一致。
4. `Clip` 内部的统一缩放和 pad 规则是否一致。
5. 恢复模型输入是否保持 `torch.Tensor(HWC, uint8, BGR)`。
6. `Clip.pop()` 的消费顺序是否一致。
7. blending 前是否先 unpad、再 resize 回原 crop 尺寸。
8. `create_blend_mask()` 的软边缘策略是否一致。
9. ROI 混合公式是否一致。
10. 编码与音频合流是否额外引入文件级差异。

## 14. 一句话总结

当前 `lada` 的 CPU 恢复链路可以概括为：

```text
CLI 解析与模型加载
  -> 检测链（独立解码一次视频，构造 Scene / Clip）
  -> 恢复链（直接对 Clip 跑 BasicVSR++ / DeepMosaics）
  -> 整帧回写链（再次解码一次视频，按 ROI soft blend）
  -> 视频编码 + 音频合流
```

其中最重要的源码事实是：

- 当前实现已经没有旧文档里的 `ClipDescriptor/materialize` 分层。
- 当前也没有 BasicVSR++ 前的 `numpy -> torch` 输入契约断层。
- 如果要把当前 CPU 版本当作 Vulkan 对照基线，应以 `mosaic_detector.py + frame_restorer.py + video_utils.py + audio_utils.py` 这套现有实现为准。
