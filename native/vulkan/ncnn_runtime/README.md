# ncnn Vulkan Runtime

这个目录存放 Lada 的本地 `ncnn` Vulkan runtime 工程。

当前只保留一个本地 `ncnn` Python runtime 产物，检测和修复共用同一套 Vulkan custom ops 与 native runner，不再维护单独的 Python bridge 模块。

## 目录职责

- `src/torchvision_deform_conv2d_layer.*`
  BasicVSR++ 依赖的自定义 deformconv Vulkan layer。
- `src/torch_conv2d_layer.*`
  检测链使用的自定义 `torch.conv2d` Vulkan layer。
- `src/lada_gridsample_layer.*`
  flow warp / grid sample 相关的自定义 Vulkan layer。
- `src/lada_yolo_attention_layer.*`
  YOLO attention 子图替换后的自定义 Vulkan layer。
- `src/lada_yolo_seg_postprocess_layer.*`
  YOLO segmentation fused postprocess Vulkan layer。
- `src/python_local_runtime_bindings.cpp`
  本地 `ncnn` Python runtime 的公共绑定入口。
- `src/basicvsrpp_clip_runner.*`
  BasicVSR++ clip 级 native runner，把 `quarter_downsample -> feat_extract -> spynet -> recurrent branches -> output_frame` 封装成单个 Python 可调用入口。
- `src/vulkan_blend_runtime.*`
  patch blending 的 Vulkan helper。
- `CMakeLists.txt`
  本地 runtime 的唯一构建定义。

## 构建方式

Windows 下推荐直接执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_ncnn_vulkan_runtime.ps1
```

脚本会优先使用本地的 `.helloagents/tmp/ncnn-src`；如果这个目录不存在，就回退到 CMake 的 `FetchContent` 自动拉取 `ncnn` 源码。

## 构建产物

本地 Python runtime 输出到：

- `native/vulkan/ncnn_runtime/build/local_runtime/`

其中最重要的是：

- `ncnn.cp*.pyd`

这个模块除了标准 `ncnn` Python API，还会额外包含：

- `register_torchvision_deform_conv2d_layers`
- `register_lada_gridsample_layers`
- `register_lada_custom_layers`
- `register_lada_yolo_attention_layers`
- `register_lada_yolo_seg_postprocess_layers`
- `LadaVulkanNetRunner`
- `LadaVulkanTensor`
- `BasicVsrppClipRunner`

## 维护约束

- `build/` 目录属于生成产物，不纳入源码结构设计。
- `BasicVsrppClipRunner` 只负责 native 封装，不负责模型导出；导出仍由 Python 侧 artifact/export 流程管理。
- 如果改动了 YOLO fused subnet 或 BasicVSR++ 模块输入输出语义，记得同步检查 Python 侧 artifact 版本号、runtime 绑定和上层装配代码。
