// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "basicvsrpp_clip_runner.h"
#include "command.h"
#include "gpu.h"
#include "net.h"
#include "lada_gridsample_layer.h"
#include "lada_yolo_seg_postprocess_layer.h"
#include "lada_yolo_attention_layer.h"
#include "torch_conv2d_layer.h"
#include "torchvision_deform_conv2d_layer.h"
#include "vulkan_blend_runtime.h"
#include "yolo_seg_postprocess_runner.h"

namespace py = pybind11;

#if NCNN_VULKAN
namespace {

struct LadaVulkanContext
{
    explicit LadaVulkanContext(py::object net_owner, ncnn::Net& net)
        : net_owner(std::move(net_owner)), net(&net)
    {
        vkdev = net.vulkan_device();
        if (vkdev == nullptr || !vkdev->is_valid()) {
            throw std::runtime_error("ncnn.Net does not expose a valid Vulkan device.");
        }

        blob_vkallocator = vkdev->acquire_blob_allocator();
        staging_vkallocator = vkdev->acquire_staging_allocator();
        if (blob_vkallocator == nullptr || staging_vkallocator == nullptr) {
            throw std::runtime_error("Failed to acquire ncnn Vulkan allocators.");
        }
    }

    ~LadaVulkanContext()
    {
        delete yolo_seg_mask_direct_pipeline;
        delete yolo_seg_mask_proto_decode_pipeline;
        delete yolo_seg_mask_input_resize_pipeline;
        delete yolo_seg_mask_orig_resize_pipeline;
        delete yolo_preprocess_pipeline;
        yolo_seg_mask_direct_pipeline = nullptr;
        yolo_seg_mask_proto_decode_pipeline = nullptr;
        yolo_seg_mask_input_resize_pipeline = nullptr;
        yolo_seg_mask_orig_resize_pipeline = nullptr;
        yolo_preprocess_pipeline = nullptr;
        if (vkdev != nullptr && blob_vkallocator != nullptr) {
            vkdev->reclaim_blob_allocator(blob_vkallocator);
        }
        if (vkdev != nullptr && staging_vkallocator != nullptr) {
            vkdev->reclaim_staging_allocator(staging_vkallocator);
        }
    }

    py::object net_owner;
    ncnn::Net* net = nullptr;
    const ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::VkAllocator* blob_vkallocator = nullptr;
    ncnn::VkAllocator* staging_vkallocator = nullptr;
    ncnn::Pipeline* yolo_seg_mask_direct_pipeline = nullptr;
    ncnn::Pipeline* yolo_seg_mask_proto_decode_pipeline = nullptr;
    ncnn::Pipeline* yolo_seg_mask_input_resize_pipeline = nullptr;
    ncnn::Pipeline* yolo_seg_mask_orig_resize_pipeline = nullptr;
    ncnn::Pipeline* yolo_preprocess_pipeline = nullptr;
};

ncnn::Mat download_vulkan_tensor_to_cpu_mat(const struct LadaVulkanTensor& tensor);

struct LadaVulkanTensor
{
    LadaVulkanTensor() = default;

    LadaVulkanTensor(
        ncnn::VkMat blob,
        std::shared_ptr<LadaVulkanContext> context)
        : blob(std::move(blob)), context(std::move(context))
    {
    }

    ncnn::Mat download() const
    {
        return download_vulkan_tensor_to_cpu_mat(*this);
    }

    bool empty() const
    {
        return blob.empty();
    }

    int dims() const
    {
        return blob.dims;
    }

    int w() const
    {
        return blob.w;
    }

    int h() const
    {
        return blob.h;
    }

    int c() const
    {
        return blob.c;
    }

    int elempack() const
    {
        return blob.elempack;
    }

    size_t elemsize() const
    {
        return blob.elemsize;
    }

    ncnn::VkMat blob;
    std::shared_ptr<LadaVulkanContext> context;
};

lada::YoloSegPostprocessResult postprocess_yolo_segmentation_mat(
    const ncnn::Mat& pred,
    const ncnn::Mat& proto,
    const lada::YoloSegPostprocessConfig& config);

lada::YoloSegPostprocessResult postprocess_yolo_segmentation_from_selected_mat(
    const ncnn::Mat& boxes,
    const ncnn::Mat& selected,
    const ncnn::Mat& proto,
    const lada::YoloSegPostprocessConfig& config);

py::dict pack_yolo_seg_postprocess_result(const lada::YoloSegPostprocessResult& result);

py::object ncnn_mat_to_numpy(const ncnn::Mat& mat);

struct YoloSegMaskFinalizeRecording
{
    int orig_height = 0;
    int orig_width = 0;
    int count = 0;
    ncnn::VkMat selected_unpacked;
    ncnn::VkMat proto_unpacked;
    ncnn::VkMat proto_logits_blob;
    ncnn::VkMat input_masks_blob;
    ncnn::VkMat masks_blob;
    ncnn::Mat boxes_mat;
    ncnn::Mat masks_mat;
};

YoloSegMaskFinalizeRecording record_yolo_segmentation_masks_gpu_finalize(
    const std::shared_ptr<LadaVulkanContext>& context,
    const ncnn::VkMat& boxes_blob,
    const ncnn::VkMat& selected_blob,
    const ncnn::VkMat& proto_blob,
    int count,
    const std::vector<int>& input_shape,
    const std::vector<int>& orig_shape,
    ncnn::VkCompute& cmd);

py::dict pack_yolo_segmentation_masks_gpu_finalize(
    const YoloSegMaskFinalizeRecording& recording);

py::dict finalize_yolo_segmentation_masks_gpu_impl(
    const std::shared_ptr<LadaVulkanContext>& context,
    const ncnn::VkMat& boxes_blob,
    const ncnn::VkMat& selected_blob,
    const ncnn::VkMat& proto_blob,
    int count,
    const std::vector<int>& input_shape,
    const std::vector<int>& orig_shape);

static const char kLadaYoloSegMaskDirectFinalizeShader[] = R"(
#version 450

layout(binding = 0) readonly buffer selected_blob { sfp selected_blob_data[]; };
layout(binding = 1) readonly buffer proto_blob { sfp proto_blob_data[]; };
layout(binding = 2) writeonly buffer masks_blob { sfp masks_blob_data[]; };

layout(push_constant) uniform parameter
{
    int selected_w;
    int selected_h;
    int proto_w;
    int proto_h;
    int proto_c;
    int proto_is_chw;
    int mask_dim;
    int count;
    int input_height;
    int input_width;
    int orig_height;
    int orig_width;
} p;

float selected_value_at(int det_index, int feature_index)
{
    return float(buffer_ld1(selected_blob_data, det_index * p.selected_w + feature_index));
}

int logical_proto_height()
{
    return p.proto_is_chw != 0 ? p.proto_h : p.proto_c;
}

int logical_proto_width()
{
    return p.proto_is_chw != 0 ? p.proto_w : p.proto_h;
}

float proto_value_at(int channel, int y, int x)
{
    if (p.proto_is_chw != 0)
    {
        return float(buffer_ld1(proto_blob_data, (channel * p.proto_h + y) * p.proto_w + x));
    }

    return float(buffer_ld1(proto_blob_data, (y * p.proto_h + x) * p.mask_dim + channel));
}

float proto_mask_logit_at(
    int det_index,
    int y,
    int x,
    int crop_x1,
    int crop_y1,
    int crop_x2,
    int crop_y2)
{
    if (y < crop_y1 || y >= crop_y2 || x < crop_x1 || x >= crop_x2)
        return 0.0;

    float value = 0.0;
    for (int channel = 0; channel < p.mask_dim; channel++)
    {
        value += selected_value_at(det_index, 6 + channel) * proto_value_at(channel, y, x);
    }
    return value;
}

float binary_input_mask_value(
    int det_index,
    int input_y,
    int input_x,
    int crop_x1,
    int crop_y1,
    int crop_x2,
    int crop_y2)
{
    if (input_y < 0 || input_y >= p.input_height || input_x < 0 || input_x >= p.input_width)
        return 0.0;

    const int proto_height = logical_proto_height();
    const int proto_width = logical_proto_width();

    const float scale_y = float(proto_height) / float(p.input_height);
    const float scale_x = float(proto_width) / float(p.input_width);
    const float source_y = max((float(input_y) + 0.5) * scale_y - 0.5, 0.0);
    const float source_x = max((float(input_x) + 0.5) * scale_x - 0.5, 0.0);

    const int y0 = min(int(floor(source_y)), proto_height - 1);
    const int x0 = min(int(floor(source_x)), proto_width - 1);
    const int y1 = min(y0 + 1, proto_height - 1);
    const int x1 = min(x0 + 1, proto_width - 1);
    const float ly = source_y - float(y0);
    const float lx = source_x - float(x0);
    const float hy = 1.0 - ly;
    const float hx = 1.0 - lx;

    const float v00 = proto_mask_logit_at(det_index, y0, x0, crop_x1, crop_y1, crop_x2, crop_y2);
    const float v01 = proto_mask_logit_at(det_index, y0, x1, crop_x1, crop_y1, crop_x2, crop_y2);
    const float v10 = proto_mask_logit_at(det_index, y1, x0, crop_x1, crop_y1, crop_x2, crop_y2);
    const float v11 = proto_mask_logit_at(det_index, y1, x1, crop_x1, crop_y1, crop_x2, crop_y2);
    const float top = v00 * hx + v01 * lx;
    const float bottom = v10 * hx + v11 * lx;
    const float logit = top * hy + bottom * ly;
    return logit > 0.0 ? 255.0 : 0.0;
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int det_index = int(gl_GlobalInvocationID.z);
    if (x >= p.orig_width || y >= p.orig_height || det_index >= p.count)
        return;

    const int proto_height = logical_proto_height();
    const int proto_width = logical_proto_width();

    const float x1 = selected_value_at(det_index, 0);
    const float y1 = selected_value_at(det_index, 1);
    const float x2 = selected_value_at(det_index, 2);
    const float y2 = selected_value_at(det_index, 3);

    const int crop_x1 = max(0, int(roundEven(x1 * float(proto_width) / float(p.input_width))));
    const int crop_y1 = max(0, int(roundEven(y1 * float(proto_height) / float(p.input_height))));
    const int crop_x2 = min(proto_width, int(roundEven(x2 * float(proto_width) / float(p.input_width))));
    const int crop_y2 = min(proto_height, int(roundEven(y2 * float(proto_height) / float(p.input_height))));

    const float gain = min(
        float(p.input_height) / float(p.orig_height),
        float(p.input_width) / float(p.orig_width));
    const float pad_x = (float(p.input_width) - float(p.orig_width) * gain) * 0.5;
    const float pad_y = (float(p.input_height) - float(p.orig_height) * gain) * 0.5;

    const int top = max(0, int(roundEven(pad_y - 0.1)));
    const int left = max(0, int(roundEven(pad_x - 0.1)));
    const int bottom = min(p.input_height, p.input_height - int(roundEven(pad_y + 0.1)));
    const int right = min(p.input_width, p.input_width - int(roundEven(pad_x + 0.1)));
    const int crop_height = max(bottom - top, 1);
    const int crop_width = max(right - left, 1);

    const float source_y = max((float(y) + 0.5) * float(crop_height) / float(p.orig_height) - 0.5, 0.0);
    const float source_x = max((float(x) + 0.5) * float(crop_width) / float(p.orig_width) - 0.5, 0.0);
    const int input_y0 = min(int(floor(source_y)), crop_height - 1);
    const int input_x0 = min(int(floor(source_x)), crop_width - 1);
    const int input_y1 = min(input_y0 + 1, crop_height - 1);
    const int input_x1 = min(input_x0 + 1, crop_width - 1);
    const float ly = source_y - float(input_y0);
    const float lx = source_x - float(input_x0);
    const float hy = 1.0 - ly;
    const float hx = 1.0 - lx;

    const float v00 = binary_input_mask_value(
        det_index,
        top + input_y0,
        left + input_x0,
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2);
    const float v01 = binary_input_mask_value(
        det_index,
        top + input_y0,
        left + input_x1,
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2);
    const float v10 = binary_input_mask_value(
        det_index,
        top + input_y1,
        left + input_x0,
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2);
    const float v11 = binary_input_mask_value(
        det_index,
        top + input_y1,
        left + input_x1,
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2);
    const float top_value = v00 * hx + v01 * lx;
    const float bottom_value = v10 * hx + v11 * lx;
    const float final_value = top_value * hy + bottom_value * ly;

    buffer_st1(
        masks_blob_data,
        (det_index * p.orig_height + y) * p.orig_width + x,
        afp(final_value > 127.0 ? 255.0 : 0.0));
}
)";

static const char kLadaYoloSegMaskProtoDecodeShader[] = R"(
#version 450

layout(binding = 0) readonly buffer selected_blob { sfp selected_blob_data[]; };
layout(binding = 1) readonly buffer proto_blob { sfp proto_blob_data[]; };
layout(binding = 2) writeonly buffer proto_logits_blob { sfp proto_logits_blob_data[]; };

layout(push_constant) uniform parameter
{
    int selected_w;
    int proto_w;
    int proto_h;
    int proto_c;
    int proto_is_chw;
    int mask_dim;
    int count;
    int input_height;
    int input_width;
} p;

float selected_value_at(int det_index, int feature_index)
{
    return float(buffer_ld1(selected_blob_data, det_index * p.selected_w + feature_index));
}

int logical_proto_height()
{
    return p.proto_is_chw != 0 ? p.proto_h : p.proto_c;
}

int logical_proto_width()
{
    return p.proto_is_chw != 0 ? p.proto_w : p.proto_h;
}

float proto_value_at(int channel, int y, int x)
{
    if (p.proto_is_chw != 0)
    {
        return float(buffer_ld1(proto_blob_data, (channel * p.proto_h + y) * p.proto_w + x));
    }

    return float(buffer_ld1(proto_blob_data, (y * p.proto_h + x) * p.mask_dim + channel));
}

void proto_logit_store(int det_index, int y, int x, float value)
{
    const int proto_width = logical_proto_width();
    const int proto_height = logical_proto_height();
    buffer_st1(
        proto_logits_blob_data,
        (det_index * proto_height + y) * proto_width + x,
        afp(value));
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int det_index = int(gl_GlobalInvocationID.z);
    const int proto_height = logical_proto_height();
    const int proto_width = logical_proto_width();
    if (x >= proto_width || y >= proto_height || det_index >= p.count)
        return;

    const float x1 = selected_value_at(det_index, 0);
    const float y1 = selected_value_at(det_index, 1);
    const float x2 = selected_value_at(det_index, 2);
    const float y2 = selected_value_at(det_index, 3);
    const int crop_x1 = max(0, int(roundEven(x1 * float(proto_width) / float(p.input_width))));
    const int crop_y1 = max(0, int(roundEven(y1 * float(proto_height) / float(p.input_height))));
    const int crop_x2 = min(proto_width, int(roundEven(x2 * float(proto_width) / float(p.input_width))));
    const int crop_y2 = min(proto_height, int(roundEven(y2 * float(proto_height) / float(p.input_height))));
    if (y < crop_y1 || y >= crop_y2 || x < crop_x1 || x >= crop_x2)
    {
        proto_logit_store(det_index, y, x, 0.0);
        return;
    }

    float value = 0.0;
    for (int channel = 0; channel < p.mask_dim; channel++)
    {
        value += selected_value_at(det_index, 6 + channel) * proto_value_at(channel, y, x);
    }
    proto_logit_store(det_index, y, x, value);
}
)";

static const char kLadaYoloSegMaskInputResizeShader[] = R"(
#version 450

layout(binding = 0) readonly buffer proto_logits_blob { sfp proto_logits_blob_data[]; };
layout(binding = 1) writeonly buffer input_masks_blob { sfp input_masks_blob_data[]; };

layout(push_constant) uniform parameter
{
    int proto_w;
    int proto_h;
    int count;
    int input_height;
    int input_width;
} p;

float proto_logit_value_at(int det_index, int y, int x)
{
    return float(buffer_ld1(proto_logits_blob_data, (det_index * p.proto_h + y) * p.proto_w + x));
}

void input_mask_store(int det_index, int y, int x, float value)
{
    buffer_st1(
        input_masks_blob_data,
        (det_index * p.input_height + y) * p.input_width + x,
        afp(value));
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int det_index = int(gl_GlobalInvocationID.z);
    if (x >= p.input_width || y >= p.input_height || det_index >= p.count)
        return;

    const float scale_y = float(p.proto_h) / float(p.input_height);
    const float scale_x = float(p.proto_w) / float(p.input_width);
    const float source_y = max((float(y) + 0.5) * scale_y - 0.5, 0.0);
    const float source_x = max((float(x) + 0.5) * scale_x - 0.5, 0.0);

    const int y0 = min(int(floor(source_y)), p.proto_h - 1);
    const int x0 = min(int(floor(source_x)), p.proto_w - 1);
    const int y1 = min(y0 + 1, p.proto_h - 1);
    const int x1 = min(x0 + 1, p.proto_w - 1);
    const float ly = source_y - float(y0);
    const float lx = source_x - float(x0);
    const float hy = 1.0 - ly;
    const float hx = 1.0 - lx;

    const float v00 = proto_logit_value_at(det_index, y0, x0);
    const float v01 = proto_logit_value_at(det_index, y0, x1);
    const float v10 = proto_logit_value_at(det_index, y1, x0);
    const float v11 = proto_logit_value_at(det_index, y1, x1);
    const float top = v00 * hx + v01 * lx;
    const float bottom = v10 * hx + v11 * lx;
    const float logit = top * hy + bottom * ly;
    input_mask_store(det_index, y, x, logit > 0.0 ? 255.0 : 0.0);
}
)";

static const char kLadaYoloSegMaskOrigResizeShader[] = R"(
#version 450

layout(binding = 0) readonly buffer input_masks_blob { sfp input_masks_blob_data[]; };
layout(binding = 1) writeonly buffer masks_blob { sfp masks_blob_data[]; };

layout(push_constant) uniform parameter
{
    int input_width;
    int input_height;
    int count;
    int orig_height;
    int orig_width;
} p;

float input_mask_value_at(int det_index, int y, int x)
{
    return float(buffer_ld1(input_masks_blob_data, (det_index * p.input_height + y) * p.input_width + x));
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int det_index = int(gl_GlobalInvocationID.z);
    if (x >= p.orig_width || y >= p.orig_height || det_index >= p.count)
        return;

    const float gain = min(
        float(p.input_height) / float(p.orig_height),
        float(p.input_width) / float(p.orig_width));
    const float pad_x = (float(p.input_width) - float(p.orig_width) * gain) * 0.5;
    const float pad_y = (float(p.input_height) - float(p.orig_height) * gain) * 0.5;

    const int top = max(0, int(roundEven(pad_y - 0.1)));
    const int left = max(0, int(roundEven(pad_x - 0.1)));
    const int bottom = min(p.input_height, p.input_height - int(roundEven(pad_y + 0.1)));
    const int right = min(p.input_width, p.input_width - int(roundEven(pad_x + 0.1)));
    const int crop_height = max(bottom - top, 1);
    const int crop_width = max(right - left, 1);

    const float source_y = max((float(y) + 0.5) * float(crop_height) / float(p.orig_height) - 0.5, 0.0);
    const float source_x = max((float(x) + 0.5) * float(crop_width) / float(p.orig_width) - 0.5, 0.0);
    const int input_y0 = min(int(floor(source_y)), crop_height - 1);
    const int input_x0 = min(int(floor(source_x)), crop_width - 1);
    const int input_y1 = min(input_y0 + 1, crop_height - 1);
    const int input_x1 = min(input_x0 + 1, crop_width - 1);
    const float ly = source_y - float(input_y0);
    const float lx = source_x - float(input_x0);
    const float hy = 1.0 - ly;
    const float hx = 1.0 - lx;

    const float v00 = input_mask_value_at(det_index, top + input_y0, left + input_x0);
    const float v01 = input_mask_value_at(det_index, top + input_y0, left + input_x1);
    const float v10 = input_mask_value_at(det_index, top + input_y1, left + input_x0);
    const float v11 = input_mask_value_at(det_index, top + input_y1, left + input_x1);
    const float top_value = v00 * hx + v01 * lx;
    const float bottom_value = v10 * hx + v11 * lx;
    const float final_value = top_value * hy + bottom_value * ly;

    buffer_st1(
        masks_blob_data,
        (det_index * p.orig_height + y) * p.orig_width + x,
        afp(final_value > 127.0 ? 255.0 : 0.0));
}
)";

static const char kLadaYoloPreprocessShader[] = R"(
#version 450

layout(binding = 0) readonly buffer input_blob { uint input_blob_data[]; };
layout(binding = 1) writeonly buffer output_blob { sfp output_blob_data[]; };

layout(push_constant) uniform parameter
{
    int src_w;
    int src_h;
    int dst_w;
    int dst_h;
    int resized_w;
    int resized_h;
    int pad_left;
    int pad_top;
    int output_cstep;
} p;

uint load_input_byte(int byte_index)
{
    const int word_index = byte_index >> 2;
    const int byte_offset = (byte_index & 3) * 8;
    return (input_blob_data[word_index] >> uint(byte_offset)) & 255u;
}

float read_input_value(int y, int x, int channel)
{
    const int byte_index = ((y * p.src_w + x) * 3) + channel;
    return float(load_input_byte(byte_index));
}

float sample_resized_value(int y, int x, int channel)
{
    const int inner_x = x - p.pad_left;
    const int inner_y = y - p.pad_top;
    if (inner_x < 0 || inner_x >= p.resized_w || inner_y < 0 || inner_y >= p.resized_h)
        return 114.0 / 255.0;

    const float src_x = (float(inner_x) + 0.5) * float(p.src_w) / float(p.resized_w) - 0.5;
    const float src_y = (float(inner_y) + 0.5) * float(p.src_h) / float(p.resized_h) - 0.5;
    const int x0 = clamp(int(floor(src_x)), 0, p.src_w - 1);
    const int y0 = clamp(int(floor(src_y)), 0, p.src_h - 1);
    const int x1 = min(x0 + 1, p.src_w - 1);
    const int y1 = min(y0 + 1, p.src_h - 1);
    const float lx = src_x - float(x0);
    const float ly = src_y - float(y0);
    const float hx = 1.0 - lx;
    const float hy = 1.0 - ly;

    const float v00 = read_input_value(y0, x0, channel);
    const float v01 = read_input_value(y0, x1, channel);
    const float v10 = read_input_value(y1, x0, channel);
    const float v11 = read_input_value(y1, x1, channel);
    const float top = v00 * hx + v01 * lx;
    const float bottom = v10 * hx + v11 * lx;
    return (top * hy + bottom * ly) / 255.0;
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int c = int(gl_GlobalInvocationID.z);
    if (x >= p.dst_w || y >= p.dst_h || c >= 3)
        return;

    const float value = sample_resized_value(y, x, c);
    buffer_st1(output_blob_data, c * p.output_cstep + y * p.dst_w + x, afp(value));
}
)";

ncnn::Mat download_vulkan_tensor_to_cpu_mat(const LadaVulkanTensor& tensor)
{
    if (!tensor.context) {
        throw std::runtime_error("LadaVulkanTensor is detached from its Vulkan context.");
    }

    ncnn::Option opt = tensor.context->net->opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.blob_vkallocator = tensor.context->blob_vkallocator;
    opt.workspace_vkallocator = tensor.context->blob_vkallocator;
    opt.staging_vkallocator = tensor.context->staging_vkallocator;

    ncnn::VkCompute cmd(tensor.context->vkdev);
    ncnn::Mat output;
    cmd.record_download(tensor.blob, output, opt);
    const int ret = cmd.submit_and_wait();
    if (ret != 0) {
        throw std::runtime_error("Failed to download Vulkan tensor from ncnn.");
    }
    return output.clone();
}

void ensure_yolo_seg_mask_finalize_pipelines(LadaVulkanContext& context)
{
    if (
        context.yolo_seg_mask_direct_pipeline != nullptr
        &&
        context.yolo_seg_mask_proto_decode_pipeline != nullptr
        && context.yolo_seg_mask_input_resize_pipeline != nullptr
        && context.yolo_seg_mask_orig_resize_pipeline != nullptr) {
        return;
    }

    ncnn::Option opt = context.net->opt;
    opt.use_vulkan_compute = true;
    opt.blob_vkallocator = context.blob_vkallocator;
    opt.workspace_vkallocator = context.blob_vkallocator;
    opt.staging_vkallocator = context.staging_vkallocator;

    auto create_pipeline =
        [&](const char* shader_source, int shader_size, const char* debug_name) -> ncnn::Pipeline* {
        std::vector<uint32_t> spirv;
        const int compile_ret = ncnn::compile_spirv_module(
            shader_source,
            shader_size,
            opt,
            spirv);
        if (compile_ret != 0) {
            throw std::runtime_error(std::string("Failed to compile ") + debug_name + " shader.");
        }

        ncnn::Pipeline* pipeline = new ncnn::Pipeline(context.vkdev);
        pipeline->set_optimal_local_size_xyz(8, 8, 1);
        if (
            pipeline->create(
                spirv.data(),
                spirv.size() * sizeof(uint32_t),
                std::vector<ncnn::vk_specialization_type>()) != 0) {
            delete pipeline;
            throw std::runtime_error(std::string("Failed to create ") + debug_name + " pipeline.");
        }
        return pipeline;
    };

    context.yolo_seg_mask_direct_pipeline = create_pipeline(
        kLadaYoloSegMaskDirectFinalizeShader,
        static_cast<int>(sizeof(kLadaYoloSegMaskDirectFinalizeShader) - 1),
        "YOLO direct finalize");
    context.yolo_seg_mask_proto_decode_pipeline = create_pipeline(
        kLadaYoloSegMaskProtoDecodeShader,
        static_cast<int>(sizeof(kLadaYoloSegMaskProtoDecodeShader) - 1),
        "YOLO proto decode");
    context.yolo_seg_mask_input_resize_pipeline = create_pipeline(
        kLadaYoloSegMaskInputResizeShader,
        static_cast<int>(sizeof(kLadaYoloSegMaskInputResizeShader) - 1),
        "YOLO input resize");
    context.yolo_seg_mask_orig_resize_pipeline = create_pipeline(
        kLadaYoloSegMaskOrigResizeShader,
        static_cast<int>(sizeof(kLadaYoloSegMaskOrigResizeShader) - 1),
        "YOLO original resize");
}

void ensure_yolo_preprocess_pipeline(LadaVulkanContext& context)
{
    if (context.yolo_preprocess_pipeline != nullptr) {
        return;
    }

    ncnn::Option opt = context.net->opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.blob_vkallocator = context.blob_vkallocator;
    opt.workspace_vkallocator = context.blob_vkallocator;
    opt.staging_vkallocator = context.staging_vkallocator;

    std::vector<uint32_t> spirv;
    const int compile_ret = ncnn::compile_spirv_module(
        kLadaYoloPreprocessShader,
        static_cast<int>(sizeof(kLadaYoloPreprocessShader) - 1),
        opt,
        spirv);
    if (compile_ret != 0) {
        throw std::runtime_error("Failed to compile YOLO preprocess shader.");
    }

    context.yolo_preprocess_pipeline = new ncnn::Pipeline(context.vkdev);
    context.yolo_preprocess_pipeline->set_optimal_local_size_xyz(8, 8, 1);
    if (
        context.yolo_preprocess_pipeline->create(
            spirv.data(),
            spirv.size() * sizeof(uint32_t),
            std::vector<ncnn::vk_specialization_type>()) != 0) {
        delete context.yolo_preprocess_pipeline;
        context.yolo_preprocess_pipeline = nullptr;
        throw std::runtime_error("Failed to create YOLO preprocess pipeline.");
    }
}

py::dict finalize_yolo_segmentation_masks_gpu_py(
    const LadaVulkanTensor& boxes_tensor,
    const LadaVulkanTensor& selected_tensor,
    const LadaVulkanTensor& proto_tensor,
    int count,
    const std::vector<int>& input_shape,
    const std::vector<int>& orig_shape)
{
    if (!boxes_tensor.context || !selected_tensor.context || !proto_tensor.context) {
        throw std::runtime_error("YOLO Vulkan tensors are detached from their contexts.");
    }
    if (boxes_tensor.context->vkdev != selected_tensor.context->vkdev) {
        throw std::runtime_error("YOLO boxes and selected tensors must share the same Vulkan device.");
    }
    if (selected_tensor.context->vkdev != proto_tensor.context->vkdev) {
        throw std::runtime_error("YOLO selected and proto tensors must share the same Vulkan device.");
    }
    return finalize_yolo_segmentation_masks_gpu_impl(
        selected_tensor.context,
        boxes_tensor.blob,
        selected_tensor.blob,
        proto_tensor.blob,
        count,
        input_shape,
        orig_shape);
}

class LadaVulkanNetRunner
{
public:
    explicit LadaVulkanNetRunner(py::object net_obj)
        : context_(std::make_shared<LadaVulkanContext>(net_obj, net_obj.cast<ncnn::Net&>()))
    {
    }

    py::list preprocess_bgr_u8_batch(
        const py::list& images,
        const std::vector<int>& input_shape) const
    {
        if (input_shape.size() < 2) {
            throw std::runtime_error("YOLO preprocess input shape must contain height and width.");
        }

        const int dst_h = static_cast<int>(input_shape[0]);
        const int dst_w = static_cast<int>(input_shape[1]);
        if (dst_h <= 0 || dst_w <= 0) {
            throw std::runtime_error("YOLO preprocess received an invalid output shape.");
        }
        if (py::len(images) == 0) {
            return py::list();
        }

        struct PackedBgrImage
        {
            int src_h = 0;
            int src_w = 0;
            int resized_w = 0;
            int resized_h = 0;
            int pad_left = 0;
            int pad_top = 0;
            ncnn::Mat packed_input;
        };

        std::vector<PackedBgrImage> packed_images;
        packed_images.reserve(py::len(images));
        for (const py::handle& item : images) {
            const auto image =
                py::array_t<unsigned char, py::array::c_style | py::array::forcecast>::ensure(item);
            if (!image) {
                throw std::runtime_error("YOLO preprocess batch expects numpy-compatible uint8 arrays.");
            }

            const py::buffer_info image_info = image.request();
            if (image_info.ndim != 3) {
                throw std::runtime_error("YOLO preprocess expects HWC uint8 images.");
            }
            if (image_info.shape[2] != 3) {
                throw std::runtime_error("YOLO preprocess expects 3-channel BGR input.");
            }

            PackedBgrImage packed;
            packed.src_h = static_cast<int>(image_info.shape[0]);
            packed.src_w = static_cast<int>(image_info.shape[1]);
            if (packed.src_h <= 0 || packed.src_w <= 0) {
                throw std::runtime_error("YOLO preprocess received an invalid input shape.");
            }

            const float ratio = std::min(
                static_cast<float>(dst_h) / static_cast<float>(packed.src_h),
                static_cast<float>(dst_w) / static_cast<float>(packed.src_w));
            packed.resized_w = std::max(
                1,
                static_cast<int>(std::round(static_cast<float>(packed.src_w) * ratio)));
            packed.resized_h = std::max(
                1,
                static_cast<int>(std::round(static_cast<float>(packed.src_h) * ratio)));
            const int pad_w = dst_w - packed.resized_w;
            const int pad_h = dst_h - packed.resized_h;
            packed.pad_left = static_cast<int>(std::round(static_cast<float>(pad_w) / 2.f - 0.1f));
            packed.pad_top = static_cast<int>(std::round(static_cast<float>(pad_h) / 2.f - 0.1f));

            const std::size_t byte_count =
                static_cast<std::size_t>(packed.src_h) * static_cast<std::size_t>(packed.src_w) * 3u;
            const std::size_t word_count = (byte_count + 3u) / 4u;
            packed.packed_input = ncnn::Mat(static_cast<int>(word_count), static_cast<size_t>(4u), 1);
            if (packed.packed_input.empty()) {
                throw std::runtime_error("Failed to allocate YOLO preprocess input buffer.");
            }
            std::memset(packed.packed_input.data, 0, word_count * sizeof(std::uint32_t));
            std::memcpy(packed.packed_input.data, image_info.ptr, byte_count);
            packed_images.push_back(std::move(packed));
        }

        ensure_yolo_preprocess_pipeline(*context_);

        ncnn::Option opt = context_->net->opt;
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = false;
        opt.blob_vkallocator = context_->blob_vkallocator;
        opt.workspace_vkallocator = context_->blob_vkallocator;
        opt.staging_vkallocator = context_->staging_vkallocator;

        ncnn::VkCompute cmd(context_->vkdev);
        std::vector<ncnn::VkMat> outputs;
        outputs.reserve(packed_images.size());

        for (const PackedBgrImage& packed : packed_images) {
            ncnn::VkMat input_blob;
            input_blob.create(
                packed.packed_input.w,
                static_cast<size_t>(4u),
                1,
                opt.blob_vkallocator);
            if (input_blob.empty()) {
                throw std::runtime_error("Failed to allocate YOLO preprocess Vulkan input tensor.");
            }
            cmd.record_upload(packed.packed_input, input_blob, opt);

            ncnn::VkMat output_blob;
            output_blob.create(dst_w, dst_h, 3, static_cast<size_t>(4u), 1, opt.blob_vkallocator);
            if (output_blob.empty()) {
                throw std::runtime_error("Failed to allocate YOLO preprocess Vulkan output tensor.");
            }

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = input_blob;
            bindings[1] = output_blob;
            std::vector<ncnn::vk_constant_type> constants(9);
            constants[0].i = packed.src_w;
            constants[1].i = packed.src_h;
            constants[2].i = dst_w;
            constants[3].i = dst_h;
            constants[4].i = packed.resized_w;
            constants[5].i = packed.resized_h;
            constants[6].i = packed.pad_left;
            constants[7].i = packed.pad_top;
            constants[8].i = static_cast<int>(output_blob.cstep);
            cmd.record_pipeline(context_->yolo_preprocess_pipeline, bindings, constants, output_blob);
            outputs.push_back(output_blob);
        }

        if (cmd.submit_and_wait() != 0) {
            throw std::runtime_error("Failed to execute YOLO preprocess Vulkan batch pipeline.");
        }

        py::list result;
        for (ncnn::VkMat& output_blob : outputs) {
            result.append(py::cast(LadaVulkanTensor(std::move(output_blob), context_)));
        }
        return result;
    }

    LadaVulkanTensor preprocess_bgr_u8(
        const py::array_t<unsigned char, py::array::c_style | py::array::forcecast>& image,
        const std::vector<int>& input_shape) const
    {
        if (input_shape.size() < 2) {
            throw std::runtime_error("YOLO preprocess input shape must contain height and width.");
        }

        const py::buffer_info image_info = image.request();
        if (image_info.ndim != 3) {
            throw std::runtime_error("YOLO preprocess expects an HWC uint8 image.");
        }
        if (image_info.shape[2] != 3) {
            throw std::runtime_error("YOLO preprocess expects 3-channel BGR input.");
        }

        const int src_h = static_cast<int>(image_info.shape[0]);
        const int src_w = static_cast<int>(image_info.shape[1]);
        const int dst_h = static_cast<int>(input_shape[0]);
        const int dst_w = static_cast<int>(input_shape[1]);
        if (src_h <= 0 || src_w <= 0 || dst_h <= 0 || dst_w <= 0) {
            throw std::runtime_error("YOLO preprocess received an invalid input or output shape.");
        }

        const float ratio = std::min(
            static_cast<float>(dst_h) / static_cast<float>(src_h),
            static_cast<float>(dst_w) / static_cast<float>(src_w));
        const int resized_w = std::max(1, static_cast<int>(std::round(static_cast<float>(src_w) * ratio)));
        const int resized_h = std::max(1, static_cast<int>(std::round(static_cast<float>(src_h) * ratio)));
        const int pad_w = dst_w - resized_w;
        const int pad_h = dst_h - resized_h;
        const int pad_left = static_cast<int>(std::round(static_cast<float>(pad_w) / 2.f - 0.1f));
        const int pad_top = static_cast<int>(std::round(static_cast<float>(pad_h) / 2.f - 0.1f));

        ensure_yolo_preprocess_pipeline(*context_);

        const std::size_t byte_count =
            static_cast<std::size_t>(src_h) * static_cast<std::size_t>(src_w) * 3u;
        const std::size_t word_count = (byte_count + 3u) / 4u;
        ncnn::Mat packed_input(static_cast<int>(word_count), static_cast<size_t>(4u), 1);
        if (packed_input.empty()) {
            throw std::runtime_error("Failed to allocate YOLO preprocess input buffer.");
        }
        std::memset(packed_input.data, 0, word_count * sizeof(std::uint32_t));
        std::memcpy(packed_input.data, image_info.ptr, byte_count);

        ncnn::Option opt = context_->net->opt;
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = false;
        opt.blob_vkallocator = context_->blob_vkallocator;
        opt.workspace_vkallocator = context_->blob_vkallocator;
        opt.staging_vkallocator = context_->staging_vkallocator;

        ncnn::VkCompute cmd(context_->vkdev);
        ncnn::VkMat input_blob;
        input_blob.create(static_cast<int>(word_count), static_cast<size_t>(4u), 1, opt.blob_vkallocator);
        if (input_blob.empty()) {
            throw std::runtime_error("Failed to allocate YOLO preprocess Vulkan input tensor.");
        }
        cmd.record_upload(packed_input, input_blob, opt);

        ncnn::VkMat output_blob;
        output_blob.create(dst_w, dst_h, 3, static_cast<size_t>(4u), 1, opt.blob_vkallocator);
        if (output_blob.empty()) {
            throw std::runtime_error("Failed to allocate YOLO preprocess Vulkan output tensor.");
        }

        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = input_blob;
        bindings[1] = output_blob;
        std::vector<ncnn::vk_constant_type> constants(9);
        constants[0].i = src_w;
        constants[1].i = src_h;
        constants[2].i = dst_w;
        constants[3].i = dst_h;
        constants[4].i = resized_w;
        constants[5].i = resized_h;
        constants[6].i = pad_left;
        constants[7].i = pad_top;
        constants[8].i = static_cast<int>(output_blob.cstep);
        cmd.record_pipeline(context_->yolo_preprocess_pipeline, bindings, constants, output_blob);

        if (cmd.submit_and_wait() != 0) {
            throw std::runtime_error("Failed to execute YOLO preprocess Vulkan pipeline.");
        }

        return LadaVulkanTensor(std::move(output_blob), context_);
    }

    LadaVulkanTensor run(const py::dict& inputs, const std::string& output_name = "out0") const
    {
        ncnn::Extractor extractor = create_extractor(false);
        feed_inputs(extractor, inputs);

        ncnn::VkCompute extract_cmd(context_->vkdev);
        ncnn::VkMat output;
        int ret = extractor.extract(output_name.c_str(), output, extract_cmd);
        if (ret != 0) {
            throw std::runtime_error("Failed to record Vulkan extractor output.");
        }

        ret = extract_cmd.submit_and_wait();
        if (ret != 0 || output.empty()) {
            throw std::runtime_error("Failed to produce Vulkan tensor from ncnn extractor output.");
        }

        return LadaVulkanTensor(std::move(output), context_);
    }

    ncnn::Mat run_to_cpu(const py::dict& inputs, const std::string& output_name = "out0") const
    {
        ncnn::Extractor extractor = create_extractor(true);
        feed_inputs(extractor, inputs);

        return extract_cpu_output(extractor, output_name);
    }

    py::dict run_many(
        const py::dict& inputs,
        const std::vector<std::string>& output_names) const
    {
        ncnn::Extractor extractor = create_extractor(false);
        feed_inputs(extractor, inputs);

        ncnn::VkCompute extract_cmd(context_->vkdev);
        py::dict outputs;
        for (const std::string& output_name : output_names) {
            ncnn::VkMat output;
            int ret = extractor.extract(output_name.c_str(), output, extract_cmd);
            if (ret != 0) {
                throw std::runtime_error("Failed to record Vulkan extractor output.");
            }

            outputs[py::str(output_name)] = py::cast(LadaVulkanTensor(std::move(output), context_));
        }

        const int ret = extract_cmd.submit_and_wait();
        if (ret != 0) {
            throw std::runtime_error("Failed to produce Vulkan tensors from ncnn extractor output.");
        }

        return outputs;
    }

    py::dict run_many_to_cpu(
        const py::dict& inputs,
        const std::vector<std::string>& output_names) const
    {
        py::dict outputs;
        for (const std::string& output_name : output_names) {
            outputs[py::str(output_name)] = ncnn_mat_to_numpy(run_to_cpu(inputs, output_name));
        }
        return outputs;
    }

    py::dict run_yolo_segmentation(
        const py::dict& inputs,
        const std::vector<std::string>& output_names,
        const std::vector<int>& input_shape,
        const std::vector<int>& orig_shape,
        float conf_threshold,
        float iou_threshold,
        int max_det,
        int num_classes,
        bool agnostic_nms) const
    {
        if (input_shape.size() < 2 || orig_shape.size() < 2) {
            throw std::runtime_error("Input and original shapes must contain height and width.");
        }

        ncnn::Mat pred;
        ncnn::Mat proto;
        bool has_pred = false;
        bool has_proto = false;
        for (const std::string& output_name : output_names) {
            ncnn::Mat output = run_to_cpu(inputs, output_name);
            if (!has_pred && output.dims == 2) {
                pred = std::move(output);
                has_pred = true;
                continue;
            }
            if (!has_proto && output.dims == 3) {
                proto = std::move(output);
                has_proto = true;
            }
        }

        if (!has_pred || !has_proto) {
            throw std::runtime_error("Failed to resolve YOLO segmentation outputs from NCNN runner.");
        }

        lada::YoloSegPostprocessConfig config;
        config.conf_threshold = conf_threshold;
        config.iou_threshold = iou_threshold;
        config.max_det = max_det;
        config.num_classes = num_classes;
        config.agnostic_nms = agnostic_nms;
        config.input_height = input_shape[0];
        config.input_width = input_shape[1];
        config.orig_height = orig_shape[0];
        config.orig_width = orig_shape[1];

        return pack_yolo_seg_postprocess_result(
            postprocess_yolo_segmentation_mat(pred, proto, config));
    }

    py::dict run_yolo_segmentation_subnet(
        const py::dict& inputs,
        const std::vector<std::string>& output_names,
        const LadaVulkanNetRunner& postprocess_runner,
        const std::vector<int>& input_shape,
        const std::vector<int>& orig_shape,
        float conf_threshold,
        float iou_threshold,
        int max_det,
        int num_classes,
        bool agnostic_nms) const
    {
        if (context_->vkdev != postprocess_runner.context_->vkdev) {
            throw std::runtime_error("YOLO detector and postprocess runners must use the same Vulkan device.");
        }
        if (input_shape.size() < 2 || orig_shape.size() < 2) {
            throw std::runtime_error("Input and original shapes must contain height and width.");
        }

        ncnn::Extractor detector_extractor = create_extractor(true);
        feed_inputs(detector_extractor, inputs);

        ncnn::VkCompute detector_cmd(context_->vkdev);
        ncnn::VkMat pred_blob;
        ncnn::VkMat proto_blob;
        bool has_pred = false;
        bool has_proto = false;
        for (const std::string& output_name : output_names) {
            ncnn::VkMat extracted_output;
            const int extract_ret = detector_extractor.extract(output_name.c_str(), extracted_output, detector_cmd);
            if (extract_ret != 0) {
                throw std::runtime_error("Failed to record YOLO detector Vulkan output.");
            }

            if (!has_pred && extracted_output.dims == 2) {
                pred_blob = extracted_output;
                has_pred = true;
                continue;
            }
            if (!has_proto && extracted_output.dims == 3) {
                proto_blob = extracted_output;
                has_proto = true;
            }
        }

        if (!has_pred || !has_proto) {
            throw std::runtime_error("Failed to resolve YOLO segmentation outputs from Vulkan detector.");
        }
        if (detector_cmd.submit_and_wait() != 0 || pred_blob.empty() || proto_blob.empty()) {
            throw std::runtime_error("Failed to finalize YOLO detector Vulkan outputs.");
        }

        ncnn::Mat config_mat(9);
        if (config_mat.empty()) {
            throw std::runtime_error("Failed to allocate YOLO postprocess config tensor.");
        }
        float* config_ptr = static_cast<float*>(config_mat.data);
        config_ptr[0] = conf_threshold;
        config_ptr[1] = iou_threshold;
        config_ptr[2] = static_cast<float>(max_det);
        config_ptr[3] = static_cast<float>(num_classes);
        config_ptr[4] = agnostic_nms ? 1.f : 0.f;
        config_ptr[5] = static_cast<float>(input_shape[0]);
        config_ptr[6] = static_cast<float>(input_shape[1]);
        config_ptr[7] = static_cast<float>(orig_shape[0]);
        config_ptr[8] = static_cast<float>(orig_shape[1]);

        ncnn::Extractor postprocess_extractor = postprocess_runner.create_extractor(true);
        if (postprocess_extractor.input("pred", pred_blob) != 0) {
            throw std::runtime_error("Failed to feed YOLO prediction tensor into Vulkan postprocess subnet.");
        }
        if (postprocess_extractor.input("proto", proto_blob) != 0) {
            throw std::runtime_error("Failed to feed YOLO proto tensor into Vulkan postprocess subnet.");
        }
        if (postprocess_extractor.input("config", config_mat) != 0) {
            throw std::runtime_error("Failed to feed YOLO config tensor into Vulkan postprocess subnet.");
        }

        ncnn::Option postprocess_opt = postprocess_runner.context_->net->opt;
        postprocess_opt.use_vulkan_compute = true;
        postprocess_opt.use_packing_layout = false;
        postprocess_opt.blob_vkallocator = postprocess_runner.context_->blob_vkallocator;
        postprocess_opt.workspace_vkallocator = postprocess_runner.context_->blob_vkallocator;
        postprocess_opt.staging_vkallocator = postprocess_runner.context_->staging_vkallocator;

        ncnn::VkCompute postprocess_cmd(context_->vkdev);
        ncnn::VkMat boxes_blob;
        ncnn::VkMat selected_blob;
        ncnn::Mat count_mat;

        {
            ncnn::VkMat extracted_boxes;
            if (postprocess_extractor.extract("boxes", extracted_boxes, postprocess_cmd) != 0) {
                throw std::runtime_error("Failed to extract YOLO boxes tensor from Vulkan postprocess subnet.");
            }
            boxes_blob = extracted_boxes;
        }
        {
            ncnn::VkMat extracted_selected;
            if (postprocess_extractor.extract("selected", extracted_selected, postprocess_cmd) != 0) {
                throw std::runtime_error("Failed to extract YOLO selected tensor from Vulkan postprocess subnet.");
            }
            selected_blob = extracted_selected;
        }
        {
            ncnn::VkMat extracted_count;
            if (postprocess_extractor.extract("count", extracted_count, postprocess_cmd) != 0) {
                throw std::runtime_error("Failed to extract YOLO count tensor from Vulkan postprocess subnet.");
            }
            postprocess_cmd.record_download(extracted_count, count_mat, postprocess_opt);
        }

        if (postprocess_cmd.submit_and_wait() != 0 || boxes_blob.empty() || selected_blob.empty()) {
            throw std::runtime_error("Failed to finalize YOLO postprocess Vulkan outputs.");
        }
        if (count_mat.dims != 1 || count_mat.w < 1 || count_mat.elemsize != sizeof(float)) {
            throw std::runtime_error("Downloaded YOLO count tensor has an unexpected shape or dtype.");
        }

        const int output_capacity = std::min(boxes_blob.h, selected_blob.h);
        const int count = std::clamp(
            static_cast<int>(static_cast<const float*>(count_mat.data)[0]),
            0,
            output_capacity);
        return finalize_yolo_segmentation_masks_gpu_impl(
            postprocess_runner.context_,
            boxes_blob,
            selected_blob,
            proto_blob,
            count,
            input_shape,
            orig_shape);
    }

    py::list run_yolo_segmentation_subnet_batch(
        const py::list& input_frames,
        const std::string& input_name,
        const std::vector<std::string>& output_names,
        const LadaVulkanNetRunner& postprocess_runner,
        const std::vector<int>& input_shape,
        const std::vector<std::vector<int>>& orig_shapes,
        float conf_threshold,
        float iou_threshold,
        int max_det,
        int num_classes,
        bool agnostic_nms) const
    {
        if (context_->vkdev != postprocess_runner.context_->vkdev) {
            throw std::runtime_error("YOLO detector and postprocess runners must use the same Vulkan device.");
        }
        if (input_shape.size() < 2) {
            throw std::runtime_error("Input shape must contain height and width.");
        }

        const std::size_t batch_size = static_cast<std::size_t>(py::len(input_frames));
        if (batch_size != orig_shapes.size()) {
            throw std::runtime_error("YOLO segmentation batch requires one original shape per input frame.");
        }
        if (batch_size == 0) {
            return py::list();
        }

        struct BatchState
        {
            ncnn::VkMat pred_blob;
            ncnn::VkMat proto_blob;
            ncnn::VkMat boxes_blob;
            ncnn::VkMat selected_blob;
            ncnn::Mat count_mat;
        };

        std::vector<BatchState> states(batch_size);
        {
            ncnn::VkCompute detector_cmd(context_->vkdev);
            std::vector<ncnn::Extractor> detector_extractors;
            detector_extractors.reserve(batch_size);

            for (std::size_t frame_index = 0; frame_index < batch_size; ++frame_index) {
                detector_extractors.push_back(create_extractor(true));
                ncnn::Extractor& detector_extractor = detector_extractors.back();
                feed_input(
                    detector_extractor,
                    input_name,
                    input_frames[static_cast<py::ssize_t>(frame_index)]);

                bool has_pred = false;
                bool has_proto = false;
                for (const std::string& output_name : output_names) {
                    ncnn::VkMat extracted_output;
                    const int extract_ret = detector_extractor.extract(
                        output_name.c_str(),
                        extracted_output,
                        detector_cmd);
                    if (extract_ret != 0) {
                        throw std::runtime_error("Failed to record YOLO detector Vulkan output.");
                    }

                    if (!has_pred && extracted_output.dims == 2) {
                        states[frame_index].pred_blob = extracted_output;
                        has_pred = true;
                        continue;
                    }
                    if (!has_proto && extracted_output.dims == 3) {
                        states[frame_index].proto_blob = extracted_output;
                        has_proto = true;
                    }
                }

                if (!has_pred || !has_proto) {
                    throw std::runtime_error("Failed to resolve YOLO segmentation outputs from Vulkan detector.");
                }
            }

            if (detector_cmd.submit_and_wait() != 0) {
                throw std::runtime_error("Failed to finalize YOLO detector Vulkan batch outputs.");
            }
        }

        ncnn::Option postprocess_opt = postprocess_runner.context_->net->opt;
        postprocess_opt.use_vulkan_compute = true;
        postprocess_opt.use_packing_layout = false;
        postprocess_opt.blob_vkallocator = postprocess_runner.context_->blob_vkallocator;
        postprocess_opt.workspace_vkallocator = postprocess_runner.context_->blob_vkallocator;
        postprocess_opt.staging_vkallocator = postprocess_runner.context_->staging_vkallocator;

        {
            ncnn::VkCompute postprocess_cmd(context_->vkdev);
            std::vector<ncnn::Extractor> postprocess_extractors;
            postprocess_extractors.reserve(batch_size);
            std::vector<ncnn::Mat> config_mats(batch_size);

            for (std::size_t frame_index = 0; frame_index < batch_size; ++frame_index) {
                const std::vector<int>& orig_shape = orig_shapes[frame_index];
                if (orig_shape.size() < 2) {
                    throw std::runtime_error("Each YOLO original shape must contain height and width.");
                }

                ncnn::Mat& config_mat = config_mats[frame_index];
                config_mat = ncnn::Mat(9);
                if (config_mat.empty()) {
                    throw std::runtime_error("Failed to allocate YOLO postprocess config tensor.");
                }
                float* config_ptr = static_cast<float*>(config_mat.data);
                config_ptr[0] = conf_threshold;
                config_ptr[1] = iou_threshold;
                config_ptr[2] = static_cast<float>(max_det);
                config_ptr[3] = static_cast<float>(num_classes);
                config_ptr[4] = agnostic_nms ? 1.f : 0.f;
                config_ptr[5] = static_cast<float>(input_shape[0]);
                config_ptr[6] = static_cast<float>(input_shape[1]);
                config_ptr[7] = static_cast<float>(orig_shape[0]);
                config_ptr[8] = static_cast<float>(orig_shape[1]);

                postprocess_extractors.push_back(postprocess_runner.create_extractor(true));
                ncnn::Extractor& postprocess_extractor = postprocess_extractors.back();
                if (postprocess_extractor.input("pred", states[frame_index].pred_blob) != 0) {
                    throw std::runtime_error(
                        "Failed to feed YOLO prediction tensor into Vulkan postprocess subnet.");
                }
                if (postprocess_extractor.input("proto", states[frame_index].proto_blob) != 0) {
                    throw std::runtime_error(
                        "Failed to feed YOLO proto tensor into Vulkan postprocess subnet.");
                }
                if (postprocess_extractor.input("config", config_mat) != 0) {
                    throw std::runtime_error("Failed to feed YOLO config tensor into Vulkan postprocess subnet.");
                }

                {
                    ncnn::VkMat extracted_boxes;
                    if (postprocess_extractor.extract("boxes", extracted_boxes, postprocess_cmd) != 0) {
                        throw std::runtime_error(
                            "Failed to extract YOLO boxes tensor from Vulkan postprocess subnet.");
                    }
                    states[frame_index].boxes_blob = extracted_boxes;
                }
                {
                    ncnn::VkMat extracted_selected;
                    if (postprocess_extractor.extract("selected", extracted_selected, postprocess_cmd) != 0) {
                        throw std::runtime_error(
                            "Failed to extract YOLO selected tensor from Vulkan postprocess subnet.");
                    }
                    states[frame_index].selected_blob = extracted_selected;
                }
                {
                    ncnn::VkMat extracted_count;
                    if (postprocess_extractor.extract("count", extracted_count, postprocess_cmd) != 0) {
                        throw std::runtime_error(
                            "Failed to extract YOLO count tensor from Vulkan postprocess subnet.");
                    }
                    postprocess_cmd.record_download(
                        extracted_count,
                        states[frame_index].count_mat,
                        postprocess_opt);
                }
            }

            if (postprocess_cmd.submit_and_wait() != 0) {
                throw std::runtime_error("Failed to finalize YOLO postprocess Vulkan batch outputs.");
            }
        }

        std::vector<YoloSegMaskFinalizeRecording> finalize_recordings(batch_size);
        bool has_finalize_work = false;
        {
            ncnn::VkCompute finalize_cmd(context_->vkdev);
            for (std::size_t frame_index = 0; frame_index < batch_size; ++frame_index) {
                if (states[frame_index].count_mat.dims != 1
                    || states[frame_index].count_mat.w < 1
                    || states[frame_index].count_mat.elemsize != sizeof(float)) {
                    throw std::runtime_error("Downloaded YOLO count tensor has an unexpected shape or dtype.");
                }

                const int output_capacity = std::min(
                    states[frame_index].boxes_blob.h,
                    states[frame_index].selected_blob.h);
                const int count = std::clamp(
                    static_cast<int>(static_cast<const float*>(states[frame_index].count_mat.data)[0]),
                    0,
                    output_capacity);
                finalize_recordings[frame_index] = record_yolo_segmentation_masks_gpu_finalize(
                    postprocess_runner.context_,
                    states[frame_index].boxes_blob,
                    states[frame_index].selected_blob,
                    states[frame_index].proto_blob,
                    count,
                    input_shape,
                    orig_shapes[frame_index],
                    finalize_cmd);
                has_finalize_work = has_finalize_work || count > 0;
            }

            if (has_finalize_work && finalize_cmd.submit_and_wait() != 0) {
                throw std::runtime_error("Failed to execute YOLO Vulkan mask finalize batch pipeline.");
            }
        }

        py::list results;
        for (std::size_t frame_index = 0; frame_index < batch_size; ++frame_index) {
            results.append(pack_yolo_segmentation_masks_gpu_finalize(finalize_recordings[frame_index]));
        }
        return results;
    }

private:
    ncnn::Extractor create_extractor(bool light_mode) const
    {
        ncnn::Extractor extractor = context_->net->create_extractor();
        extractor.set_light_mode(light_mode);
        extractor.set_blob_vkallocator(context_->blob_vkallocator);
        extractor.set_workspace_vkallocator(context_->blob_vkallocator);
        extractor.set_staging_vkallocator(context_->staging_vkallocator);
        return extractor;
    }

    void feed_inputs(ncnn::Extractor& extractor, const py::dict& inputs) const
    {
        for (const auto& item : inputs) {
            feed_input(
                extractor,
                py::cast<std::string>(item.first),
                item.second);
        }
    }

    void feed_input(
        ncnn::Extractor& extractor,
        const std::string& blob_name,
        const py::handle& value) const
    {
        if (py::isinstance<LadaVulkanTensor>(value)) {
            const auto& tensor = value.cast<const LadaVulkanTensor&>();
            if (!tensor.context) {
                throw std::runtime_error("Input Vulkan tensor is detached from its context.");
            }
            if (tensor.context->vkdev != context_->vkdev) {
                throw std::runtime_error("Input Vulkan tensor belongs to a different ncnn Vulkan device.");
            }
            const int ret = extractor.input(blob_name.c_str(), tensor.blob);
            if (ret != 0) {
                throw std::runtime_error("Failed to feed Vulkan tensor into ncnn extractor.");
            }
            return;
        }

        if (py::isinstance<ncnn::Mat>(value)) {
            const auto& mat = value.cast<const ncnn::Mat&>();
            const int ret = extractor.input(blob_name.c_str(), mat);
            if (ret != 0) {
                throw std::runtime_error("Failed to feed CPU tensor into ncnn extractor.");
            }
            return;
        }

        throw std::runtime_error("LadaVulkanNetRunner inputs must be ncnn.Mat or ncnn.LadaVulkanTensor.");
    }

    ncnn::Mat extract_cpu_output(
        ncnn::Extractor& extractor,
        const std::string& output_name) const
    {
        ncnn::Mat output;
        const int ret = extractor.extract(output_name.c_str(), output);
        if (ret != 0) {
            throw std::runtime_error("Failed to extract CPU tensor from ncnn extractor.");
        }

        if (output.elempack != 1) {
            ncnn::Option unpack_opt = context_->net->opt;
            unpack_opt.use_vulkan_compute = false;
            unpack_opt.use_packing_layout = false;

            ncnn::Mat unpacked_output;
            ncnn::convert_packing(output, unpacked_output, 1, unpack_opt);
            output = std::move(unpacked_output);
        }

        return output.clone();
    }

    std::shared_ptr<LadaVulkanContext> context_;
};

YoloSegMaskFinalizeRecording record_yolo_segmentation_masks_gpu_finalize(
    const std::shared_ptr<LadaVulkanContext>& context,
    const ncnn::VkMat& boxes_blob,
    const ncnn::VkMat& selected_blob,
    const ncnn::VkMat& proto_blob,
    int count,
    const std::vector<int>& input_shape,
    const std::vector<int>& orig_shape,
    ncnn::VkCompute& cmd)
{
    if (!context) {
        throw std::runtime_error("YOLO Vulkan finalize context is missing.");
    }
    if (input_shape.size() < 2 || orig_shape.size() < 2) {
        throw std::runtime_error("Input and original shapes must contain height and width.");
    }
    if (boxes_blob.dims != 2 || boxes_blob.w != 6) {
        throw std::runtime_error("YOLO boxes tensor must be a 2D Nx6 tensor.");
    }
    if (selected_blob.dims != 2 || selected_blob.w < 7) {
        throw std::runtime_error("YOLO selected tensor must be a 2D tensor with mask coefficients.");
    }
    if (proto_blob.dims != 3) {
        throw std::runtime_error("YOLO proto tensor must be 3D.");
    }

    const int input_height = std::max(input_shape[0], 0);
    const int input_width = std::max(input_shape[1], 0);
    const int orig_height = std::max(orig_shape[0], 0);
    const int orig_width = std::max(orig_shape[1], 0);
    if (input_height <= 0 || input_width <= 0 || orig_height <= 0 || orig_width <= 0) {
        throw std::runtime_error("YOLO input/original shapes must be positive.");
    }

    const int output_capacity = std::min(boxes_blob.h, selected_blob.h);
    count = std::clamp(count, 0, output_capacity);
    YoloSegMaskFinalizeRecording recording;
    recording.orig_height = orig_height;
    recording.orig_width = orig_width;
    recording.count = count;
    if (count == 0) {
        return recording;
    }

    ensure_yolo_seg_mask_finalize_pipelines(*context);

    ncnn::Option opt = context->net->opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.blob_vkallocator = context->blob_vkallocator;
    opt.workspace_vkallocator = context->blob_vkallocator;
    opt.staging_vkallocator = context->staging_vkallocator;

    recording.selected_unpacked = selected_blob;
    if (recording.selected_unpacked.elempack != 1) {
        context->vkdev->convert_packing(selected_blob, recording.selected_unpacked, 1, cmd, opt);
    }

    recording.proto_unpacked = proto_blob;
    if (recording.proto_unpacked.elempack != 1) {
        context->vkdev->convert_packing(proto_blob, recording.proto_unpacked, 1, cmd, opt);
    }

    if (recording.selected_unpacked.elemsize != recording.proto_unpacked.elemsize) {
        throw std::runtime_error("YOLO selected/proto Vulkan tensors must use the same storage precision.");
    }

    const int mask_dim = recording.selected_unpacked.w - 6;
    const bool proto_is_chw =
        mask_dim == recording.proto_unpacked.c || mask_dim != recording.proto_unpacked.w;
    const int proto_height = proto_is_chw ? recording.proto_unpacked.h : recording.proto_unpacked.c;
    const int proto_width = proto_is_chw ? recording.proto_unpacked.w : recording.proto_unpacked.h;
    if (mask_dim <= 0 || proto_height <= 0 || proto_width <= 0) {
        throw std::runtime_error("YOLO Vulkan tensors produced an invalid mask layout.");
    }

    recording.masks_blob.create(
        orig_width,
        orig_height,
        count,
        recording.selected_unpacked.elemsize,
        1,
        opt.blob_vkallocator);
    if (recording.masks_blob.empty()) {
        throw std::runtime_error("Failed to allocate YOLO Vulkan mask output tensor.");
    }
    if (recording.selected_unpacked.elemsize == sizeof(float)) {
        recording.proto_logits_blob.create(
            proto_width,
            proto_height,
            count,
            recording.selected_unpacked.elemsize,
            1,
            opt.workspace_vkallocator);
        recording.input_masks_blob.create(
            input_width,
            input_height,
            count,
            recording.selected_unpacked.elemsize,
            1,
            opt.workspace_vkallocator);
        if (recording.proto_logits_blob.empty() || recording.input_masks_blob.empty()) {
            throw std::runtime_error("Failed to allocate staged YOLO Vulkan mask buffers.");
        }

        {
            std::vector<ncnn::VkMat> bindings(3);
            bindings[0] = recording.selected_unpacked;
            bindings[1] = recording.proto_unpacked;
            bindings[2] = recording.proto_logits_blob;

            std::vector<ncnn::vk_constant_type> constants(9);
            constants[0].i = recording.selected_unpacked.w;
            constants[1].i = recording.proto_unpacked.w;
            constants[2].i = recording.proto_unpacked.h;
            constants[3].i = recording.proto_unpacked.c;
            constants[4].i = proto_is_chw ? 1 : 0;
            constants[5].i = mask_dim;
            constants[6].i = count;
            constants[7].i = input_height;
            constants[8].i = input_width;
            cmd.record_pipeline(
                context->yolo_seg_mask_proto_decode_pipeline,
                bindings,
                constants,
                recording.proto_logits_blob);
        }

        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = recording.proto_logits_blob;
            bindings[1] = recording.input_masks_blob;

            std::vector<ncnn::vk_constant_type> constants(5);
            constants[0].i = proto_width;
            constants[1].i = proto_height;
            constants[2].i = count;
            constants[3].i = input_height;
            constants[4].i = input_width;
            cmd.record_pipeline(
                context->yolo_seg_mask_input_resize_pipeline,
                bindings,
                constants,
                recording.input_masks_blob);
        }

        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = recording.input_masks_blob;
            bindings[1] = recording.masks_blob;

            std::vector<ncnn::vk_constant_type> constants(5);
            constants[0].i = input_width;
            constants[1].i = input_height;
            constants[2].i = count;
            constants[3].i = orig_height;
            constants[4].i = orig_width;
            cmd.record_pipeline(
                context->yolo_seg_mask_orig_resize_pipeline,
                bindings,
                constants,
                recording.masks_blob);
        }
    } else {
        std::vector<ncnn::VkMat> bindings(3);
        bindings[0] = recording.selected_unpacked;
        bindings[1] = recording.proto_unpacked;
        bindings[2] = recording.masks_blob;

        std::vector<ncnn::vk_constant_type> constants(12);
        constants[0].i = recording.selected_unpacked.w;
        constants[1].i = recording.selected_unpacked.h;
        constants[2].i = recording.proto_unpacked.w;
        constants[3].i = recording.proto_unpacked.h;
        constants[4].i = recording.proto_unpacked.c;
        constants[5].i = proto_is_chw ? 1 : 0;
        constants[6].i = mask_dim;
        constants[7].i = count;
        constants[8].i = input_height;
        constants[9].i = input_width;
        constants[10].i = orig_height;
        constants[11].i = orig_width;
        cmd.record_pipeline(
            context->yolo_seg_mask_direct_pipeline,
            bindings,
            constants,
            recording.masks_blob);
    }

    cmd.record_download(boxes_blob, recording.boxes_mat, opt);
    cmd.record_download(recording.masks_blob, recording.masks_mat, opt);
    return recording;
}

py::dict pack_yolo_segmentation_masks_gpu_finalize(const YoloSegMaskFinalizeRecording& recording)
{
    py::dict output;
    if (recording.count == 0) {
        output[py::str("boxes")] = py::array_t<float>(
            py::array::ShapeContainer{static_cast<py::ssize_t>(0), static_cast<py::ssize_t>(6)});
        output[py::str("masks")] = py::array_t<unsigned char>(
            py::array::ShapeContainer{
                static_cast<py::ssize_t>(0),
                static_cast<py::ssize_t>(recording.orig_height),
                static_cast<py::ssize_t>(recording.orig_width),
            });
        return output;
    }

    if (recording.boxes_mat.dims != 2
        || recording.boxes_mat.w != 6
        || recording.boxes_mat.elemsize != sizeof(float)) {
        throw std::runtime_error("Downloaded YOLO boxes tensor has an unexpected shape or dtype.");
    }
    if (
        recording.masks_mat.dims != 3
        || recording.masks_mat.w != recording.orig_width
        || recording.masks_mat.h != recording.orig_height
        || recording.masks_mat.c != recording.count
        || recording.masks_mat.elemsize != sizeof(float)) {
        throw std::runtime_error("Downloaded YOLO masks tensor has an unexpected shape or dtype.");
    }

    const std::size_t plane_size = static_cast<std::size_t>(recording.orig_height)
        * static_cast<std::size_t>(recording.orig_width);
    std::vector<int> kept_indices;
    kept_indices.reserve(static_cast<std::size_t>(recording.count));
    for (int det_index = 0; det_index < recording.count; ++det_index) {
        const float* mask_ptr = static_cast<const float*>(recording.masks_mat.channel(det_index).data);
        bool active = false;
        for (std::size_t pixel_index = 0; pixel_index < plane_size; ++pixel_index) {
            if (mask_ptr[pixel_index] > 127.f) {
                active = true;
                break;
            }
        }
        if (active) {
            kept_indices.push_back(det_index);
        }
    }

    py::array_t<float> kept_boxes(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(kept_indices.size()),
            static_cast<py::ssize_t>(6),
        });
    py::array_t<unsigned char> kept_masks(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(kept_indices.size()),
            static_cast<py::ssize_t>(recording.orig_height),
            static_cast<py::ssize_t>(recording.orig_width),
        });

    float* kept_boxes_ptr = kept_boxes.mutable_data();
    unsigned char* kept_masks_ptr = kept_masks.mutable_data();
    for (std::size_t output_index = 0; output_index < kept_indices.size(); ++output_index) {
        const int det_index = kept_indices[output_index];
        std::memcpy(
            kept_boxes_ptr + output_index * 6,
            recording.boxes_mat.row(det_index),
            static_cast<std::size_t>(6) * sizeof(float));

        const float* mask_ptr = static_cast<const float*>(recording.masks_mat.channel(det_index).data);
        unsigned char* dst_mask = kept_masks_ptr + output_index * plane_size;
        for (std::size_t pixel_index = 0; pixel_index < plane_size; ++pixel_index) {
            dst_mask[pixel_index] = mask_ptr[pixel_index] > 127.f ? static_cast<unsigned char>(255) : 0;
        }
    }

    output[py::str("boxes")] = std::move(kept_boxes);
    output[py::str("masks")] = std::move(kept_masks);
    return output;
}

py::dict finalize_yolo_segmentation_masks_gpu_impl(
    const std::shared_ptr<LadaVulkanContext>& context,
    const ncnn::VkMat& boxes_blob,
    const ncnn::VkMat& selected_blob,
    const ncnn::VkMat& proto_blob,
    int count,
    const std::vector<int>& input_shape,
    const std::vector<int>& orig_shape)
{
    if (!context) {
        throw std::runtime_error("Vulkan context is required for YOLO mask finalize.");
    }
    ncnn::VkCompute cmd(context->vkdev);
    YoloSegMaskFinalizeRecording recording = record_yolo_segmentation_masks_gpu_finalize(
        context,
        boxes_blob,
        selected_blob,
        proto_blob,
        count,
        input_shape,
        orig_shape,
        cmd);
    if (recording.count > 0 && cmd.submit_and_wait() != 0) {
        throw std::runtime_error("Failed to execute YOLO Vulkan mask finalize pipeline.");
    }
    return pack_yolo_segmentation_masks_gpu_finalize(recording);
}

lada::YoloSegPostprocessResult postprocess_yolo_segmentation_mat(
    const ncnn::Mat& pred,
    const ncnn::Mat& proto,
    const lada::YoloSegPostprocessConfig& config)
{
    if (pred.dims != 2) {
        throw std::runtime_error("YOLO prediction tensor must be 2D.");
    }
    if (proto.dims != 3) {
        throw std::runtime_error("YOLO proto tensor must be 3D.");
    }
    if (pred.elemsize != sizeof(float) || proto.elemsize != sizeof(float)) {
        throw std::runtime_error("YOLO postprocess expects fp32 NCNN tensors.");
    }

    return lada::postprocess_yolo_segmentation_cpu(
        static_cast<const float*>(pred.data),
        pred.h,
        pred.w,
        static_cast<const float*>(proto.data),
        proto.c,
        proto.h,
        proto.w,
        config);
}

lada::YoloSegPostprocessResult postprocess_yolo_segmentation_from_selected_mat(
    const ncnn::Mat& boxes,
    const ncnn::Mat& selected,
    const ncnn::Mat& proto,
    const lada::YoloSegPostprocessConfig& config)
{
    if (boxes.dims != 2 || selected.dims != 2) {
        throw std::runtime_error("YOLO selected boxes and features tensors must be 2D.");
    }
    if (proto.dims != 3) {
        throw std::runtime_error("YOLO proto tensor must be 3D.");
    }
    if (
        boxes.elemsize != sizeof(float)
        || selected.elemsize != sizeof(float)
        || proto.elemsize != sizeof(float)) {
        throw std::runtime_error("YOLO selected postprocess expects fp32 NCNN tensors.");
    }
    if (boxes.w != 6 || selected.w < 7 || boxes.h != selected.h) {
        throw std::runtime_error("YOLO selected postprocess tensors have unexpected shapes.");
    }

    const int mask_dim = selected.w - 6;
    lada::YoloSegSelectionResult selection_result;
    selection_result.mask_dim = mask_dim;
    selection_result.proto_is_chw = mask_dim == proto.c || mask_dim != proto.w;
    selection_result.detections.reserve(static_cast<std::size_t>(boxes.h));

    for (int row_index = 0; row_index < boxes.h; ++row_index) {
        const float* box_ptr = boxes.row(row_index);
        const float* selected_ptr = selected.row(row_index);

        lada::YoloSegSelectedDetection detection;
        for (int feature_index = 0; feature_index < 6; ++feature_index) {
            detection.box[static_cast<std::size_t>(feature_index)] = box_ptr[feature_index];
        }
        detection.selected.assign(selected_ptr, selected_ptr + selected.w);
        selection_result.detections.push_back(std::move(detection));
    }

    return lada::finalize_yolo_segmentation_cpu(
        selection_result,
        static_cast<const float*>(proto.data),
        proto.c,
        proto.h,
        proto.w,
        config);
}

py::object ncnn_mat_to_numpy(const ncnn::Mat& mat)
{
    if (mat.elempack != 1 || mat.elemsize != sizeof(float)) {
        throw std::runtime_error("NCNN Python bridge expects unpacked float32 tensors.");
    }

    if (mat.dims == 1) {
        py::array_t<float> output({static_cast<py::ssize_t>(mat.w)});
        std::memcpy(output.mutable_data(), mat.data, static_cast<size_t>(mat.w) * sizeof(float));
        return std::move(output);
    }

    if (mat.dims == 2) {
        py::array_t<float> output({static_cast<py::ssize_t>(mat.h), static_cast<py::ssize_t>(mat.w)});
        std::memcpy(output.mutable_data(), mat.data, static_cast<size_t>(mat.h) * mat.w * sizeof(float));
        return std::move(output);
    }

    if (mat.dims == 3) {
        py::array_t<float> output(
            {static_cast<py::ssize_t>(mat.c), static_cast<py::ssize_t>(mat.h), static_cast<py::ssize_t>(mat.w)});
        std::memcpy(
            output.mutable_data(),
            mat.data,
            static_cast<size_t>(mat.c) * mat.h * mat.w * sizeof(float));
        return std::move(output);
    }

    throw std::runtime_error("NCNN Python bridge only supports 1D/2D/3D tensors.");
}

py::dict pack_yolo_seg_postprocess_result(const lada::YoloSegPostprocessResult& result)
{
    const py::ssize_t detection_count = static_cast<py::ssize_t>(result.detections.size());
    py::array_t<float> boxes({detection_count, static_cast<py::ssize_t>(6)});
    py::array_t<unsigned char> masks(
        {detection_count, static_cast<py::ssize_t>(result.mask_height), static_cast<py::ssize_t>(result.mask_width)});

    auto boxes_view = boxes.mutable_unchecked<2>();
    auto masks_view = masks.mutable_unchecked<3>();
    for (py::ssize_t detection_index = 0; detection_index < detection_count; ++detection_index) {
        const auto& detection = result.detections[static_cast<std::size_t>(detection_index)];
        for (int box_index = 0; box_index < 6; ++box_index) {
            boxes_view(detection_index, box_index) = detection.box[static_cast<std::size_t>(box_index)];
        }
        for (int y = 0; y < result.mask_height; ++y) {
            for (int x = 0; x < result.mask_width; ++x) {
                masks_view(detection_index, y, x) =
                    detection.mask[static_cast<std::size_t>(y) * result.mask_width + x];
            }
        }
    }

    py::dict output;
    output[py::str("boxes")] = std::move(boxes);
    output[py::str("masks")] = std::move(masks);
    return output;
}

py::dict postprocess_yolo_segmentation_py(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& pred,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& proto,
    const std::vector<int>& input_shape,
    const std::vector<int>& orig_shape,
    float conf_threshold,
    float iou_threshold,
    int max_det,
    int num_classes,
    bool agnostic_nms)
{
    if (pred.ndim() != 2) {
        throw std::runtime_error("YOLO prediction tensor must be 2D.");
    }
    if (proto.ndim() != 3) {
        throw std::runtime_error("YOLO proto tensor must be 3D.");
    }
    if (input_shape.size() < 2 || orig_shape.size() < 2) {
        throw std::runtime_error("Input and original shapes must contain height and width.");
    }

    lada::YoloSegPostprocessConfig config;
    config.conf_threshold = conf_threshold;
    config.iou_threshold = iou_threshold;
    config.max_det = max_det;
    config.num_classes = num_classes;
    config.agnostic_nms = agnostic_nms;
    config.input_height = input_shape[0];
    config.input_width = input_shape[1];
    config.orig_height = orig_shape[0];
    config.orig_width = orig_shape[1];

    const py::buffer_info pred_info = pred.request();
    const py::buffer_info proto_info = proto.request();
    return pack_yolo_seg_postprocess_result(
        lada::postprocess_yolo_segmentation_cpu(
            static_cast<const float*>(pred_info.ptr),
            static_cast<int>(pred.shape(0)),
            static_cast<int>(pred.shape(1)),
            static_cast<const float*>(proto_info.ptr),
            static_cast<int>(proto.shape(0)),
            static_cast<int>(proto.shape(1)),
            static_cast<int>(proto.shape(2)),
            config));
}

py::dict postprocess_yolo_segmentation_from_selected_py(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& boxes,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& selected,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& proto,
    const std::vector<int>& input_shape,
    const std::vector<int>& orig_shape)
{
    if (boxes.ndim() != 2 || selected.ndim() != 2) {
        throw std::runtime_error("YOLO selected boxes and features tensors must be 2D.");
    }
    if (proto.ndim() != 3) {
        throw std::runtime_error("YOLO proto tensor must be 3D.");
    }
    if (boxes.shape(0) != selected.shape(0)) {
        throw std::runtime_error("YOLO selected boxes and features must have the same row count.");
    }
    if (input_shape.size() < 2 || orig_shape.size() < 2) {
        throw std::runtime_error("Input and original shapes must contain height and width.");
    }

    lada::YoloSegPostprocessConfig config;
    config.input_height = input_shape[0];
    config.input_width = input_shape[1];
    config.orig_height = orig_shape[0];
    config.orig_width = orig_shape[1];

    const py::buffer_info boxes_info = boxes.request();
    const py::buffer_info selected_info = selected.request();
    const py::buffer_info proto_info = proto.request();

    ncnn::Mat boxes_mat(static_cast<int>(boxes.shape(1)), static_cast<int>(boxes.shape(0)));
    ncnn::Mat selected_mat(static_cast<int>(selected.shape(1)), static_cast<int>(selected.shape(0)));
    ncnn::Mat proto_mat(
        static_cast<int>(proto.shape(2)),
        static_cast<int>(proto.shape(1)),
        static_cast<int>(proto.shape(0)));
    if (boxes_mat.empty() || selected_mat.empty() || proto_mat.empty()) {
        throw std::runtime_error("Failed to allocate NCNN tensors for YOLO selected postprocess.");
    }

    std::memcpy(
        boxes_mat.data,
        boxes_info.ptr,
        static_cast<std::size_t>(boxes.shape(0)) * static_cast<std::size_t>(boxes.shape(1)) * sizeof(float));
    std::memcpy(
        selected_mat.data,
        selected_info.ptr,
        static_cast<std::size_t>(selected.shape(0)) * static_cast<std::size_t>(selected.shape(1)) * sizeof(float));
    std::memcpy(
        proto_mat.data,
        proto_info.ptr,
        static_cast<std::size_t>(proto.shape(0)) * static_cast<std::size_t>(proto.shape(1)) * static_cast<std::size_t>(proto.shape(2)) * sizeof(float));

    return pack_yolo_seg_postprocess_result(
        postprocess_yolo_segmentation_from_selected_mat(
            boxes_mat,
            selected_mat,
            proto_mat,
            config));
}

} // namespace
#endif

void bind_lada_local_runtime(py::module_& m)
{
    m.def(
        "register_lada_custom_layers",
        [](ncnn::Net& net) {
            int ret = lada::register_torch_conv2d_layers(net);
            if (ret != 0) {
                return ret;
            }
            ret = lada::register_torchvision_deform_conv2d_layers(net);
            if (ret != 0) {
                return ret;
            }
            ret = lada::register_lada_gridsample_layers(net);
            if (ret != 0) {
                return ret;
            }
            ret = lada::register_lada_yolo_attention_layers(net);
            if (ret != 0) {
                return ret;
            }
            return lada::register_lada_yolo_seg_postprocess_layers(net);
        },
        py::arg("net"),
        "Register all built-in Lada custom layers on an ncnn.Net instance.");

    m.def(
        "register_torch_conv2d_layers",
        [](ncnn::Net& net) {
            return lada::register_torch_conv2d_layers(net);
        },
        py::arg("net"),
        "Register the built-in torch conv2d custom layers on an ncnn.Net instance.");

    m.def(
        "register_torchvision_deform_conv2d_layers",
        [](ncnn::Net& net) {
            return lada::register_torchvision_deform_conv2d_layers(net);
        },
        py::arg("net"),
        "Register the built-in torchvision deform-conv custom layers on an ncnn.Net instance.");

    m.def(
        "register_lada_gridsample_layers",
        [](ncnn::Net& net) {
            return lada::register_lada_gridsample_layers(net);
        },
        py::arg("net"),
        "Register the built-in Lada GridSample custom layers on an ncnn.Net instance.");

    m.def(
        "register_lada_yolo_attention_layers",
        [](ncnn::Net& net) {
            return lada::register_lada_yolo_attention_layers(net);
        },
        py::arg("net"),
        "Register the built-in Lada YOLO attention custom layers on an ncnn.Net instance.");

    m.def(
        "register_lada_yolo_seg_postprocess_layers",
        [](ncnn::Net& net) {
            return lada::register_lada_yolo_seg_postprocess_layers(net);
        },
        py::arg("net"),
        "Register the built-in Lada YOLO segmentation postprocess custom layers on an ncnn.Net instance.");

    m.attr("has_lada_torch_conv2d") = py::bool_(true);
    m.attr("has_lada_torchvision_deform_conv2d") = py::bool_(true);
    m.attr("has_lada_gridsample") = py::bool_(true);
    m.attr("has_lada_yolo_attention") = py::bool_(true);
    m.attr("has_lada_yolo_seg_postprocess_layer") = py::bool_(true);
    m.attr("has_lada_yolo_seg_postprocess_vulkan") = py::bool_(true);
#if NCNN_VULKAN
    m.def(
        "postprocess_yolo_segmentation",
        &postprocess_yolo_segmentation_py,
        py::arg("pred"),
        py::arg("proto"),
        py::arg("input_shape"),
        py::arg("orig_shape"),
        py::arg("conf_threshold"),
        py::arg("iou_threshold"),
        py::arg("max_det"),
        py::arg("num_classes"),
        py::arg("agnostic_nms") = false,
        "Run native YOLO segmentation postprocess and return final boxes and masks.");
    m.def(
        "postprocess_yolo_segmentation_from_selected",
        &postprocess_yolo_segmentation_from_selected_py,
        py::arg("boxes"),
        py::arg("selected"),
        py::arg("proto"),
        py::arg("input_shape"),
        py::arg("orig_shape"),
        "Finalize YOLO segmentation masks from selected detections and proto features.");
    m.def(
        "finalize_yolo_segmentation_masks_gpu",
        &finalize_yolo_segmentation_masks_gpu_py,
        py::arg("boxes"),
        py::arg("selected"),
        py::arg("proto"),
        py::arg("count"),
        py::arg("input_shape"),
        py::arg("orig_shape"),
        "Finalize YOLO segmentation masks on Vulkan from selected detections.");
    m.attr("has_lada_yolo_seg_postprocess") = py::bool_(true);
    m.attr("has_lada_yolo_seg_mask_finalize_vulkan") = py::bool_(true);
#else
    m.attr("has_lada_yolo_seg_postprocess") = py::bool_(false);
    m.attr("has_lada_yolo_seg_mask_finalize_vulkan") = py::bool_(false);
#endif

#if NCNN_VULKAN
    py::class_<LadaVulkanTensor>(m, "LadaVulkanTensor")
        .def("download", &LadaVulkanTensor::download)
        .def_property_readonly("dims", &LadaVulkanTensor::dims)
        .def_property_readonly("w", &LadaVulkanTensor::w)
        .def_property_readonly("h", &LadaVulkanTensor::h)
        .def_property_readonly("c", &LadaVulkanTensor::c)
        .def_property_readonly("elempack", &LadaVulkanTensor::elempack)
        .def_property_readonly("elemsize", &LadaVulkanTensor::elemsize)
        .def("empty", &LadaVulkanTensor::empty);

    py::class_<LadaVulkanNetRunner>(m, "LadaVulkanNetRunner")
        .def(py::init<py::object>(), py::arg("net"))
        .def(
            "preprocess_bgr_u8",
            &LadaVulkanNetRunner::preprocess_bgr_u8,
            py::arg("image"),
            py::arg("input_shape"))
        .def(
            "preprocess_bgr_u8_batch",
            &LadaVulkanNetRunner::preprocess_bgr_u8_batch,
            py::arg("images"),
            py::arg("input_shape"))
        .def("run", &LadaVulkanNetRunner::run, py::arg("inputs"), py::arg("output_name") = "out0")
        .def("run_many", &LadaVulkanNetRunner::run_many, py::arg("inputs"), py::arg("output_names"))
        .def("run_to_cpu", &LadaVulkanNetRunner::run_to_cpu, py::arg("inputs"), py::arg("output_name") = "out0")
        .def(
            "run_many_to_cpu",
            &LadaVulkanNetRunner::run_many_to_cpu,
            py::arg("inputs"),
            py::arg("output_names"))
        .def(
            "run_yolo_segmentation",
            &LadaVulkanNetRunner::run_yolo_segmentation,
            py::arg("inputs"),
            py::arg("output_names"),
            py::arg("input_shape"),
            py::arg("orig_shape"),
            py::arg("conf_threshold"),
            py::arg("iou_threshold"),
            py::arg("max_det"),
            py::arg("num_classes"),
            py::arg("agnostic_nms") = false)
        .def(
            "run_yolo_segmentation_subnet",
            &LadaVulkanNetRunner::run_yolo_segmentation_subnet,
            py::arg("inputs"),
            py::arg("output_names"),
            py::arg("postprocess_runner"),
            py::arg("input_shape"),
            py::arg("orig_shape"),
            py::arg("conf_threshold"),
            py::arg("iou_threshold"),
            py::arg("max_det"),
            py::arg("num_classes"),
            py::arg("agnostic_nms") = false)
        .def(
            "run_yolo_segmentation_subnet_batch",
            &LadaVulkanNetRunner::run_yolo_segmentation_subnet_batch,
            py::arg("input_frames"),
            py::arg("input_name"),
            py::arg("output_names"),
            py::arg("postprocess_runner"),
            py::arg("input_shape"),
            py::arg("orig_shapes"),
            py::arg("conf_threshold"),
            py::arg("iou_threshold"),
            py::arg("max_det"),
            py::arg("num_classes"),
            py::arg("agnostic_nms") = false);

    m.attr("has_lada_vulkan_net_runner") = py::bool_(true);
#else
    m.attr("has_lada_vulkan_net_runner") = py::bool_(false);
#endif

    lada::bind_basicvsrpp_clip_runner(m);
    lada::bind_vulkan_blend_runtime(m);
}
