// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#include "vulkan_blend_runtime.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "command.h"
#include "gpu.h"
#include "mat.h"
#include "modelbin.h"
#include "net.h"
#include "torch_conv2d_layer.h"

namespace py = pybind11;

#if NCNN_VULKAN
namespace {

static const char kLadaBlendBaseMaskShader[] = R"(
#version 450

layout(binding = 0) readonly buffer mask_blob { sfp mask_blob_data[]; };
layout(binding = 1) writeonly buffer base_blob { sfp base_blob_data[]; };

layout(push_constant) uniform parameter
{
    int width;
    int height;
    int inner_top;
    int inner_bottom;
    int inner_left;
    int inner_right;
    int mask_cstep;
    int base_cstep;
} p;

float mask_value_at(int x, int y)
{
    return float(buffer_ld1(mask_blob_data, y * p.width + x));
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    if (x >= p.width || y >= p.height)
        return;

    const float mask_value = mask_value_at(x, y) > 0.0 ? 1.0 : 0.0;
    const float inner_value =
        (x >= p.inner_left && x < p.width - p.inner_right && y >= p.inner_top && y < p.height - p.inner_bottom)
        ? 1.0
        : 0.0;
    buffer_st1(base_blob_data, y * p.width + x, afp(max(mask_value, inner_value)));
}
)";

static const char kLadaBlendApplyShader[] = R"(
#version 450

layout(binding = 0) readonly buffer blend_mask_blob { sfp blend_mask_blob_data[]; };
layout(binding = 1) readonly buffer frame_blob { sfp frame_blob_data[]; };
layout(binding = 2) readonly buffer clip_blob { sfp clip_blob_data[]; };
layout(binding = 3) writeonly buffer output_blob { sfp output_blob_data[]; };

layout(push_constant) uniform parameter
{
    int width;
    int height;
    int channels;
    int blend_mask_cstep;
    int frame_cstep;
    int clip_cstep;
    int output_cstep;
} p;

float blend_mask_value_at(int x, int y)
{
    return float(buffer_ld1(blend_mask_blob_data, y * p.width + x));
}

float frame_value_at(int channel, int x, int y)
{
    return float(buffer_ld1(frame_blob_data, channel * p.frame_cstep + y * p.width + x));
}

float clip_value_at(int channel, int x, int y)
{
    return float(buffer_ld1(clip_blob_data, channel * p.clip_cstep + y * p.width + x));
}

void output_store(int channel, int x, int y, float value)
{
    buffer_st1(output_blob_data, channel * p.output_cstep + y * p.width + x, afp(value));
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int channel = int(gl_GlobalInvocationID.z);
    if (x >= p.width || y >= p.height || channel >= p.channels)
        return;

    const float blend_mask = blend_mask_value_at(x, y);
    const float frame_value = frame_value_at(channel, x, y);
    const float clip_value = clip_value_at(channel, x, y);
    const float blended = clamp(round(frame_value + (clip_value - frame_value) * blend_mask), 0.0, 255.0);
    output_store(channel, x, y, blended);
}
)";

static const char kLadaBlendCopyShader[] = R"(
#version 450

layout(binding = 0) readonly buffer clip_blob { sfp clip_blob_data[]; };
layout(binding = 1) writeonly buffer output_blob { sfp output_blob_data[]; };

layout(push_constant) uniform parameter
{
    int width;
    int height;
    int channels;
    int clip_cstep;
    int output_cstep;
} p;

float clip_value_at(int channel, int x, int y)
{
    return float(buffer_ld1(clip_blob_data, channel * p.clip_cstep + y * p.width + x));
}

void output_store(int channel, int x, int y, float value)
{
    buffer_st1(output_blob_data, channel * p.output_cstep + y * p.width + x, afp(value));
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int channel = int(gl_GlobalInvocationID.z);
    if (x >= p.width || y >= p.height || channel >= p.channels)
        return;

    output_store(channel, x, y, clip_value_at(channel, x, y));
}
)";

static const char kLadaBlendResizeImageShader[] = R"(
#version 450

layout(binding = 0) readonly buffer input_blob { sfp input_blob_data[]; };
layout(binding = 1) writeonly buffer output_blob { sfp output_blob_data[]; };

layout(push_constant) uniform parameter
{
    int src_w;
    int src_h;
    int dst_w;
    int dst_h;
    int crop_left;
    int crop_top;
    int crop_w;
    int crop_h;
    int input_cstep;
    int output_cstep;
} p;

float input_value_at(int channel, int x, int y)
{
    return float(buffer_ld1(input_blob_data, channel * p.input_cstep + y * p.src_w + x));
}

void output_store(int channel, int x, int y, float value)
{
    buffer_st1(output_blob_data, channel * p.output_cstep + y * p.dst_w + x, afp(value));
}

float sample_resized_value(int channel, int x, int y)
{
    if (p.crop_w == 1 && p.crop_h == 1)
        return input_value_at(channel, p.crop_left, p.crop_top);

    const float crop_x = clamp(
        (float(x) + 0.5) * float(p.crop_w) / float(p.dst_w) - 0.5,
        0.0,
        float(max(p.crop_w - 1, 0)));
    const float crop_y = clamp(
        (float(y) + 0.5) * float(p.crop_h) / float(p.dst_h) - 0.5,
        0.0,
        float(max(p.crop_h - 1, 0)));

    const int x0_local = min(int(floor(crop_x)), p.crop_w - 1);
    const int y0_local = min(int(floor(crop_y)), p.crop_h - 1);
    const int x1_local = min(x0_local + 1, p.crop_w - 1);
    const int y1_local = min(y0_local + 1, p.crop_h - 1);
    const float lx = crop_x - float(x0_local);
    const float ly = crop_y - float(y0_local);
    const float hx = 1.0 - lx;
    const float hy = 1.0 - ly;

    const int x0 = p.crop_left + x0_local;
    const int y0 = p.crop_top + y0_local;
    const int x1 = p.crop_left + x1_local;
    const int y1 = p.crop_top + y1_local;

    const float v00 = input_value_at(channel, x0, y0);
    const float v01 = input_value_at(channel, x1, y0);
    const float v10 = input_value_at(channel, x0, y1);
    const float v11 = input_value_at(channel, x1, y1);
    const float top = v00 * hx + v01 * lx;
    const float bottom = v10 * hx + v11 * lx;
    return top * hy + bottom * ly;
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int channel = int(gl_GlobalInvocationID.z);
    if (x >= p.dst_w || y >= p.dst_h || channel >= 3)
        return;

    output_store(channel, x, y, sample_resized_value(channel, x, y));
}
)";

static const char kLadaBlendResizeMaskShader[] = R"(
#version 450

layout(binding = 0) readonly buffer input_blob { sfp input_blob_data[]; };
layout(binding = 1) writeonly buffer output_blob { sfp output_blob_data[]; };

layout(push_constant) uniform parameter
{
    int src_w;
    int src_h;
    int dst_w;
    int dst_h;
    int crop_left;
    int crop_top;
    int crop_w;
    int crop_h;
    int input_cstep;
    int output_cstep;
} p;

float input_value_at(int x, int y)
{
    return float(buffer_ld1(input_blob_data, y * p.src_w + x));
}

void output_store(int x, int y, float value)
{
    buffer_st1(output_blob_data, y * p.dst_w + x, afp(value));
}

float sample_resized_value(int x, int y)
{
    if (p.crop_w == 1 && p.crop_h == 1)
        return input_value_at(p.crop_left, p.crop_top) > 0.0 ? 1.0 : 0.0;

    const float crop_x = (float(x) + 0.5) * float(p.crop_w) / float(p.dst_w) - 0.5;
    const float crop_y = (float(y) + 0.5) * float(p.crop_h) / float(p.dst_h) - 0.5;
    const int src_x = p.crop_left + clamp(int(floor(crop_x + 0.5)), 0, p.crop_w - 1);
    const int src_y = p.crop_top + clamp(int(floor(crop_y + 0.5)), 0, p.crop_h - 1);
    return input_value_at(src_x, src_y) > 0.0 ? 1.0 : 0.0;
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    if (x >= p.dst_w || y >= p.dst_h)
        return;

    output_store(x, y, sample_resized_value(x, y));
}
)";

struct BlendConvKey
{
    int kernel_size = 0;

    bool operator==(const BlendConvKey& other) const
    {
        return kernel_size == other.kernel_size;
    }
};

struct BlendConvKeyHash
{
    std::size_t operator()(const BlendConvKey& key) const noexcept
    {
        return std::hash<int>()(key.kernel_size);
    }
};

struct LadaVulkanBlendContext
{
    LadaVulkanBlendContext()
    {
        net = std::make_unique<ncnn::Net>();
        net->set_vulkan_device(0);
        net->opt.use_vulkan_compute = true;
        net->opt.use_packing_layout = false;
        net->opt.use_fp16_storage = false;
        net->opt.use_fp16_packed = false;
        net->opt.use_fp16_arithmetic = false;
        net->opt.num_threads = 1;

        vkdev = net->vulkan_device();
        if (vkdev == nullptr || !vkdev->is_valid()) {
            throw std::runtime_error("Failed to initialize Vulkan device for Lada blend runtime.");
        }

        blob_vkallocator = vkdev->acquire_blob_allocator();
        staging_vkallocator = vkdev->acquire_staging_allocator();
        if (blob_vkallocator == nullptr || staging_vkallocator == nullptr) {
            throw std::runtime_error("Failed to acquire Vulkan allocators for Lada blend runtime.");
        }
    }

    ~LadaVulkanBlendContext()
    {
        blur_conv_cache.clear();

        delete build_base_mask_pipeline;
        delete apply_blend_pipeline;
        delete copy_clip_pipeline;
        delete resize_image_pipeline;
        delete resize_mask_pipeline;
        build_base_mask_pipeline = nullptr;
        apply_blend_pipeline = nullptr;
        copy_clip_pipeline = nullptr;
        resize_image_pipeline = nullptr;
        resize_mask_pipeline = nullptr;

        if (vkdev != nullptr && blob_vkallocator != nullptr) {
            vkdev->reclaim_blob_allocator(blob_vkallocator);
        }
        if (vkdev != nullptr && staging_vkallocator != nullptr) {
            vkdev->reclaim_staging_allocator(staging_vkallocator);
        }
    }

    ncnn::Option option() const
    {
        ncnn::Option opt = net->opt;
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_arithmetic = false;
        opt.blob_vkallocator = blob_vkallocator;
        opt.workspace_vkallocator = blob_vkallocator;
        opt.staging_vkallocator = staging_vkallocator;
        return opt;
    }

    std::unique_ptr<ncnn::Net> net;
    const ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::VkAllocator* blob_vkallocator = nullptr;
    ncnn::VkAllocator* staging_vkallocator = nullptr;
    ncnn::Pipeline* build_base_mask_pipeline = nullptr;
    ncnn::Pipeline* apply_blend_pipeline = nullptr;
    ncnn::Pipeline* copy_clip_pipeline = nullptr;
    ncnn::Pipeline* resize_image_pipeline = nullptr;
    ncnn::Pipeline* resize_mask_pipeline = nullptr;
    std::unordered_map<BlendConvKey, std::unique_ptr<lada::TorchConv2DLayer>, BlendConvKeyHash> blur_conv_cache;
};

LadaVulkanBlendContext& get_blend_context()
{
    static LadaVulkanBlendContext context;
    return context;
}

ncnn::Mat create_box_blur_weight(int kernel_size)
{
    ncnn::Mat weight(kernel_size * kernel_size);
    if (weight.empty()) {
        throw std::runtime_error("Failed to allocate box-blur kernel weights.");
    }

    const float value = 1.f / float(kernel_size * kernel_size);
    float* ptr = weight;
    for (int index = 0; index < kernel_size * kernel_size; ++index) {
        ptr[index] = value;
    }
    return weight;
}

void ensure_shader_pipelines(LadaVulkanBlendContext& context)
{
    if (
        context.build_base_mask_pipeline != nullptr
        && context.apply_blend_pipeline != nullptr
        && context.copy_clip_pipeline != nullptr
        && context.resize_image_pipeline != nullptr
        && context.resize_mask_pipeline != nullptr) {
        return;
    }

    const ncnn::Option opt = context.option();
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

    context.build_base_mask_pipeline = create_pipeline(
        kLadaBlendBaseMaskShader,
        static_cast<int>(sizeof(kLadaBlendBaseMaskShader) - 1),
        "Lada blend base mask");
    context.apply_blend_pipeline = create_pipeline(
        kLadaBlendApplyShader,
        static_cast<int>(sizeof(kLadaBlendApplyShader) - 1),
        "Lada blend apply");
    context.copy_clip_pipeline = create_pipeline(
        kLadaBlendCopyShader,
        static_cast<int>(sizeof(kLadaBlendCopyShader) - 1),
        "Lada blend copy");
    context.resize_image_pipeline = create_pipeline(
        kLadaBlendResizeImageShader,
        static_cast<int>(sizeof(kLadaBlendResizeImageShader) - 1),
        "Lada blend resize image");
    context.resize_mask_pipeline = create_pipeline(
        kLadaBlendResizeMaskShader,
        static_cast<int>(sizeof(kLadaBlendResizeMaskShader) - 1),
        "Lada blend resize mask");
}

lada::TorchConv2DLayer& get_or_create_blur_conv(
    LadaVulkanBlendContext& context,
    int kernel_size)
{
    const BlendConvKey key{kernel_size};
    const auto it = context.blur_conv_cache.find(key);
    if (it != context.blur_conv_cache.end()) {
        return *it->second;
    }

    auto layer = std::make_unique<lada::TorchConv2DLayer>();
    layer->vkdev = context.vkdev;

    ncnn::ParamDict pd;
    pd.set(0, 1);
    pd.set(1, kernel_size);
    pd.set(11, kernel_size);
    pd.set(2, 1);
    pd.set(12, 1);
    pd.set(3, 1);
    pd.set(13, 1);
    pd.set(4, kernel_size / 2);
    pd.set(14, kernel_size / 2);
    pd.set(15, kernel_size / 2);
    pd.set(16, kernel_size / 2);
    pd.set(18, 0.f);
    pd.set(5, 0);
    pd.set(6, kernel_size * kernel_size);
    pd.set(7, 1);
    pd.set(9, 0);
    pd.set(10, ncnn::Mat());
    pd.set(20, 3);
    if (layer->load_param(pd) != 0) {
        throw std::runtime_error("Failed to load torch conv2d params for blur layer.");
    }

    ncnn::Mat weights[1];
    weights[0] = create_box_blur_weight(kernel_size);
    if (layer->load_model(ncnn::ModelBinFromMatArray(weights)) != 0) {
        throw std::runtime_error("Failed to load torch conv2d weights for blur layer.");
    }

    const ncnn::Option opt = context.option();
    if (layer->create_pipeline(opt) != 0) {
        throw std::runtime_error("Failed to create torch conv2d blur pipeline.");
    }

    ncnn::VkTransfer upload_cmd(context.vkdev);
    if (layer->upload_model(upload_cmd, opt) != 0 || upload_cmd.submit_and_wait() != 0) {
        throw std::runtime_error("Failed to upload torch conv2d blur weights to Vulkan.");
    }

    auto [inserted_it, _] = context.blur_conv_cache.emplace(key, std::move(layer));
    return *inserted_it->second;
}

struct UInt8ArrayView
{
    py::array array;
    py::buffer_info info;
};

UInt8ArrayView require_uint8_array(const py::object& obj, const char* name)
{
    py::array array = py::array::ensure(obj);
    if (!array) {
        throw std::runtime_error(std::string(name) + " must be convertible to a NumPy-style array.");
    }

    const py::dtype dtype = array.dtype();
    if (!dtype.is(py::dtype::of<unsigned char>())) {
        throw std::runtime_error(std::string(name) + " must use uint8 dtype.");
    }

    return UInt8ArrayView{std::move(array), {}};
}

UInt8ArrayView require_hwc_u8_array(
    const py::object& obj,
    const char* name,
    bool require_writeable)
{
    UInt8ArrayView view = require_uint8_array(obj, name);
    view.info = view.array.request();
    if (view.info.ndim != 3 || view.info.shape[2] != 3) {
        throw std::runtime_error(std::string(name) + " must be an HWC uint8 image with 3 channels.");
    }
    if (require_writeable && view.info.readonly) {
        throw std::runtime_error(std::string(name) + " must be writeable.");
    }
    return view;
}

UInt8ArrayView require_mask_u8_array(const py::object& obj, const char* name)
{
    UInt8ArrayView view = require_uint8_array(obj, name);
    view.info = view.array.request();
    const bool valid_mask_shape =
        (view.info.ndim == 2)
        || (view.info.ndim == 3 && view.info.shape[2] == 1);
    if (!valid_mask_shape) {
        throw std::runtime_error(std::string(name) + " must have shape HW or HW1.");
    }
    return view;
}

ncnn::Mat hwc_u8_view_to_chw_float_mat(const UInt8ArrayView& view)
{
    const int height = static_cast<int>(view.info.shape[0]);
    const int width = static_cast<int>(view.info.shape[1]);
    const std::ptrdiff_t stride_y = static_cast<std::ptrdiff_t>(view.info.strides[0]);
    const std::ptrdiff_t stride_x = static_cast<std::ptrdiff_t>(view.info.strides[1]);
    const std::ptrdiff_t stride_c = static_cast<std::ptrdiff_t>(view.info.strides[2]);
    if (
        stride_c == 1
        && stride_x == 3
        && stride_y == static_cast<std::ptrdiff_t>(width) * 3) {
        return ncnn::Mat::from_pixels(
            static_cast<const unsigned char*>(view.info.ptr),
            ncnn::Mat::PIXEL_BGR,
            width,
            height);
    }

    ncnn::Mat mat(width, height, 3);
    if (mat.empty()) {
        throw std::runtime_error("Failed to allocate NCNN tensor for image upload.");
    }

    const auto* src = static_cast<const unsigned char*>(view.info.ptr);
    for (int channel = 0; channel < 3; ++channel) {
        float* dst = mat.channel(channel);
        for (int y = 0; y < height; ++y) {
            const auto* row_ptr = src + static_cast<std::ptrdiff_t>(y) * stride_y;
            for (int x = 0; x < width; ++x) {
                const auto* pixel_ptr = row_ptr + static_cast<std::ptrdiff_t>(x) * stride_x;
                dst[static_cast<std::size_t>(y) * width + x] = static_cast<float>(
                    *(pixel_ptr + static_cast<std::ptrdiff_t>(channel) * stride_c));
            }
        }
    }
    return mat;
}

ncnn::Mat mask_u8_view_to_float_mat(const UInt8ArrayView& view)
{
    const int height = static_cast<int>(view.info.shape[0]);
    const int width = static_cast<int>(view.info.shape[1]);
    const std::ptrdiff_t stride_y = static_cast<std::ptrdiff_t>(view.info.strides[0]);
    const std::ptrdiff_t stride_x = static_cast<std::ptrdiff_t>(view.info.strides[1]);

    ncnn::Mat mat(width, height, 1);
    if (mat.empty()) {
        throw std::runtime_error("Failed to allocate NCNN tensor for mask upload.");
    }

    const auto* src = static_cast<const unsigned char*>(view.info.ptr);
    float* dst = mat.channel(0);
    for (int y = 0; y < height; ++y) {
        const auto* row_ptr = src + static_cast<std::ptrdiff_t>(y) * stride_y;
        for (int x = 0; x < width; ++x) {
            const auto* pixel_ptr = row_ptr + static_cast<std::ptrdiff_t>(x) * stride_x;
            const unsigned char value = *pixel_ptr;
            dst[static_cast<std::size_t>(y) * width + x] = value > 0 ? 1.f : 0.f;
        }
    }
    return mat;
}

py::array_t<unsigned char> ncnn_mat_to_hwc_u8(const ncnn::Mat& mat)
{
    if (mat.dims != 3 || mat.c != 3 || mat.elempack != 1 || mat.elemsize != sizeof(float)) {
        throw std::runtime_error("Unexpected Vulkan blend output tensor shape or dtype.");
    }

    py::array_t<unsigned char> output({mat.h, mat.w, mat.c});
    const py::buffer_info info = output.request();
    if (
        info.ndim == 3
        && info.shape[2] == 3
        && info.strides[2] == 1
        && info.strides[1] == 3
        && info.strides[0] == static_cast<py::ssize_t>(mat.w) * 3) {
        mat.to_pixels(
            static_cast<unsigned char*>(info.ptr),
            ncnn::Mat::PIXEL_BGR,
            static_cast<int>(info.strides[0]));
        return output;
    }

    auto out = output.mutable_unchecked<3>();
    for (int channel = 0; channel < mat.c; ++channel) {
        const float* src = mat.channel(channel);
        for (int y = 0; y < mat.h; ++y) {
            for (int x = 0; x < mat.w; ++x) {
                out(y, x, channel) = static_cast<unsigned char>(std::clamp(
                    std::lround(src[static_cast<std::size_t>(y) * mat.w + x]),
                    0l,
                    255l));
            }
        }
    }
    return output;
}

void write_ncnn_mat_to_hwc_u8_view(const ncnn::Mat& mat, UInt8ArrayView& view)
{
    if (mat.dims != 3 || mat.c != 3 || mat.elempack != 1 || mat.elemsize != sizeof(float)) {
        throw std::runtime_error("Unexpected Vulkan blend output tensor shape or dtype.");
    }
    if (view.info.ndim != 3 || view.info.shape[2] != 3) {
        throw std::runtime_error("Vulkan blend output target must be an HWC uint8 image.");
    }
    if (mat.h != view.info.shape[0] || mat.w != view.info.shape[1]) {
        throw std::runtime_error("Vulkan blend output target shape mismatch.");
    }

    auto* dst = static_cast<unsigned char*>(view.info.ptr);
    const std::ptrdiff_t stride_y = static_cast<std::ptrdiff_t>(view.info.strides[0]);
    const std::ptrdiff_t stride_x = static_cast<std::ptrdiff_t>(view.info.strides[1]);
    const std::ptrdiff_t stride_c = static_cast<std::ptrdiff_t>(view.info.strides[2]);
    if (
        stride_c == 1
        && stride_x == 3
        && stride_y == static_cast<std::ptrdiff_t>(mat.w) * 3) {
        mat.to_pixels(dst, ncnn::Mat::PIXEL_BGR, static_cast<int>(stride_y));
        return;
    }

    for (int channel = 0; channel < mat.c; ++channel) {
        const float* src = mat.channel(channel);
        for (int y = 0; y < mat.h; ++y) {
            auto* row_ptr = dst + static_cast<std::ptrdiff_t>(y) * stride_y;
            for (int x = 0; x < mat.w; ++x) {
                auto* pixel_ptr = row_ptr + static_cast<std::ptrdiff_t>(x) * stride_x;
                *(pixel_ptr + static_cast<std::ptrdiff_t>(channel) * stride_c) =
                    static_cast<unsigned char>(std::clamp(
                        std::lround(src[static_cast<std::size_t>(y) * mat.w + x]),
                        0l,
                        255l));
            }
        }
    }
}

void copy_hwc_u8_view(const UInt8ArrayView& src_view, UInt8ArrayView& dst_view)
{
    if (
        src_view.info.ndim != 3
        || dst_view.info.ndim != 3
        || src_view.info.shape[0] != dst_view.info.shape[0]
        || src_view.info.shape[1] != dst_view.info.shape[1]
        || src_view.info.shape[2] != 3
        || dst_view.info.shape[2] != 3) {
        throw std::runtime_error("HWC uint8 copy shape mismatch.");
    }

    const auto* src = static_cast<const unsigned char*>(src_view.info.ptr);
    auto* dst = static_cast<unsigned char*>(dst_view.info.ptr);
    const std::ptrdiff_t src_stride_y = static_cast<std::ptrdiff_t>(src_view.info.strides[0]);
    const std::ptrdiff_t src_stride_x = static_cast<std::ptrdiff_t>(src_view.info.strides[1]);
    const std::ptrdiff_t src_stride_c = static_cast<std::ptrdiff_t>(src_view.info.strides[2]);
    const std::ptrdiff_t dst_stride_y = static_cast<std::ptrdiff_t>(dst_view.info.strides[0]);
    const std::ptrdiff_t dst_stride_x = static_cast<std::ptrdiff_t>(dst_view.info.strides[1]);
    const std::ptrdiff_t dst_stride_c = static_cast<std::ptrdiff_t>(dst_view.info.strides[2]);

    const int height = static_cast<int>(src_view.info.shape[0]);
    const int width = static_cast<int>(src_view.info.shape[1]);
    for (int y = 0; y < height; ++y) {
        const auto* src_row = src + static_cast<std::ptrdiff_t>(y) * src_stride_y;
        auto* dst_row = dst + static_cast<std::ptrdiff_t>(y) * dst_stride_y;
        for (int x = 0; x < width; ++x) {
            const auto* src_pixel = src_row + static_cast<std::ptrdiff_t>(x) * src_stride_x;
            auto* dst_pixel = dst_row + static_cast<std::ptrdiff_t>(x) * dst_stride_x;
            for (int channel = 0; channel < 3; ++channel) {
                *(dst_pixel + static_cast<std::ptrdiff_t>(channel) * dst_stride_c) =
                    *(src_pixel + static_cast<std::ptrdiff_t>(channel) * src_stride_c);
            }
        }
    }
}

ncnn::VkMat upload_to_gpu(
    const ncnn::Mat& mat,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt);

struct BlendPatchViews
{
    UInt8ArrayView frame_roi;
    UInt8ArrayView clip_img;
    UInt8ArrayView clip_mask;
    int height = 0;
    int width = 0;
    int inner_top = 0;
    int inner_bottom = 0;
    int inner_left = 0;
    int inner_right = 0;
    int border_size = 0;
    int crop_top = 0;
    int crop_bottom = 0;
    int crop_left = 0;
    int crop_right = 0;
    int crop_height = 0;
    int crop_width = 0;
    bool preprocess_clip = false;
};

constexpr float kBlendMaskBoxBorderRatio = 0.05f;

std::size_t hwc_offset(int y, int x, int width, int channels, int channel = 0)
{
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x))
        * static_cast<std::size_t>(channels)
        + static_cast<std::size_t>(channel);
}

unsigned char read_hwc_u8_value(const UInt8ArrayView& view, int y, int x, int channel)
{
    const auto* base = static_cast<const unsigned char*>(view.info.ptr);
    const std::ptrdiff_t stride_y = static_cast<std::ptrdiff_t>(view.info.strides[0]);
    const std::ptrdiff_t stride_x = static_cast<std::ptrdiff_t>(view.info.strides[1]);
    const std::ptrdiff_t stride_c = static_cast<std::ptrdiff_t>(view.info.strides[2]);
    const auto* row_ptr = base + static_cast<std::ptrdiff_t>(y) * stride_y;
    const auto* pixel_ptr = row_ptr + static_cast<std::ptrdiff_t>(x) * stride_x;
    return *(pixel_ptr + static_cast<std::ptrdiff_t>(channel) * stride_c);
}

unsigned char read_mask_u8_value(const UInt8ArrayView& view, int y, int x)
{
    const auto* base = static_cast<const unsigned char*>(view.info.ptr);
    const std::ptrdiff_t stride_y = static_cast<std::ptrdiff_t>(view.info.strides[0]);
    const std::ptrdiff_t stride_x = static_cast<std::ptrdiff_t>(view.info.strides[1]);
    const auto* row_ptr = base + static_cast<std::ptrdiff_t>(y) * stride_y;
    const auto* pixel_ptr = row_ptr + static_cast<std::ptrdiff_t>(x) * stride_x;
    return *pixel_ptr;
}

void write_hwc_u8_value(UInt8ArrayView& view, int y, int x, int channel, unsigned char value)
{
    auto* base = static_cast<unsigned char*>(view.info.ptr);
    const std::ptrdiff_t stride_y = static_cast<std::ptrdiff_t>(view.info.strides[0]);
    const std::ptrdiff_t stride_x = static_cast<std::ptrdiff_t>(view.info.strides[1]);
    const std::ptrdiff_t stride_c = static_cast<std::ptrdiff_t>(view.info.strides[2]);
    auto* row_ptr = base + static_cast<std::ptrdiff_t>(y) * stride_y;
    auto* pixel_ptr = row_ptr + static_cast<std::ptrdiff_t>(x) * stride_x;
    *(pixel_ptr + static_cast<std::ptrdiff_t>(channel) * stride_c) = value;
}

int reflect_border_index(int index, int size)
{
    if (size <= 1) {
        return 0;
    }
    while (index < 0 || index >= size) {
        if (index < 0) {
            index = -index - 1;
        }
        else {
            index = size * 2 - index - 1;
        }
    }
    return index;
}

struct PreparedBlendInputs
{
    std::vector<unsigned char> clip_img;
    std::vector<unsigned char> clip_mask;
    int height = 0;
    int width = 0;
};

struct BlendMaskGeometry
{
    int inner_top = 0;
    int inner_bottom = 0;
    int inner_left = 0;
    int inner_right = 0;
    int border_size = 0;
};

std::vector<unsigned char> resize_clip_image_linear_cpu(
    const UInt8ArrayView& view,
    int src_top,
    int src_left,
    int src_height,
    int src_width,
    int dst_height,
    int dst_width)
{
    std::vector<unsigned char> output(
        static_cast<std::size_t>(dst_height) * static_cast<std::size_t>(dst_width) * 3u);
    if (dst_height <= 0 || dst_width <= 0) {
        return output;
    }

    if (src_height == dst_height && src_width == dst_width) {
        for (int y = 0; y < dst_height; ++y) {
            for (int x = 0; x < dst_width; ++x) {
                for (int channel = 0; channel < 3; ++channel) {
                    output[hwc_offset(y, x, dst_width, 3, channel)] =
                        read_hwc_u8_value(view, src_top + y, src_left + x, channel);
                }
            }
        }
        return output;
    }

    for (int y = 0; y < dst_height; ++y) {
        const float src_y = std::clamp(
            (static_cast<float>(y) + 0.5f) * static_cast<float>(src_height) / static_cast<float>(dst_height) - 0.5f,
            0.0f,
            static_cast<float>(std::max(src_height - 1, 0)));
        const int y0 = std::min(static_cast<int>(std::floor(src_y)), src_height - 1);
        const int y1 = std::min(y0 + 1, src_height - 1);
        const float ly = src_y - static_cast<float>(y0);
        const float hy = 1.0f - ly;

        for (int x = 0; x < dst_width; ++x) {
            const float src_x = std::clamp(
                (static_cast<float>(x) + 0.5f) * static_cast<float>(src_width) / static_cast<float>(dst_width) - 0.5f,
                0.0f,
                static_cast<float>(std::max(src_width - 1, 0)));
            const int x0 = std::min(static_cast<int>(std::floor(src_x)), src_width - 1);
            const int x1 = std::min(x0 + 1, src_width - 1);
            const float lx = src_x - static_cast<float>(x0);
            const float hx = 1.0f - lx;

            for (int channel = 0; channel < 3; ++channel) {
                const float v00 = static_cast<float>(
                    read_hwc_u8_value(view, src_top + y0, src_left + x0, channel));
                const float v01 = static_cast<float>(
                    read_hwc_u8_value(view, src_top + y0, src_left + x1, channel));
                const float v10 = static_cast<float>(
                    read_hwc_u8_value(view, src_top + y1, src_left + x0, channel));
                const float v11 = static_cast<float>(
                    read_hwc_u8_value(view, src_top + y1, src_left + x1, channel));
                const float top = v00 * hx + v01 * lx;
                const float bottom = v10 * hx + v11 * lx;
                output[hwc_offset(y, x, dst_width, 3, channel)] = static_cast<unsigned char>(std::clamp(
                    std::lround(top * hy + bottom * ly),
                    0l,
                    255l));
            }
        }
    }
    return output;
}

py::module_& get_cv2_module()
{
    static py::module_ cv2 = py::module_::import("cv2");
    return cv2;
}

py::object get_cv2_resize_interpolation(bool linear)
{
    static py::object inter_linear = get_cv2_module().attr("INTER_LINEAR");
    static py::object inter_nearest = get_cv2_module().attr("INTER_NEAREST");
    return linear ? inter_linear : inter_nearest;
}

py::array_t<unsigned char> copy_clip_image_crop_to_numpy(
    const UInt8ArrayView& view,
    int src_top,
    int src_left,
    int src_height,
    int src_width)
{
    py::array_t<unsigned char> crop({src_height, src_width, 3});
    UInt8ArrayView crop_view{crop, crop.request()};
    for (int y = 0; y < src_height; ++y) {
        for (int x = 0; x < src_width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                write_hwc_u8_value(
                    crop_view,
                    y,
                    x,
                    channel,
                    read_hwc_u8_value(view, src_top + y, src_left + x, channel));
            }
        }
    }
    return crop;
}

py::array_t<unsigned char> copy_clip_mask_crop_to_numpy(
    const UInt8ArrayView& view,
    int src_top,
    int src_left,
    int src_height,
    int src_width)
{
    py::array_t<unsigned char> crop({src_height, src_width});
    auto* dst = static_cast<unsigned char*>(crop.request().ptr);
    for (int y = 0; y < src_height; ++y) {
        for (int x = 0; x < src_width; ++x) {
            dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(src_width) + static_cast<std::size_t>(x)] =
                read_mask_u8_value(view, src_top + y, src_left + x);
        }
    }
    return crop;
}

py::object resize_with_cv2(
    const py::array_t<unsigned char>& crop,
    int dst_height,
    int dst_width,
    bool linear)
{
    py::tuple args(6);
    args[0] = crop;
    args[1] = py::make_tuple(dst_width, dst_height);
    args[2] = py::none();
    args[3] = py::float_(0.0);
    args[4] = py::float_(0.0);
    args[5] = get_cv2_resize_interpolation(linear);
    return get_cv2_module().attr("resize")(*args);
}

std::vector<unsigned char> resize_clip_image_linear_cv2(
    const UInt8ArrayView& view,
    int src_top,
    int src_left,
    int src_height,
    int src_width,
    int dst_height,
    int dst_width)
{
    py::gil_scoped_acquire acquire;
    py::object resized_obj = resize_with_cv2(
        copy_clip_image_crop_to_numpy(view, src_top, src_left, src_height, src_width),
        dst_height,
        dst_width,
        true);
    UInt8ArrayView resized = require_hwc_u8_array(resized_obj, "cv2.resize image output", false);
    std::vector<unsigned char> output(
        static_cast<std::size_t>(dst_height) * static_cast<std::size_t>(dst_width) * 3u);
    for (int y = 0; y < dst_height; ++y) {
        for (int x = 0; x < dst_width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                output[hwc_offset(y, x, dst_width, 3, channel)] =
                    read_hwc_u8_value(resized, y, x, channel);
            }
        }
    }
    return output;
}

std::vector<unsigned char> resize_clip_mask_nearest_cpu(
    const UInt8ArrayView& view,
    int src_top,
    int src_left,
    int src_height,
    int src_width,
    int dst_height,
    int dst_width)
{
    std::vector<unsigned char> output(
        static_cast<std::size_t>(dst_height) * static_cast<std::size_t>(dst_width));
    if (dst_height <= 0 || dst_width <= 0) {
        return output;
    }

    if (src_height == dst_height && src_width == dst_width) {
        for (int y = 0; y < dst_height; ++y) {
            for (int x = 0; x < dst_width; ++x) {
                output[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_width) + static_cast<std::size_t>(x)] =
                    read_mask_u8_value(view, src_top + y, src_left + x);
            }
        }
        return output;
    }

    for (int y = 0; y < dst_height; ++y) {
        const int src_local_y = std::min(
            static_cast<int>(
                (static_cast<std::int64_t>(y) * static_cast<std::int64_t>(src_height)) / static_cast<std::int64_t>(dst_height)),
            src_height - 1);
        for (int x = 0; x < dst_width; ++x) {
            const int src_local_x = std::min(
                static_cast<int>(
                    (static_cast<std::int64_t>(x) * static_cast<std::int64_t>(src_width)) / static_cast<std::int64_t>(dst_width)),
                src_width - 1);
            output[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_width) + static_cast<std::size_t>(x)] =
                read_mask_u8_value(view, src_top + src_local_y, src_left + src_local_x);
        }
    }
    return output;
}

std::vector<unsigned char> resize_clip_mask_nearest_cv2(
    const UInt8ArrayView& view,
    int src_top,
    int src_left,
    int src_height,
    int src_width,
    int dst_height,
    int dst_width)
{
    py::gil_scoped_acquire acquire;
    py::object resized_obj = resize_with_cv2(
        copy_clip_mask_crop_to_numpy(view, src_top, src_left, src_height, src_width),
        dst_height,
        dst_width,
        false);
    UInt8ArrayView resized = require_mask_u8_array(resized_obj, "cv2.resize mask output");
    std::vector<unsigned char> output(
        static_cast<std::size_t>(dst_height) * static_cast<std::size_t>(dst_width));
    for (int y = 0; y < dst_height; ++y) {
        for (int x = 0; x < dst_width; ++x) {
            output[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_width) + static_cast<std::size_t>(x)] =
                read_mask_u8_value(resized, y, x);
        }
    }
    return output;
}

PreparedBlendInputs prepare_blend_inputs_cpu(const BlendPatchViews& views)
{
    const int clip_height = static_cast<int>(views.clip_img.info.shape[0]);
    const int clip_width = static_cast<int>(views.clip_img.info.shape[1]);
    const int src_top = views.preprocess_clip ? views.crop_top : 0;
    const int src_left = views.preprocess_clip ? views.crop_left : 0;
    const int src_height = views.preprocess_clip ? views.crop_height : clip_height;
    const int src_width = views.preprocess_clip ? views.crop_width : clip_width;

    PreparedBlendInputs prepared;
    prepared.height = views.height;
    prepared.width = views.width;
    if (views.preprocess_clip) {
        prepared.clip_img = resize_clip_image_linear_cv2(
            views.clip_img,
            src_top,
            src_left,
            src_height,
            src_width,
            views.height,
            views.width);
        prepared.clip_mask = resize_clip_mask_nearest_cv2(
            views.clip_mask,
            src_top,
            src_left,
            src_height,
            src_width,
            views.height,
            views.width);
    }
    else {
        prepared.clip_img = resize_clip_image_linear_cpu(
            views.clip_img,
            src_top,
            src_left,
            src_height,
            src_width,
            views.height,
            views.width);
        prepared.clip_mask = resize_clip_mask_nearest_cpu(
            views.clip_mask,
            src_top,
            src_left,
            src_height,
            src_width,
            views.height,
            views.width);
    }
    return prepared;
}

BlendMaskGeometry resolve_blend_geometry_from_mask_buffer(
    const std::vector<unsigned char>& mask,
    int height,
    int width)
{
    BlendMaskGeometry geometry{};
    int top = height;
    int bottom = -1;
    int left = width;
    int right = -1;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (mask[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] == 0) {
                continue;
            }
            top = std::min(top, y);
            bottom = std::max(bottom, y);
            left = std::min(left, x);
            right = std::max(right, x);
        }
    }

    if (bottom < top || right < left) {
        return geometry;
    }

    const int box_height = std::max(bottom - top + 1, 1);
    const int box_width = std::max(right - left + 1, 1);
    const int border_y = std::max(1, static_cast<int>(std::round(static_cast<float>(box_height) * kBlendMaskBoxBorderRatio)));
    const int border_x = std::max(1, static_cast<int>(std::round(static_cast<float>(box_width) * kBlendMaskBoxBorderRatio)));

    geometry.inner_top = std::max(0, top - border_y);
    geometry.inner_bottom = std::min(height, bottom + border_y + 1);
    geometry.inner_left = std::max(0, left - border_x);
    geometry.inner_right = std::min(width, right + border_x + 1);

    const int border_height = (top - geometry.inner_top) + (geometry.inner_bottom - (bottom + 1));
    const int border_width = (left - geometry.inner_left) + (geometry.inner_right - (right + 1));
    geometry.border_size = std::min(border_height, border_width);
    return geometry;
}

std::vector<float> box_blur_reflect_cpu(
    const std::vector<float>& input,
    int height,
    int width,
    int kernel_size)
{
    const int radius = kernel_size / 2;
    std::vector<float> horizontal(
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width),
        0.0f);
    std::vector<float> output(
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width),
        0.0f);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int offset = -radius; offset <= radius; ++offset) {
                const int src_x = reflect_border_index(x + offset, width);
                sum += input[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(src_x)];
            }
            horizontal[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
                static_cast<float>(sum);
        }
    }

    const double normalization = 1.0 / static_cast<double>(kernel_size * kernel_size);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int offset = -radius; offset <= radius; ++offset) {
                const int src_y = reflect_border_index(y + offset, height);
                sum += horizontal[static_cast<std::size_t>(src_y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)];
            }
            output[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
                static_cast<float>(sum * normalization);
        }
    }
    return output;
}

std::vector<float> create_blend_mask_cpu(
    const std::vector<unsigned char>& clip_mask,
    int height,
    int width)
{
    std::vector<float> blend_mask(
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width),
        0.0f);
    const BlendMaskGeometry geometry = resolve_blend_geometry_from_mask_buffer(clip_mask, height, width);
    if (geometry.border_size == 0) {
        return blend_mask;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const bool inner = (
                y >= geometry.inner_top
                && y < geometry.inner_bottom
                && x >= geometry.inner_left
                && x < geometry.inner_right);
            const bool masked =
                clip_mask[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] > 0;
            blend_mask[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
                (inner || masked) ? 1.0f : 0.0f;
        }
    }

    int blur_size = geometry.border_size;
    if ((blur_size & 1) == 0) {
        blur_size += 1;
    }
    if (blur_size >= 5) {
        return box_blur_reflect_cpu(blend_mask, height, width, blur_size);
    }
    return blend_mask;
}

void apply_reference_blend(
    const BlendPatchViews& views,
    UInt8ArrayView& output_view)
{
    const PreparedBlendInputs prepared = prepare_blend_inputs_cpu(views);
    const std::vector<float> blend_mask = create_blend_mask_cpu(
        prepared.clip_mask,
        prepared.height,
        prepared.width);

    for (int y = 0; y < prepared.height; ++y) {
        for (int x = 0; x < prepared.width; ++x) {
            const float mask_value =
                blend_mask[static_cast<std::size_t>(y) * static_cast<std::size_t>(prepared.width) + static_cast<std::size_t>(x)];
            for (int channel = 0; channel < 3; ++channel) {
                const float frame_value = static_cast<float>(
                    read_hwc_u8_value(views.frame_roi, y, x, channel));
                const float clip_value = static_cast<float>(
                    prepared.clip_img[hwc_offset(y, x, prepared.width, 3, channel)]);
                const float blended = (clip_value - frame_value) * mask_value + frame_value;
                write_hwc_u8_value(
                    output_view,
                    y,
                    x,
                    channel,
                    static_cast<unsigned char>(std::clamp(blended, 0.0f, 255.0f)));
            }
        }
    }
}

void populate_blend_geometry(
    int height,
    int width,
    int& inner_top,
    int& inner_bottom,
    int& inner_left,
    int& inner_right,
    int& border_size)
{
    const int h_inner = static_cast<int>(height * (1.0f - 0.05f));
    const int w_inner = static_cast<int>(width * (1.0f - 0.05f));
    const int h_outer = height - h_inner;
    const int w_outer = width - w_inner;
    border_size = std::min(h_outer, w_outer);
    inner_top = h_outer / 2;
    inner_bottom = h_outer - inner_top;
    inner_left = w_outer / 2;
    inner_right = w_outer - inner_left;
}

void populate_blend_geometry_from_mask(
    const UInt8ArrayView& clip_mask,
    int& inner_top,
    int& inner_bottom,
    int& inner_left,
    int& inner_right,
    int& border_size)
{
    const int height = static_cast<int>(clip_mask.info.shape[0]);
    const int width = static_cast<int>(clip_mask.info.shape[1]);
    const auto* src = static_cast<const unsigned char*>(clip_mask.info.ptr);
    const std::ptrdiff_t stride_y = static_cast<std::ptrdiff_t>(clip_mask.info.strides[0]);
    const std::ptrdiff_t stride_x = static_cast<std::ptrdiff_t>(clip_mask.info.strides[1]);

    int mask_top = height;
    int mask_bottom = -1;
    int mask_left = width;
    int mask_right = -1;
    for (int y = 0; y < height; ++y) {
        const auto* row_ptr = src + static_cast<std::ptrdiff_t>(y) * stride_y;
        for (int x = 0; x < width; ++x) {
            const auto* pixel_ptr = row_ptr + static_cast<std::ptrdiff_t>(x) * stride_x;
            if (*pixel_ptr == 0) {
                continue;
            }
            mask_top = std::min(mask_top, y);
            mask_bottom = std::max(mask_bottom, y);
            mask_left = std::min(mask_left, x);
            mask_right = std::max(mask_right, x);
        }
    }

    if (mask_bottom < mask_top || mask_right < mask_left) {
        inner_top = height;
        inner_bottom = 0;
        inner_left = width;
        inner_right = 0;
        border_size = 0;
        return;
    }

    const int mask_height = mask_bottom - mask_top + 1;
    const int mask_width = mask_right - mask_left + 1;
    const int border_y = std::max(1, static_cast<int>(std::round(mask_height * kBlendMaskBoxBorderRatio)));
    const int border_x = std::max(1, static_cast<int>(std::round(mask_width * kBlendMaskBoxBorderRatio)));

    const int inner_rect_top = std::max(0, mask_top - border_y);
    const int inner_rect_bottom = std::min(height, mask_bottom + border_y + 1);
    const int inner_rect_left = std::max(0, mask_left - border_x);
    const int inner_rect_right = std::min(width, mask_right + border_x + 1);

    inner_top = inner_rect_top;
    inner_bottom = height - inner_rect_bottom;
    inner_left = inner_rect_left;
    inner_right = width - inner_rect_right;
    const int border_height =
        (mask_top - inner_rect_top) + (inner_rect_bottom - (mask_bottom + 1));
    const int border_width =
        (mask_left - inner_rect_left) + (inner_rect_right - (mask_right + 1));
    border_size = std::min(border_height, border_width);
}

void validate_clip_mask_pair(const UInt8ArrayView& clip_img, const UInt8ArrayView& clip_mask)
{
    const bool valid_mask_shape =
        clip_mask.info.shape[0] == clip_img.info.shape[0]
        && clip_mask.info.shape[1] == clip_img.info.shape[1];
    if (!valid_mask_shape) {
        throw std::runtime_error("clip_mask must match clip_img height and width.");
    }
}

ncnn::Mat download_vkmat(
    ncnn::VkCompute& cmd,
    const ncnn::VkMat& blob,
    const ncnn::Option& opt,
    const char* error_message)
{
    ncnn::Mat output_mat;
    cmd.record_download(blob, output_mat, opt);
    if (cmd.submit_and_wait() != 0) {
        throw std::runtime_error(error_message);
    }
    return output_mat;
}

void submit_and_wait(
    ncnn::VkCompute& cmd,
    const char* error_message)
{
    if (cmd.submit_and_wait() != 0) {
        throw std::runtime_error(error_message);
    }
}

ncnn::VkMat resize_clip_image_on_gpu(
    LadaVulkanBlendContext& context,
    const ncnn::VkMat& clip_vk,
    const BlendPatchViews& views,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt)
{
    ncnn::VkMat output_vk;
    output_vk.create(views.width, views.height, 3, static_cast<size_t>(4u), 1, opt.blob_vkallocator);
    if (output_vk.empty()) {
        throw std::runtime_error("Failed to allocate Vulkan tensor for resized clip image.");
    }

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = clip_vk;
    bindings[1] = output_vk;
    std::vector<ncnn::vk_constant_type> constants(10);
    constants[0].i = static_cast<int>(views.clip_img.info.shape[1]);
    constants[1].i = static_cast<int>(views.clip_img.info.shape[0]);
    constants[2].i = views.width;
    constants[3].i = views.height;
    constants[4].i = views.crop_left;
    constants[5].i = views.crop_top;
    constants[6].i = views.crop_width;
    constants[7].i = views.crop_height;
    constants[8].i = static_cast<int>(clip_vk.cstep);
    constants[9].i = static_cast<int>(output_vk.cstep);
    cmd.record_pipeline(context.resize_image_pipeline, bindings, constants, output_vk);
    return output_vk;
}

ncnn::VkMat resize_clip_mask_on_gpu(
    LadaVulkanBlendContext& context,
    const ncnn::VkMat& mask_vk,
    const BlendPatchViews& views,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt)
{
    ncnn::VkMat output_vk;
    output_vk.create(views.width, views.height, 1, static_cast<size_t>(4u), 1, opt.blob_vkallocator);
    if (output_vk.empty()) {
        throw std::runtime_error("Failed to allocate Vulkan tensor for resized clip mask.");
    }

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = mask_vk;
    bindings[1] = output_vk;
    std::vector<ncnn::vk_constant_type> constants(10);
    constants[0].i = static_cast<int>(views.clip_mask.info.shape[1]);
    constants[1].i = static_cast<int>(views.clip_mask.info.shape[0]);
    constants[2].i = views.width;
    constants[3].i = views.height;
    constants[4].i = views.crop_left;
    constants[5].i = views.crop_top;
    constants[6].i = views.crop_width;
    constants[7].i = views.crop_height;
    constants[8].i = static_cast<int>(mask_vk.cstep);
    constants[9].i = static_cast<int>(output_vk.cstep);
    cmd.record_pipeline(context.resize_mask_pipeline, bindings, constants, output_vk);
    return output_vk;
}

ncnn::VkMat record_blend_resized_patches_on_gpu(
    LadaVulkanBlendContext& context,
    const ncnn::VkMat& frame_vk,
    const ncnn::VkMat& clip_vk,
    const ncnn::VkMat& mask_vk,
    int width,
    int height,
    int inner_top,
    int inner_bottom,
    int inner_left,
    int inner_right,
    int border_size,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt)
{
    ncnn::VkMat base_mask_vk;
    base_mask_vk.create(width, height, 1, static_cast<size_t>(4u), 1, opt.workspace_vkallocator);
    if (base_mask_vk.empty()) {
        throw std::runtime_error("Failed to allocate Vulkan tensor for base mask.");
    }

    {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = mask_vk;
        bindings[1] = base_mask_vk;
        std::vector<ncnn::vk_constant_type> constants(8);
        constants[0].i = width;
        constants[1].i = height;
        constants[2].i = inner_top;
        constants[3].i = inner_bottom;
        constants[4].i = inner_left;
        constants[5].i = inner_right;
        constants[6].i = static_cast<int>(mask_vk.cstep);
        constants[7].i = static_cast<int>(base_mask_vk.cstep);
        cmd.record_pipeline(context.build_base_mask_pipeline, bindings, constants, base_mask_vk);
    }

    ncnn::VkMat blend_mask_vk;
    if (border_size >= 5) {
        int blur_size = border_size;
        if (blur_size % 2 == 0) {
            blur_size += 1;
        }
        lada::TorchConv2DLayer& blur_conv = get_or_create_blur_conv(context, blur_size);
        std::vector<ncnn::VkMat> inputs(1);
        inputs[0] = base_mask_vk;
        std::vector<ncnn::VkMat> outputs(1);
        if (blur_conv.forward(inputs, outputs, cmd, opt) != 0 || outputs[0].empty()) {
            throw std::runtime_error("Failed to run torch conv2d blur on Vulkan.");
        }
        blend_mask_vk = outputs[0];
    }
    else {
        blend_mask_vk = base_mask_vk;
    }

    ncnn::VkMat output_vk;
    output_vk.create(width, height, 3, static_cast<size_t>(4u), 1, opt.blob_vkallocator);
    if (output_vk.empty()) {
        throw std::runtime_error("Failed to allocate Vulkan output tensor for blended ROI.");
    }

    {
        std::vector<ncnn::VkMat> bindings(4);
        bindings[0] = blend_mask_vk;
        bindings[1] = frame_vk;
        bindings[2] = clip_vk;
        bindings[3] = output_vk;
        std::vector<ncnn::vk_constant_type> constants(7);
        constants[0].i = width;
        constants[1].i = height;
        constants[2].i = 3;
        constants[3].i = static_cast<int>(blend_mask_vk.cstep);
        constants[4].i = static_cast<int>(frame_vk.cstep);
        constants[5].i = static_cast<int>(clip_vk.cstep);
        constants[6].i = static_cast<int>(output_vk.cstep);
        cmd.record_pipeline(context.apply_blend_pipeline, bindings, constants, output_vk);
    }

    return output_vk;
}

ncnn::Mat blend_resized_patches_on_gpu(
    LadaVulkanBlendContext& context,
    const ncnn::VkMat& frame_vk,
    const ncnn::VkMat& clip_vk,
    const ncnn::VkMat& mask_vk,
    int width,
    int height,
    int inner_top,
    int inner_bottom,
    int inner_left,
    int inner_right,
    int border_size,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt)
{
    const ncnn::VkMat output_vk = record_blend_resized_patches_on_gpu(
        context,
        frame_vk,
        clip_vk,
        mask_vk,
        width,
        height,
        inner_top,
        inner_bottom,
        inner_left,
        inner_right,
        border_size,
        cmd,
        opt);
    return download_vkmat(cmd, output_vk, opt, "Failed to execute Vulkan blend pipeline.");
}

ncnn::Mat run_blend_patch_gpu(const BlendPatchViews& views)
{
    LadaVulkanBlendContext& context = get_blend_context();
    ensure_shader_pipelines(context);
    const ncnn::Option opt = context.option();

    const ncnn::Mat frame_mat = hwc_u8_view_to_chw_float_mat(views.frame_roi);
    const ncnn::Mat clip_mat = hwc_u8_view_to_chw_float_mat(views.clip_img);
    const ncnn::Mat mask_mat = mask_u8_view_to_float_mat(views.clip_mask);

    ncnn::VkCompute cmd(context.vkdev);
    const ncnn::VkMat frame_vk = upload_to_gpu(frame_mat, cmd, opt);
    ncnn::VkMat clip_vk = upload_to_gpu(clip_mat, cmd, opt);
    ncnn::VkMat mask_vk = upload_to_gpu(mask_mat, cmd, opt);
    if (views.preprocess_clip) {
        clip_vk = resize_clip_image_on_gpu(context, clip_vk, views, cmd, opt);
        mask_vk = resize_clip_mask_on_gpu(context, mask_vk, views, cmd, opt);
    }
    return blend_resized_patches_on_gpu(
        context,
        frame_vk,
        clip_vk,
        mask_vk,
        views.width,
        views.height,
        views.inner_top,
        views.inner_bottom,
        views.inner_left,
        views.inner_right,
        views.border_size,
        cmd,
        opt);
}

void run_blend_patch_gpu_inplace_batch(std::vector<BlendPatchViews>& views_batch)
{
    if (views_batch.empty()) {
        return;
    }
    for (BlendPatchViews& views : views_batch) {
        apply_reference_blend(views, views.frame_roi);
    }
}

BlendPatchViews prepare_blend_patch_views(
    const py::object& frame_roi_obj,
    const py::object& clip_img_obj,
    const py::object& clip_mask_obj,
    bool require_writeable_frame)
{
    BlendPatchViews views{
        require_hwc_u8_array(frame_roi_obj, "frame_roi", require_writeable_frame),
        require_hwc_u8_array(clip_img_obj, "clip_img", false),
        require_mask_u8_array(clip_mask_obj, "clip_mask"),
    };

    if (
        views.frame_roi.info.shape[0] != views.clip_img.info.shape[0]
        || views.frame_roi.info.shape[1] != views.clip_img.info.shape[1]) {
        throw std::runtime_error("frame_roi and clip_img must have the same height and width.");
    }
    const bool valid_mask_shape =
        views.clip_mask.info.shape[0] == views.frame_roi.info.shape[0]
        && views.clip_mask.info.shape[1] == views.frame_roi.info.shape[1];
    if (!valid_mask_shape) {
        throw std::runtime_error("clip_mask must match frame_roi height and width.");
    }

    views.height = static_cast<int>(views.frame_roi.info.shape[0]);
    views.width = static_cast<int>(views.frame_roi.info.shape[1]);
    return views;
}

BlendPatchViews prepare_blend_patch_preprocess_views(
    const py::object& frame_roi_obj,
    const py::object& clip_img_obj,
    const py::object& clip_mask_obj,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    bool require_writeable_frame)
{
    BlendPatchViews views{
        require_hwc_u8_array(frame_roi_obj, "frame_roi", require_writeable_frame),
        require_hwc_u8_array(clip_img_obj, "clip_img", false),
        require_mask_u8_array(clip_mask_obj, "clip_mask"),
    };

    validate_clip_mask_pair(views.clip_img, views.clip_mask);
    if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0) {
        throw std::runtime_error("Blend pad values must be non-negative.");
    }

    const int clip_height = static_cast<int>(views.clip_img.info.shape[0]);
    const int clip_width = static_cast<int>(views.clip_img.info.shape[1]);
    const int crop_height = clip_height - pad_top - pad_bottom;
    const int crop_width = clip_width - pad_left - pad_right;
    if (crop_height <= 0 || crop_width <= 0) {
        throw std::runtime_error("Blend pad removes the entire clip image.");
    }

    views.height = static_cast<int>(views.frame_roi.info.shape[0]);
    views.width = static_cast<int>(views.frame_roi.info.shape[1]);
    views.crop_top = pad_top;
    views.crop_bottom = pad_bottom;
    views.crop_left = pad_left;
    views.crop_right = pad_right;
    views.crop_height = crop_height;
    views.crop_width = crop_width;
    views.preprocess_clip = true;
    return views;
}

ncnn::VkMat upload_to_gpu(
    const ncnn::Mat& mat,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt)
{
    ncnn::VkMat gpu_mat;
    if (mat.dims == 3) {
        gpu_mat.create(mat.w, mat.h, mat.c, mat.elemsize, 1, opt.blob_vkallocator);
    } else if (mat.dims == 2) {
        gpu_mat.create(mat.w, mat.h, mat.elemsize, 1, opt.blob_vkallocator);
    } else {
        throw std::runtime_error("Unsupported NCNN tensor rank for Vulkan upload.");
    }
    if (gpu_mat.empty()) {
        throw std::runtime_error("Failed to allocate Vulkan tensor for upload.");
    }
    cmd.record_upload(mat, gpu_mat, opt);
    return gpu_mat;
}

py::array_t<unsigned char> blend_patch_gpu_py(
    const py::object& frame_roi_obj,
    const py::object& clip_img_obj,
    const py::object& clip_mask_obj)
{
    BlendPatchViews views = prepare_blend_patch_views(frame_roi_obj, clip_img_obj, clip_mask_obj, false);
    py::array_t<unsigned char> output({views.height, views.width, 3});
    UInt8ArrayView output_view{output, output.request()};
    {
        py::gil_scoped_release release;
        apply_reference_blend(views, output_view);
    }
    return output;
}

void blend_patch_gpu_inplace_py(
    const py::object& frame_roi_obj,
    const py::object& clip_img_obj,
    const py::object& clip_mask_obj)
{
    BlendPatchViews views = prepare_blend_patch_views(frame_roi_obj, clip_img_obj, clip_mask_obj, true);
    {
        py::gil_scoped_release release;
        apply_reference_blend(views, views.frame_roi);
    }
}

py::array_t<unsigned char> blend_patch_gpu_preprocess_py(
    const py::object& frame_roi_obj,
    const py::object& clip_img_obj,
    const py::object& clip_mask_obj,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right)
{
    BlendPatchViews views = prepare_blend_patch_preprocess_views(
        frame_roi_obj,
        clip_img_obj,
        clip_mask_obj,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        false);
    py::array_t<unsigned char> output({views.height, views.width, 3});
    UInt8ArrayView output_view{output, output.request()};
    {
        py::gil_scoped_release release;
        apply_reference_blend(views, output_view);
    }
    return output;
}

void blend_patch_gpu_preprocess_inplace_py(
    const py::object& frame_roi_obj,
    const py::object& clip_img_obj,
    const py::object& clip_mask_obj,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right)
{
    BlendPatchViews views = prepare_blend_patch_preprocess_views(
        frame_roi_obj,
        clip_img_obj,
        clip_mask_obj,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        true);
    {
        py::gil_scoped_release release;
        apply_reference_blend(views, views.frame_roi);
    }
}

void blend_patch_gpu_preprocess_inplace_batch_py(
    const py::list& frame_rois,
    const py::list& clip_imgs,
    const py::list& clip_masks,
    const py::list& pads)
{
    const py::ssize_t count = py::len(frame_rois);
    if (count != py::len(clip_imgs) || count != py::len(clip_masks) || count != py::len(pads)) {
        throw std::runtime_error(
            "blend_patch_gpu_preprocess_inplace_batch expects equal-length frame/clip/mask/pad lists.");
    }

    std::vector<BlendPatchViews> views_batch;
    views_batch.reserve(static_cast<std::size_t>(count));
    for (py::ssize_t index = 0; index < count; ++index) {
        const py::sequence pad_values = py::reinterpret_borrow<py::sequence>(pads[index]);
        if (py::len(pad_values) != 4) {
            throw std::runtime_error("Each blend patch batch pad entry must contain 4 integers.");
        }
        views_batch.push_back(prepare_blend_patch_preprocess_views(
            frame_rois[index],
            clip_imgs[index],
            clip_masks[index],
            py::cast<int>(pad_values[0]),
            py::cast<int>(pad_values[1]),
            py::cast<int>(pad_values[2]),
            py::cast<int>(pad_values[3]),
            true));
    }

    {
        py::gil_scoped_release release;
        run_blend_patch_gpu_inplace_batch(views_batch);
    }
}

} // namespace
#endif

namespace lada {

void bind_vulkan_blend_runtime(py::module_& m)
{
#if NCNN_VULKAN
    m.def(
        "blend_patch_gpu",
        &blend_patch_gpu_py,
        py::arg("frame_roi"),
        py::arg("clip_img"),
        py::arg("clip_mask"),
        "Blend a resized clip patch into a frame ROI on Vulkan and return the blended ROI.");
    m.def(
        "blend_patch_gpu_inplace",
        &blend_patch_gpu_inplace_py,
        py::arg("frame_roi"),
        py::arg("clip_img"),
        py::arg("clip_mask"),
        "Blend a resized clip patch into a frame ROI on Vulkan and overwrite the input ROI.");
    m.def(
        "blend_patch_gpu_preprocess",
        &blend_patch_gpu_preprocess_py,
        py::arg("frame_roi"),
        py::arg("clip_img"),
        py::arg("clip_mask"),
        py::arg("pad_top"),
        py::arg("pad_bottom"),
        py::arg("pad_left"),
        py::arg("pad_right"),
        "Crop pad, resize, and blend a padded clip patch into a frame ROI on Vulkan.");
    m.def(
        "blend_patch_gpu_preprocess_inplace",
        &blend_patch_gpu_preprocess_inplace_py,
        py::arg("frame_roi"),
        py::arg("clip_img"),
        py::arg("clip_mask"),
        py::arg("pad_top"),
        py::arg("pad_bottom"),
        py::arg("pad_left"),
        py::arg("pad_right"),
        "Crop pad, resize, and blend a padded clip patch into a frame ROI on Vulkan in place.");
    m.def(
        "blend_patch_gpu_preprocess_inplace_batch",
        &blend_patch_gpu_preprocess_inplace_batch_py,
        py::arg("frame_rois"),
        py::arg("clip_imgs"),
        py::arg("clip_masks"),
        py::arg("pads"),
        "Crop pad, resize, and blend a batch of padded clip patches into frame ROIs on Vulkan in place.");
    m.attr("has_lada_blend_patch_vulkan") = py::bool_(true);
    m.attr("has_lada_blend_patch_vulkan_inplace") = py::bool_(true);
    m.attr("has_lada_blend_patch_vulkan_preprocess") = py::bool_(true);
    m.attr("has_lada_blend_patch_vulkan_preprocess_inplace") = py::bool_(true);
    m.attr("has_lada_blend_patch_vulkan_preprocess_inplace_batch") = py::bool_(true);
#else
    m.attr("has_lada_blend_patch_vulkan") = py::bool_(false);
    m.attr("has_lada_blend_patch_vulkan_inplace") = py::bool_(false);
    m.attr("has_lada_blend_patch_vulkan_preprocess") = py::bool_(false);
    m.attr("has_lada_blend_patch_vulkan_preprocess_inplace") = py::bool_(false);
    m.attr("has_lada_blend_patch_vulkan_preprocess_inplace_batch") = py::bool_(false);
#endif
}

} // namespace lada
