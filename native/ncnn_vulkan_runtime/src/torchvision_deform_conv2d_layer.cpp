// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#include "torchvision_deform_conv2d_layer.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include "net.h"
#include "native_op_profile.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif

namespace lada {

namespace {

static bool deformconv_debug_enabled()
{
    static const bool enabled = []() {
        const char* value = std::getenv("LADA_NCNN_DEBUG");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static void deformconv_debug_log(const char* format, ...)
{
    if (!deformconv_debug_enabled())
        return;

    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fflush(stderr);
}

static inline float bilinear_sample(const ncnn::Mat& blob, int channel, float y, float x)
{
    const int h = blob.h;
    const int w = blob.w;

    if (y <= -1.f || x <= -1.f || y >= h || x >= w)
        return 0.f;

    const int y_low = static_cast<int>(std::floor(y));
    const int x_low = static_cast<int>(std::floor(x));
    const int y_high = y_low + 1;
    const int x_high = x_low + 1;

    const float ly = y - y_low;
    const float lx = x - x_low;
    const float hy = 1.f - ly;
    const float hx = 1.f - lx;

    const float* channel_ptr = blob.channel(channel);

    const auto value_at = [&](int iy, int ix) -> float {
        if (iy < 0 || iy >= h || ix < 0 || ix >= w)
            return 0.f;
        return channel_ptr[iy * w + ix];
    };

    const float v1 = value_at(y_low, x_low);
    const float v2 = value_at(y_low, x_high);
    const float v3 = value_at(y_high, x_low);
    const float v4 = value_at(y_high, x_high);

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

#if NCNN_VULKAN
static const char kTorchVisionDeformConv2dShader[] = R"(
#version 450

layout(binding = 0) readonly buffer input_blob { sfp input_blob_data[]; };
layout(binding = 1) readonly buffer weight_blob { sfp weight_blob_data[]; };
layout(binding = 2) readonly buffer offset_blob { sfp offset_blob_data[]; };
layout(binding = 3) readonly buffer mask_blob { sfp mask_blob_data[]; };
layout(binding = 4) readonly buffer bias_blob { sfp bias_blob_data[]; };
layout(binding = 5) writeonly buffer output_blob { sfp output_blob_data[]; };

layout(push_constant) uniform parameter
{
    int input_w;
    int input_h;
    int input_c;
    int input_cstep;
    int weight_w;
    int weight_h;
    int weight_d;
    int weight_c;
    int weight_cstep;
    int offset_cstep;
    int mask_cstep;
    int output_w;
    int output_h;
    int output_c;
    int output_cstep;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int dilation_w;
    int dilation_h;
    int deform_groups;
    int use_mask;
    int use_bias;
} p;

float input_value_at(int channel, int y, int x)
{
    if (channel < 0 || channel >= p.input_c || y < 0 || y >= p.input_h || x < 0 || x >= p.input_w)
        return 0.f;

    const int index = channel * p.input_cstep + y * p.input_w + x;
    return float(buffer_ld1(input_blob_data, index));
}

float bilinear_sample_input(int channel, float y, float x)
{
    if (y <= -1.f || x <= -1.f || y >= float(p.input_h) || x >= float(p.input_w))
        return 0.f;

    const int y_low = int(floor(y));
    const int x_low = int(floor(x));
    const int y_high = y_low + 1;
    const int x_high = x_low + 1;

    const float ly = y - float(y_low);
    const float lx = x - float(x_low);
    const float hy = 1.f - ly;
    const float hx = 1.f - lx;

    const float v1 = input_value_at(channel, y_low, x_low);
    const float v2 = input_value_at(channel, y_low, x_high);
    const float v3 = input_value_at(channel, y_high, x_low);
    const float v4 = input_value_at(channel, y_high, x_high);

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

float offset_value_at(int channel, int y, int x)
{
    const int index = channel * p.offset_cstep + y * p.output_w + x;
    return float(buffer_ld1(offset_blob_data, index));
}

float mask_value_at(int channel, int y, int x)
{
    const int index = channel * p.mask_cstep + y * p.output_w + x;
    return float(buffer_ld1(mask_blob_data, index));
}

float weight_value_at(int out_channel, int in_channel, int ky, int kx)
{
    const int index =
        out_channel * p.weight_cstep + in_channel * p.weight_h * p.weight_w + ky * p.weight_w + kx;
    return float(buffer_ld1(weight_blob_data, index));
}

void main()
{
    const int ox = int(gl_GlobalInvocationID.x);
    const int oy = int(gl_GlobalInvocationID.y);
    const int oc = int(gl_GlobalInvocationID.z);

    if (ox >= p.output_w || oy >= p.output_h || oc >= p.output_c)
        return;

    float sum = p.use_bias != 0 ? float(buffer_ld1(bias_blob_data, oc)) : 0.f;
    const int base_y = oy * p.stride_h - p.pad_h;
    const int base_x = ox * p.stride_w - p.pad_w;
    const int kernel_size = p.weight_h * p.weight_w;
    const int channels_per_group = p.input_c / p.deform_groups;

    for (int dg = 0; dg < p.deform_groups; dg++)
    {
        const int channel_start = dg * channels_per_group;
        for (int ky = 0; ky < p.weight_h; ky++)
        {
            for (int kx = 0; kx < p.weight_w; kx++)
            {
                const int kernel_index = ky * p.weight_w + kx;
                const int offset_channel = (dg * kernel_size + kernel_index) * 2;
                const float offset_y = offset_value_at(offset_channel, oy, ox);
                const float offset_x = offset_value_at(offset_channel + 1, oy, ox);
                const float mask_value =
                    p.use_mask != 0 ? mask_value_at(dg * kernel_size + kernel_index, oy, ox) : 1.f;

                const float sample_y = float(base_y + ky * p.dilation_h) + offset_y;
                const float sample_x = float(base_x + kx * p.dilation_w) + offset_x;

                for (int c = 0; c < channels_per_group; c++)
                {
                    const int input_channel = channel_start + c;
                    const float value = bilinear_sample_input(input_channel, sample_y, sample_x);
                    const float weight = weight_value_at(oc, input_channel, ky, kx);
                    sum += value * mask_value * weight;
                }
            }
        }
    }

    const int output_index = oc * p.output_cstep + oy * p.output_w + ox;
    buffer_st1(output_blob_data, output_index, afp(sum));
}
)";
#endif

} // namespace

TorchVisionDeformConv2DLayer::TorchVisionDeformConv2DLayer()
{
    one_blob_only = false;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
    pipeline_deformconv = 0;
#endif
}

int TorchVisionDeformConv2DLayer::load_param(const ncnn::ParamDict& pd)
{
    stride_h = pd.get(5, 1);
    stride_w = pd.get(6, stride_h);
    pad_h = pd.get(7, 0);
    pad_w = pd.get(8, pad_h);
    dilation_h = pd.get(9, 1);
    dilation_w = pd.get(10, dilation_h);
    groups = pd.get(11, 1);
    deform_groups = pd.get(12, 1);
    use_mask = pd.get(13, 1);
    return 0;
}

int TorchVisionDeformConv2DLayer::create_pipeline(const ncnn::Option& opt)
{
#if NCNN_VULKAN
    deformconv_debug_log(
        "[layer] create_pipeline enter use_vulkan=%d groups=%d deform_groups=%d use_mask=%d\n",
        opt.use_vulkan_compute ? 1 : 0,
        groups,
        deform_groups,
        use_mask);

    if (!opt.use_vulkan_compute)
        return 0;

    if (groups != 1)
        return -100;

    std::vector<uint32_t> spirv;
    deformconv_debug_log("[layer] create_pipeline compile_spirv begin\n");
    const int retc = ncnn::compile_spirv_module(
        kTorchVisionDeformConv2dShader,
        static_cast<int>(sizeof(kTorchVisionDeformConv2dShader) - 1),
        opt,
        spirv);
    deformconv_debug_log(
        "[layer] create_pipeline compile_spirv end ret=%d words=%zu\n",
        retc,
        spirv.size());
    if (retc != 0)
        return retc;

    deformconv_debug_log("[layer] create_pipeline pipeline_ctor begin\n");
    pipeline_deformconv = new ncnn::Pipeline(vkdev);
    deformconv_debug_log("[layer] create_pipeline pipeline_ctor end\n");
    deformconv_debug_log("[layer] create_pipeline set_local_size begin\n");
    pipeline_deformconv->set_optimal_local_size_xyz(4, 4, 1);
    deformconv_debug_log("[layer] create_pipeline set_local_size end\n");
    deformconv_debug_log("[layer] create_pipeline pipeline_create begin\n");
    const int retp = pipeline_deformconv->create(
        spirv.data(),
        spirv.size() * sizeof(uint32_t),
        std::vector<ncnn::vk_specialization_type>());
    deformconv_debug_log("[layer] create_pipeline pipeline_create end ret=%d\n", retp);
    return retp;
#else
    (void)opt;
    return 0;
#endif
}

int TorchVisionDeformConv2DLayer::destroy_pipeline(const ncnn::Option& opt)
{
#if NCNN_VULKAN
    (void)opt;
    delete pipeline_deformconv;
    pipeline_deformconv = 0;
#else
    (void)opt;
#endif
    return 0;
}

int TorchVisionDeformConv2DLayer::forward(
    const std::vector<ncnn::Mat>& bottom_blobs,
    std::vector<ncnn::Mat>& top_blobs,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() < 4)
        return -100;
    if (groups != 1)
        return -100;

    const ncnn::Mat& input = bottom_blobs[0];
    const ncnn::Mat& weight = bottom_blobs[1];
    const ncnn::Mat& offset = bottom_blobs[2];
    const ncnn::Mat& mask = bottom_blobs[3];
    const ncnn::Mat& bias = bottom_blobs.size() >= 5 ? bottom_blobs[4] : ncnn::Mat();

    if (input.dims != 3 || weight.dims != 4 || offset.dims != 3 || (!mask.empty() && mask.dims != 3))
        return -100;

    const int in_channels = input.c;
    if (weight.d != in_channels)
        return -100;
    const int out_channels = weight.c;
    const int kernel_h = weight.h;
    const int kernel_w = weight.w;
    const int channels_per_group = deform_groups > 0 ? in_channels / deform_groups : 0;
    if (channels_per_group <= 0 || channels_per_group * deform_groups != in_channels)
        return -100;

    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int out_h = (input.h + 2 * pad_h - kernel_extent_h) / stride_h + 1;
    const int out_w = (input.w + 2 * pad_w - kernel_extent_w) / stride_w + 1;
    const int kernel_size = kernel_h * kernel_w;

    if (top_blobs.empty())
        top_blobs.resize(1);

    ncnn::Mat& output = top_blobs[0];
    output.create(out_w, out_h, out_channels, input.elemsize, opt.blob_allocator);
    if (output.empty())
        return -100;

    const ScopedNativeOpTimer timer(NativeOpKind::DeformConv, NativeOpBackend::Cpu);
    const float* weight_ptr = weight;
    const float* bias_ptr = bias.empty() ? nullptr : static_cast<const float*>(bias);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int oc = 0; oc < out_channels; oc++)
    {
        float* out_channel = output.channel(oc);
        for (int oy = 0; oy < out_h; oy++)
        {
            for (int ox = 0; ox < out_w; ox++)
            {
                float sum = bias_ptr ? bias_ptr[oc] : 0.f;
                const int base_y = oy * stride_h - pad_h;
                const int base_x = ox * stride_w - pad_w;

                for (int dg = 0; dg < deform_groups; dg++)
                {
                    const int channel_start = dg * channels_per_group;
                    for (int ky = 0; ky < kernel_h; ky++)
                    {
                        for (int kx = 0; kx < kernel_w; kx++)
                        {
                            const int kernel_index = ky * kernel_w + kx;
                            const int offset_index = (dg * kernel_size + kernel_index) * 2;
                            const float offset_y = offset.channel(offset_index).row(oy)[ox];
                            const float offset_x = offset.channel(offset_index + 1).row(oy)[ox];
                            const float mask_value = use_mask
                                ? mask.channel(dg * kernel_size + kernel_index).row(oy)[ox]
                                : 1.f;

                            const float sample_y = static_cast<float>(base_y + ky * dilation_h) + offset_y;
                            const float sample_x = static_cast<float>(base_x + kx * dilation_w) + offset_x;

                            for (int c = 0; c < channels_per_group; c++)
                            {
                                const int input_channel = channel_start + c;
                                const float value = bilinear_sample(input, input_channel, sample_y, sample_x);
                                const size_t weight_index =
                                    (((static_cast<size_t>(oc) * in_channels + input_channel) * kernel_h + ky) * kernel_w + kx);
                                sum += value * mask_value * weight_ptr[weight_index];
                            }
                        }
                    }
                }

                out_channel[oy * out_w + ox] = sum;
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int TorchVisionDeformConv2DLayer::forward(
    const std::vector<ncnn::VkMat>& bottom_blobs,
    std::vector<ncnn::VkMat>& top_blobs,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt) const
{
    deformconv_debug_log("[layer] forward enter\n");

    if (bottom_blobs.size() < 4)
        return -100;
    if (groups != 1 || pipeline_deformconv == 0)
        return -100;

    const ncnn::VkMat& input = bottom_blobs[0];
    const ncnn::VkMat& weight = bottom_blobs[1];
    const ncnn::VkMat& offset = bottom_blobs[2];
    const ncnn::VkMat& mask = bottom_blobs[3];
    const ncnn::VkMat bias = bottom_blobs.size() >= 5 ? bottom_blobs[4] : vkdev->get_dummy_buffer();

    deformconv_debug_log(
        "[layer] dims input=%d(%d,%d,%d) weight=%d(%d,%d,%d,%d) offset=%d(%d,%d,%d) mask=%d(%d,%d,%d) bias=%d\n",
        input.dims,
        input.w,
        input.h,
        input.c,
        weight.dims,
        weight.w,
        weight.h,
        weight.d,
        weight.c,
        offset.dims,
        offset.w,
        offset.h,
        offset.c,
        mask.dims,
        mask.w,
        mask.h,
        mask.c,
        bias.dims);

    if (input.elempack != 1 || weight.elempack != 1 || offset.elempack != 1 || mask.elempack != 1)
        return -100;
    if (input.dims != 3 || weight.dims != 4 || offset.dims != 3 || mask.dims != 3)
        return -100;
    if (weight.d != input.c)
        return -100;

    const int in_channels = input.c;
    const int out_channels = weight.c;
    const int kernel_h = weight.h;
    const int kernel_w = weight.w;
    const int channels_per_group = deform_groups > 0 ? in_channels / deform_groups : 0;
    if (channels_per_group <= 0 || channels_per_group * deform_groups != in_channels)
        return -100;

    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int out_h = (input.h + 2 * pad_h - kernel_extent_h) / stride_h + 1;
    const int out_w = (input.w + 2 * pad_w - kernel_extent_w) / stride_w + 1;

    if (top_blobs.empty())
        top_blobs.resize(1);

    ncnn::VkMat& output = top_blobs[0];
    output.create(out_w, out_h, out_channels, input.elemsize, 1, opt.blob_vkallocator);
    if (output.empty())
        return -100;

    const ScopedNativeOpTimer timer(NativeOpKind::DeformConv, NativeOpBackend::Vulkan);
    const ScopedNativeOpGpuTimestampQuery gpu_timer(NativeOpKind::DeformConv, cmd);
    deformconv_debug_log(
        "[layer] output=%d(%d,%d,%d) csteps in=%d w=%d off=%d mask=%d out=%d\n",
        output.dims,
        output.w,
        output.h,
        output.c,
        static_cast<int>(input.cstep),
        static_cast<int>(weight.cstep),
        static_cast<int>(offset.cstep),
        static_cast<int>(mask.cstep),
        static_cast<int>(output.cstep));

    std::vector<ncnn::VkMat> bindings(6);
    bindings[0] = input;
    bindings[1] = weight;
    bindings[2] = offset;
    bindings[3] = mask;
    bindings[4] = bias;
    bindings[5] = output;

    std::vector<ncnn::vk_constant_type> constants(24);
    constants[0].i = input.w;
    constants[1].i = input.h;
    constants[2].i = input.c;
    constants[3].i = static_cast<int>(input.cstep);
    constants[4].i = weight.w;
    constants[5].i = weight.h;
    constants[6].i = weight.d;
    constants[7].i = weight.c;
    constants[8].i = static_cast<int>(weight.cstep);
    constants[9].i = static_cast<int>(offset.cstep);
    constants[10].i = static_cast<int>(mask.cstep);
    constants[11].i = output.w;
    constants[12].i = output.h;
    constants[13].i = output.c;
    constants[14].i = static_cast<int>(output.cstep);
    constants[15].i = stride_w;
    constants[16].i = stride_h;
    constants[17].i = pad_w;
    constants[18].i = pad_h;
    constants[19].i = dilation_w;
    constants[20].i = dilation_h;
    constants[21].i = deform_groups;
    constants[22].i = use_mask;
    constants[23].i = bottom_blobs.size() >= 5 ? 1 : 0;

    deformconv_debug_log("[layer] record_pipeline\n");
    cmd.record_pipeline(pipeline_deformconv, bindings, constants, output);
    deformconv_debug_log("[layer] record_pipeline done\n");
    return 0;
}
#endif

ncnn::Layer* torchvision_deform_conv2d_layer_creator(void* userdata)
{
    (void)userdata;
    return new TorchVisionDeformConv2DLayer;
}

void torchvision_deform_conv2d_layer_destroyer(ncnn::Layer* layer, void* userdata)
{
    (void)userdata;
    delete layer;
}

int register_torchvision_deform_conv2d_layers(ncnn::Net& net)
{
    int ret = net.register_custom_layer(
        "torchvision.deform_conv2d",
        torchvision_deform_conv2d_layer_creator,
        torchvision_deform_conv2d_layer_destroyer);
    if (ret != 0)
        return ret;

    return net.register_custom_layer(
        "pnnx.custom_op.torchvision.deform_conv2d",
        torchvision_deform_conv2d_layer_creator,
        torchvision_deform_conv2d_layer_destroyer);
}

} // namespace lada
