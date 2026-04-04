// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#include "lada_gridsample_layer.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "net.h"
#include "native_op_profile.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif

namespace lada {

namespace {

static inline float grid_sample_unormalize(int size, float coord, int align_corner)
{
    return align_corner ? (coord + 1.f) * 0.5f * (size - 1) : ((coord + 1.f) * size - 1.f) * 0.5f;
}

static inline float clamp_border(float value, int limit)
{
    return std::min(std::max(value, 0.f), static_cast<float>(limit - 1));
}

static inline float input_value_at(const ncnn::Mat& image, int channel, int y, int x)
{
    if (channel < 0 || channel >= image.c || y < 0 || y >= image.h || x < 0 || x >= image.w)
        return 0.f;

    const float* channel_ptr = image.channel(channel);
    return channel_ptr[y * image.w + x];
}

static inline float bilinear_sample(const ncnn::Mat& image, int channel, float sample_y, float sample_x)
{
    if (sample_y <= -1.f || sample_x <= -1.f || sample_y >= image.h || sample_x >= image.w)
        return 0.f;

    const int y_low = static_cast<int>(std::floor(sample_y));
    const int x_low = static_cast<int>(std::floor(sample_x));
    const int y_high = y_low + 1;
    const int x_high = x_low + 1;

    const float ly = sample_y - y_low;
    const float lx = sample_x - x_low;
    const float hy = 1.f - ly;
    const float hx = 1.f - lx;

    const float v1 = input_value_at(image, channel, y_low, x_low);
    const float v2 = input_value_at(image, channel, y_low, x_high);
    const float v3 = input_value_at(image, channel, y_high, x_low);
    const float v4 = input_value_at(image, channel, y_high, x_high);

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

#if NCNN_VULKAN
static const char kLadaGridSampleShader[] = R"(
#version 450

layout(binding = 0) readonly buffer input_blob { sfp input_blob_data[]; };
layout(binding = 1) readonly buffer grid_blob { sfp grid_blob_data[]; };
layout(binding = 2) writeonly buffer output_blob { sfp output_blob_data[]; };

layout(push_constant) uniform parameter
{
    int input_w;
    int input_h;
    int input_c;
    int input_cstep;
    int grid_cstep;
    int output_w;
    int output_h;
    int output_c;
    int output_cstep;
    int padding_mode;
    int align_corner;
} p;

float grid_sample_unormalize(int size, float coord)
{
    return p.align_corner != 0 ? (coord + 1.0) * 0.5 * float(size - 1) : ((coord + 1.0) * float(size) - 1.0) * 0.5;
}

float input_value_at(int channel, int y, int x)
{
    if (channel < 0 || channel >= p.input_c || y < 0 || y >= p.input_h || x < 0 || x >= p.input_w)
        return 0.0;

    const int index = channel * p.input_cstep + y * p.input_w + x;
    return float(buffer_ld1(input_blob_data, index));
}

float bilinear_sample_input(int channel, float sample_y, float sample_x)
{
    if (p.padding_mode == 2)
    {
        sample_x = clamp(sample_x, 0.0, float(p.input_w - 1));
        sample_y = clamp(sample_y, 0.0, float(p.input_h - 1));
    }

    if (sample_y <= -1.0 || sample_x <= -1.0 || sample_y >= float(p.input_h) || sample_x >= float(p.input_w))
        return 0.0;

    const int y_low = int(floor(sample_y));
    const int x_low = int(floor(sample_x));
    const int y_high = y_low + 1;
    const int x_high = x_low + 1;

    const float ly = sample_y - float(y_low);
    const float lx = sample_x - float(x_low);
    const float hy = 1.0 - ly;
    const float hx = 1.0 - lx;

    const float v1 = input_value_at(channel, y_low, x_low);
    const float v2 = input_value_at(channel, y_low, x_high);
    const float v3 = input_value_at(channel, y_high, x_low);
    const float v4 = input_value_at(channel, y_high, x_high);

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

float grid_value_at(int out_y, int out_x, int component)
{
    const int index = out_y * p.grid_cstep + out_x * 2 + component;
    return float(buffer_ld1(grid_blob_data, index));
}

void main()
{
    const int ox = int(gl_GlobalInvocationID.x);
    const int oy = int(gl_GlobalInvocationID.y);
    const int oc = int(gl_GlobalInvocationID.z);

    if (ox >= p.output_w || oy >= p.output_h || oc >= p.output_c)
        return;

    float sample_x = grid_value_at(oy, ox, 0);
    float sample_y = grid_value_at(oy, ox, 1);
    sample_x = grid_sample_unormalize(p.input_w, sample_x);
    sample_y = grid_sample_unormalize(p.input_h, sample_y);

    const float value = bilinear_sample_input(oc, sample_y, sample_x);
    const int output_index = oc * p.output_cstep + oy * p.output_w + ox;
    buffer_st1(output_blob_data, output_index, afp(value));
}
)";
#endif

} // namespace

LadaGridSampleLayer::LadaGridSampleLayer()
{
    one_blob_only = false;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
    pipeline_gridsample = 0;
#endif
}

int LadaGridSampleLayer::load_param(const ncnn::ParamDict& pd)
{
    sample_type = pd.get(0, 1);
    padding_mode = pd.get(1, 1);
    align_corner = pd.get(2, 0);
    permute_fusion = pd.get(3, 0);

    if (sample_type != 1)
        return -1;
    if (padding_mode != 1 && padding_mode != 2)
        return -1;
    if (permute_fusion != 0)
        return -1;

    return 0;
}

int LadaGridSampleLayer::create_pipeline(const ncnn::Option& opt)
{
#if NCNN_VULKAN
    if (!opt.use_vulkan_compute)
        return 0;

    std::vector<uint32_t> spirv;
    const int compile_ret = ncnn::compile_spirv_module(
        kLadaGridSampleShader,
        static_cast<int>(sizeof(kLadaGridSampleShader) - 1),
        opt,
        spirv);
    if (compile_ret != 0)
        return compile_ret;

    pipeline_gridsample = new ncnn::Pipeline(vkdev);
    pipeline_gridsample->set_optimal_local_size_xyz(8, 8, 1);
    return pipeline_gridsample->create(
        spirv.data(),
        spirv.size() * sizeof(uint32_t),
        std::vector<ncnn::vk_specialization_type>());
#else
    (void)opt;
    return 0;
#endif
}

int LadaGridSampleLayer::destroy_pipeline(const ncnn::Option& opt)
{
#if NCNN_VULKAN
    (void)opt;
    delete pipeline_gridsample;
    pipeline_gridsample = 0;
#else
    (void)opt;
#endif
    return 0;
}

int LadaGridSampleLayer::forward(
    const std::vector<ncnn::Mat>& bottom_blobs,
    std::vector<ncnn::Mat>& top_blobs,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 2)
        return -100;

    const ncnn::Mat& input = bottom_blobs[0];
    const ncnn::Mat& grid = bottom_blobs[1];
    if (input.dims != 3 || grid.dims != 3 || grid.w != 2)
        return -100;

    const int out_w = grid.h;
    const int out_h = grid.c;
    const int channels = input.c;

    if (top_blobs.empty())
        top_blobs.resize(1);

    ncnn::Mat& output = top_blobs[0];
    output.create(out_w, out_h, channels, input.elemsize, opt.blob_allocator);
    if (output.empty())
        return -100;

    const ScopedNativeOpTimer timer(NativeOpKind::GridSample, NativeOpBackend::Cpu);
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int channel = 0; channel < channels; channel++)
    {
        float* out_ptr = output.channel(channel);
        for (int oy = 0; oy < out_h; oy++)
        {
            const float* grid_ptr = grid.channel(oy);
            for (int ox = 0; ox < out_w; ox++)
            {
                float sample_x = grid_sample_unormalize(input.w, grid_ptr[0], align_corner);
                float sample_y = grid_sample_unormalize(input.h, grid_ptr[1], align_corner);

                if (padding_mode == 2)
                {
                    sample_x = clamp_border(sample_x, input.w);
                    sample_y = clamp_border(sample_y, input.h);
                }

                out_ptr[oy * out_w + ox] = bilinear_sample(input, channel, sample_y, sample_x);
                grid_ptr += 2;
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int LadaGridSampleLayer::forward(
    const std::vector<ncnn::VkMat>& bottom_blobs,
    std::vector<ncnn::VkMat>& top_blobs,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 2 || pipeline_gridsample == 0)
        return -100;

    const ncnn::VkMat& input = bottom_blobs[0];
    const ncnn::VkMat& grid = bottom_blobs[1];
    if (input.dims != 3 || grid.dims != 3 || input.elempack != 1 || grid.elempack != 1 || grid.w != 2)
        return -100;

    const int out_w = grid.h;
    const int out_h = grid.c;
    const int channels = input.c;

    if (top_blobs.empty())
        top_blobs.resize(1);

    ncnn::VkMat& output = top_blobs[0];
    output.create(out_w, out_h, channels, input.elemsize, 1, opt.blob_vkallocator);
    if (output.empty())
        return -100;

    const ScopedNativeOpTimer timer(NativeOpKind::GridSample, NativeOpBackend::Vulkan);
    const ScopedNativeOpGpuTimestampQuery gpu_timer(NativeOpKind::GridSample, cmd);
    std::vector<ncnn::VkMat> bindings(3);
    bindings[0] = input;
    bindings[1] = grid;
    bindings[2] = output;

    std::vector<ncnn::vk_constant_type> constants(11);
    constants[0].i = input.w;
    constants[1].i = input.h;
    constants[2].i = input.c;
    constants[3].i = static_cast<int>(input.cstep);
    constants[4].i = static_cast<int>(grid.cstep);
    constants[5].i = output.w;
    constants[6].i = output.h;
    constants[7].i = output.c;
    constants[8].i = static_cast<int>(output.cstep);
    constants[9].i = padding_mode;
    constants[10].i = align_corner;

    cmd.record_pipeline(pipeline_gridsample, bindings, constants, output);
    return 0;
}
#endif

ncnn::Layer* lada_gridsample_layer_creator(void* userdata)
{
    (void)userdata;
    return new LadaGridSampleLayer;
}

void lada_gridsample_layer_destroyer(ncnn::Layer* layer, void* userdata)
{
    (void)userdata;
    delete layer;
}

int register_lada_gridsample_layers(ncnn::Net& net)
{
    int ret = net.register_custom_layer(
        "lada.GridSample",
        lada_gridsample_layer_creator,
        lada_gridsample_layer_destroyer);
    if (ret != 0)
        return ret;

    return net.register_custom_layer(
        "pnnx.custom_op.lada.GridSample",
        lada_gridsample_layer_creator,
        lada_gridsample_layer_destroyer);
}

} // namespace lada
