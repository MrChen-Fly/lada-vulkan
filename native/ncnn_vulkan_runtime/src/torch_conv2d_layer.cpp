// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#include "torch_conv2d_layer.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "gpu.h"
#include "layer_type.h"
#include "modelbin.h"
#include "net.h"

namespace lada {

namespace {

void destroy_layer(ncnn::Layer*& layer, const ncnn::Option& opt)
{
    if (layer == nullptr)
        return;

    layer->destroy_pipeline(opt);
    delete layer;
    layer = nullptr;
}

int create_padding_layer_cpu(
    ncnn::Layer*& layer,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    int pad_mode,
    float pad_value,
    const ncnn::Option& opt)
{
    layer = ncnn::create_layer_cpu(ncnn::LayerType::Padding);
    if (layer == nullptr)
        return -100;

    ncnn::ParamDict pd;
    pd.set(0, pad_top);
    pd.set(1, pad_bottom);
    pd.set(2, pad_left);
    pd.set(3, pad_right);
    pd.set(4, pad_mode);
    pd.set(5, pad_value);

    int ret = layer->load_param(pd);
    if (ret != 0)
        return ret;

    ret = layer->load_model(ncnn::ModelBinFromMatArray(nullptr));
    if (ret != 0)
        return ret;

    return layer->create_pipeline(opt);
}

int create_conv_layer_cpu(
    ncnn::Layer*& layer,
    int num_output,
    int kernel_w,
    int kernel_h,
    int dilation_w,
    int dilation_h,
    int stride_w,
    int stride_h,
    int bias_term,
    int weight_data_size,
    int group,
    int activation_type,
    const ncnn::Mat& activation_params,
    const ncnn::Mat& weight_data,
    const ncnn::Mat& bias_data,
    const ncnn::Option& opt)
{
    layer = ncnn::create_layer_cpu(ncnn::LayerType::ConvolutionDepthWise);
    if (layer == nullptr)
        return -100;

    ncnn::ParamDict pd;
    pd.set(0, num_output);
    pd.set(1, kernel_w);
    pd.set(11, kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, 0);
    pd.set(14, 0);
    pd.set(15, 0);
    pd.set(16, 0);
    pd.set(18, 0.f);
    pd.set(5, bias_term);
    pd.set(6, weight_data_size);
    pd.set(7, group);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    int ret = layer->load_param(pd);
    if (ret != 0)
        return ret;

    ncnn::Mat weights[2];
    weights[0] = weight_data;
    weights[1] = bias_data;
    ret = layer->load_model(ncnn::ModelBinFromMatArray(weights));
    if (ret != 0)
        return ret;

    return layer->create_pipeline(opt);
}

int reflect101_coord(int coord, int limit)
{
    if (limit <= 1)
        return 0;

    while (coord < 0 || coord >= limit)
    {
        coord = coord < 0 ? -coord : 2 * limit - coord - 2;
    }
    return coord;
}

int reflect_coord(int coord, int limit)
{
    if (limit <= 1)
        return 0;

    while (coord < 0 || coord >= limit)
    {
        coord = coord < 0 ? -coord - 1 : 2 * limit - coord - 1;
    }
    return coord;
}

ncnn::Mat create_reflect_padding_cpu(
    const ncnn::Mat& input,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right)
{
    if (input.dims != 3 || input.elemsize != sizeof(float))
        return ncnn::Mat();

    ncnn::Mat output(input.w + pad_left + pad_right, input.h + pad_top + pad_bottom, input.c);
    if (output.empty())
        return output;

    for (int channel = 0; channel < input.c; ++channel)
    {
        const float* src = input.channel(channel);
        float* dst = output.channel(channel);
        for (int y = 0; y < output.h; ++y)
        {
            const int src_y = reflect_coord(y - pad_top, input.h);
            for (int x = 0; x < output.w; ++x)
            {
                const int src_x = reflect_coord(x - pad_left, input.w);
                dst[static_cast<std::size_t>(y) * output.w + x] =
                    src[static_cast<std::size_t>(src_y) * input.w + src_x];
            }
        }
    }

    return output;
}

#if NCNN_VULKAN
static const char kTorchConv2dShader[] = R"(
#version 450

layout(binding = 0) readonly buffer input_blob { sfp input_blob_data[]; };
layout(binding = 1) readonly buffer weight_blob { sfp weight_blob_data[]; };
layout(binding = 2) readonly buffer bias_blob { sfp bias_blob_data[]; };
layout(binding = 3) writeonly buffer output_blob { sfp output_blob_data[]; };

layout(push_constant) uniform parameter
{
    int input_w;
    int input_h;
    int input_c;
    int input_cstep;
    int output_w;
    int output_h;
    int output_c;
    int output_cstep;
    int kernel_w;
    int kernel_h;
    int stride_w;
    int stride_h;
    int dilation_w;
    int dilation_h;
    int pad_left;
    int pad_top;
    int group;
    int channels_per_group;
    int outputs_per_group;
    int pad_mode;
    int use_bias;
    float pad_value;
} p;

int reflect101_coord(int coord, int limit)
{
    if (limit <= 1)
        return 0;

    while (coord < 0 || coord >= limit)
    {
        coord = coord < 0 ? -coord : 2 * limit - coord - 2;
    }
    return coord;
}

int reflect_coord(int coord, int limit)
{
    if (limit <= 1)
        return 0;

    while (coord < 0 || coord >= limit)
    {
        coord = coord < 0 ? -coord - 1 : 2 * limit - coord - 1;
    }
    return coord;
}

float input_value_at(int channel, int y, int x)
{
    const int index = channel * p.input_cstep + y * p.input_w + x;
    return float(buffer_ld1(input_blob_data, index));
}

float sample_input_value(int channel, int y, int x)
{
    if (channel < 0 || channel >= p.input_c)
        return p.pad_value;

    if (x >= 0 && x < p.input_w && y >= 0 && y < p.input_h)
        return input_value_at(channel, y, x);

    if (p.pad_mode == 0)
        return p.pad_value;

    if (p.pad_mode == 1)
    {
        x = clamp(x, 0, p.input_w - 1);
        y = clamp(y, 0, p.input_h - 1);
    }
    else if (p.pad_mode == 2)
    {
        x = reflect101_coord(x, p.input_w);
        y = reflect101_coord(y, p.input_h);
    }
    else if (p.pad_mode == 3)
    {
        x = reflect_coord(x, p.input_w);
        y = reflect_coord(y, p.input_h);
    }
    else
    {
        return p.pad_value;
    }

    return input_value_at(channel, y, x);
}

float weight_value_at(int out_channel, int in_channel_local, int ky, int kx)
{
    const int index =
        (((out_channel * p.channels_per_group + in_channel_local) * p.kernel_h + ky) * p.kernel_w + kx);
    return float(buffer_ld1(weight_blob_data, index));
}

void output_store(int out_channel, int y, int x, float value)
{
    const int index = out_channel * p.output_cstep + y * p.output_w + x;
    buffer_st1(output_blob_data, index, afp(value));
}

void main()
{
    const int ox = int(gl_GlobalInvocationID.x);
    const int oy = int(gl_GlobalInvocationID.y);
    const int oc = int(gl_GlobalInvocationID.z);
    if (ox >= p.output_w || oy >= p.output_h || oc >= p.output_c)
        return;

    const int group_index = oc / p.outputs_per_group;
    const int input_channel_base = group_index * p.channels_per_group;
    const int base_x = ox * p.stride_w - p.pad_left;
    const int base_y = oy * p.stride_h - p.pad_top;

    float sum = p.use_bias != 0 ? float(buffer_ld1(bias_blob_data, oc)) : 0.f;
    for (int ic_local = 0; ic_local < p.channels_per_group; ++ic_local)
    {
        const int input_channel = input_channel_base + ic_local;
        for (int ky = 0; ky < p.kernel_h; ++ky)
        {
            const int sample_y = base_y + ky * p.dilation_h;
            for (int kx = 0; kx < p.kernel_w; ++kx)
            {
                const int sample_x = base_x + kx * p.dilation_w;
                const float input_value = sample_input_value(input_channel, sample_y, sample_x);
                const float weight_value = weight_value_at(oc, ic_local, ky, kx);
                sum += input_value * weight_value;
            }
        }
    }

    output_store(oc, oy, ox, sum);
}
)";
#endif

} // namespace

TorchConv2DLayer::TorchConv2DLayer()
    : num_output_(0),
      kernel_w_(0),
      kernel_h_(0),
      dilation_w_(1),
      dilation_h_(1),
      stride_w_(1),
      stride_h_(1),
      pad_left_(0),
      pad_right_(0),
      pad_top_(0),
      pad_bottom_(0),
      pad_value_(0.f),
      bias_term_(0),
      weight_data_size_(0),
      group_(1),
      activation_type_(0),
      pad_mode_(0),
      padding_cpu_(nullptr),
      conv_cpu_(nullptr)
#if NCNN_VULKAN
      ,
      pipeline_vulkan_(nullptr)
#endif
{
    one_blob_only = true;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
#endif
}

TorchConv2DLayer::~TorchConv2DLayer()
{
    ncnn::Option cpu_opt;
    cpu_opt.use_vulkan_compute = false;
    destroy_layer(padding_cpu_, cpu_opt);
    destroy_layer(conv_cpu_, cpu_opt);

#if NCNN_VULKAN
    delete pipeline_vulkan_;
    pipeline_vulkan_ = nullptr;
    weight_data_gpu_.release();
    bias_data_gpu_.release();
#endif
}

int TorchConv2DLayer::load_param(const ncnn::ParamDict& pd)
{
    num_output_ = pd.get(0, 0);
    kernel_w_ = pd.get(1, 0);
    kernel_h_ = pd.get(11, kernel_w_);
    dilation_w_ = pd.get(2, 1);
    dilation_h_ = pd.get(12, dilation_w_);
    stride_w_ = pd.get(3, 1);
    stride_h_ = pd.get(13, stride_w_);
    pad_left_ = pd.get(4, 0);
    pad_right_ = pd.get(15, pad_left_);
    pad_top_ = pd.get(14, pad_left_);
    pad_bottom_ = pd.get(16, pad_top_);
    pad_value_ = pd.get(18, 0.f);
    bias_term_ = pd.get(5, 0);
    weight_data_size_ = pd.get(6, 0);
    group_ = pd.get(7, 1);
    activation_type_ = pd.get(9, 0);
    activation_params_ = pd.get(10, ncnn::Mat());
    pad_mode_ = pd.get(20, 0);
    return 0;
}

int TorchConv2DLayer::load_model(const ncnn::ModelBin& mb)
{
    weight_data_ = mb.load(weight_data_size_, 0);
    if (weight_data_.empty())
        return -100;

    if (bias_term_)
    {
        bias_data_ = mb.load(num_output_, 1);
        if (bias_data_.empty())
            return -100;
    }

    return 0;
}

bool TorchConv2DLayer::needs_padding() const
{
    return pad_left_ != 0 || pad_right_ != 0 || pad_top_ != 0 || pad_bottom_ != 0;
}

int TorchConv2DLayer::create_pipeline(const ncnn::Option& opt)
{
    ncnn::Option cpu_opt = opt;
    cpu_opt.use_vulkan_compute = false;
    cpu_opt.use_packing_layout = false;

    int ret = create_conv_layer_cpu(
        conv_cpu_,
        num_output_,
        kernel_w_,
        kernel_h_,
        dilation_w_,
        dilation_h_,
        stride_w_,
        stride_h_,
        bias_term_,
        weight_data_size_,
        group_,
        activation_type_,
        activation_params_,
        weight_data_,
        bias_data_,
        cpu_opt);
    if (ret != 0)
        return ret;

    if (needs_padding() && pad_mode_ != 3)
    {
        ret = create_padding_layer_cpu(
            padding_cpu_,
            pad_top_,
            pad_bottom_,
            pad_left_,
            pad_right_,
            pad_mode_,
            pad_value_,
            cpu_opt);
        if (ret != 0)
            return ret;
    }

#if NCNN_VULKAN
    if (opt.use_vulkan_compute && vkdev != nullptr)
    {
        delete pipeline_vulkan_;
        pipeline_vulkan_ = nullptr;

        std::vector<uint32_t> spirv;
        ret = ncnn::compile_spirv_module(
            kTorchConv2dShader,
            static_cast<int>(sizeof(kTorchConv2dShader) - 1),
            opt,
            spirv);
        if (ret != 0)
            return ret;

        pipeline_vulkan_ = new ncnn::Pipeline(vkdev);
        pipeline_vulkan_->set_optimal_local_size_xyz(4, 4, 1);
        ret = pipeline_vulkan_->create(
            spirv.data(),
            spirv.size() * sizeof(uint32_t),
            std::vector<ncnn::vk_specialization_type>());
        if (ret != 0)
            return ret;
    }
#else
    (void)opt;
#endif

    return 0;
}

int TorchConv2DLayer::destroy_pipeline(const ncnn::Option& opt)
{
    ncnn::Option cpu_opt = opt;
    cpu_opt.use_vulkan_compute = false;
    destroy_layer(padding_cpu_, cpu_opt);
    destroy_layer(conv_cpu_, cpu_opt);

#if NCNN_VULKAN
    delete pipeline_vulkan_;
    pipeline_vulkan_ = nullptr;
    weight_data_gpu_.release();
    bias_data_gpu_.release();
#else
    (void)opt;
#endif

    return 0;
}

int TorchConv2DLayer::forward(
    const std::vector<ncnn::Mat>& bottom_blobs,
    std::vector<ncnn::Mat>& top_blobs,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 1 || conv_cpu_ == nullptr)
        return -100;

    if (top_blobs.empty())
        top_blobs.resize(1);

    const ncnn::Mat& bottom_blob = bottom_blobs[0];
    const ncnn::Mat* conv_input = &bottom_blob;
    ncnn::Mat padded_blob;
    if (padding_cpu_ != nullptr)
    {
        const int ret = padding_cpu_->forward(bottom_blob, padded_blob, opt);
        if (ret != 0)
            return ret;
        conv_input = &padded_blob;
    }
    else if (needs_padding() && pad_mode_ == 3)
    {
        padded_blob = create_reflect_padding_cpu(bottom_blob, pad_top_, pad_bottom_, pad_left_, pad_right_);
        if (padded_blob.empty())
            return -100;
        conv_input = &padded_blob;
    }

    return conv_cpu_->forward(*conv_input, top_blobs[0], opt);
}

#if NCNN_VULKAN
int TorchConv2DLayer::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    if (weight_data_.dims != 1 || weight_data_.elempack != 1)
        return -100;

    weight_data_gpu_.create(weight_data_.w, weight_data_.elemsize, 1, opt.blob_vkallocator);
    if (weight_data_gpu_.empty())
        return -100;
    cmd.record_upload(weight_data_, weight_data_gpu_, opt);

    if (bias_term_)
    {
        if (bias_data_.dims != 1 || bias_data_.elempack != 1)
            return -100;
        bias_data_gpu_.create(bias_data_.w, bias_data_.elemsize, 1, opt.blob_vkallocator);
        if (bias_data_gpu_.empty())
            return -100;
        cmd.record_upload(bias_data_, bias_data_gpu_, opt);
    }
    else
    {
        bias_data_gpu_.release();
    }

    return 0;
}

int TorchConv2DLayer::forward(
    const std::vector<ncnn::VkMat>& bottom_blobs,
    std::vector<ncnn::VkMat>& top_blobs,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 1 || pipeline_vulkan_ == nullptr)
        return -100;

    const ncnn::VkMat& bottom_blob = bottom_blobs[0];
    if (bottom_blob.dims != 3 || bottom_blob.elempack != 1 || weight_data_gpu_.empty())
        return -100;

    if (group_ <= 0 || bottom_blob.c % group_ != 0 || num_output_ % group_ != 0)
        return -100;

    const int channels_per_group = bottom_blob.c / group_;
    const int outputs_per_group = num_output_ / group_;
    const int expected_weight_size = num_output_ * channels_per_group * kernel_h_ * kernel_w_;
    if (expected_weight_size != weight_data_size_)
        return -100;

    const int kernel_extent_h = dilation_h_ * (kernel_h_ - 1) + 1;
    const int kernel_extent_w = dilation_w_ * (kernel_w_ - 1) + 1;
    const int out_h = (bottom_blob.h + pad_top_ + pad_bottom_ - kernel_extent_h) / stride_h_ + 1;
    const int out_w = (bottom_blob.w + pad_left_ + pad_right_ - kernel_extent_w) / stride_w_ + 1;
    if (out_h <= 0 || out_w <= 0)
        return -100;

    if (top_blobs.empty())
        top_blobs.resize(1);

    ncnn::VkMat& output = top_blobs[0];
    output.create(out_w, out_h, num_output_, bottom_blob.elemsize, 1, opt.blob_vkallocator);
    if (output.empty())
        return -100;

    const ncnn::VkMat bias_gpu = bias_term_ ? bias_data_gpu_ : vkdev->get_dummy_buffer();

    std::vector<ncnn::VkMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = weight_data_gpu_;
    bindings[2] = bias_gpu;
    bindings[3] = output;

    std::vector<ncnn::vk_constant_type> constants(22);
    constants[0].i = bottom_blob.w;
    constants[1].i = bottom_blob.h;
    constants[2].i = bottom_blob.c;
    constants[3].i = static_cast<int>(bottom_blob.cstep);
    constants[4].i = output.w;
    constants[5].i = output.h;
    constants[6].i = output.c;
    constants[7].i = static_cast<int>(output.cstep);
    constants[8].i = kernel_w_;
    constants[9].i = kernel_h_;
    constants[10].i = stride_w_;
    constants[11].i = stride_h_;
    constants[12].i = dilation_w_;
    constants[13].i = dilation_h_;
    constants[14].i = pad_left_;
    constants[15].i = pad_top_;
    constants[16].i = group_;
    constants[17].i = channels_per_group;
    constants[18].i = outputs_per_group;
    constants[19].i = pad_mode_;
    constants[20].i = bias_term_ ? 1 : 0;
    constants[21].f = pad_value_;

    cmd.record_pipeline(pipeline_vulkan_, bindings, constants, output);
    return 0;
}
#endif

ncnn::Layer* torch_conv2d_layer_creator(void* userdata)
{
    (void)userdata;
    return new TorchConv2DLayer;
}

void torch_conv2d_layer_destroyer(ncnn::Layer* layer, void* userdata)
{
    (void)userdata;
    delete layer;
}

int register_torch_conv2d_layers(ncnn::Net& net)
{
    int ret = net.register_custom_layer(
        "torch.conv2d",
        torch_conv2d_layer_creator,
        torch_conv2d_layer_destroyer);
    if (ret != 0)
        return ret;

    ret = net.register_custom_layer(
        "pnnx.custom_op.torch.conv2d",
        torch_conv2d_layer_creator,
        torch_conv2d_layer_destroyer);
    if (ret != 0)
        return ret;

    return net.register_custom_layer(
        "lada.TorchConv2D",
        torch_conv2d_layer_creator,
        torch_conv2d_layer_destroyer);
}

} // namespace lada
