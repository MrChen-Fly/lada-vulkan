// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#include "lada_yolo_attention_layer.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "layer_type.h"
#include "net.h"

namespace lada {

namespace {

bool yolo_attention_debug_enabled()
{
    static const bool enabled = []() {
        const char* value = std::getenv("LADA_NCNN_DEBUG");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

void yolo_attention_debug_log(const char* format, ...)
{
    if (!yolo_attention_debug_enabled())
        return;

    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fflush(stderr);
}

int create_transpose_layer_cpu(ncnn::Layer*& layer, const ncnn::Option& opt)
{
    layer = ncnn::create_layer_cpu(ncnn::LayerType::Permute);
    if (layer == nullptr)
        return -100;

    ncnn::ParamDict pd;
    pd.set(0, 1);
    int ret = layer->load_param(pd);
    if (ret != 0)
        return ret;

    ret = layer->load_model(ncnn::ModelBinFromMatArray(0));
    if (ret != 0)
        return ret;

    return layer->create_pipeline(opt);
}

int create_sdpa_layer_cpu(ncnn::Layer*& layer, float scale, const ncnn::Option& opt)
{
    layer = ncnn::create_layer_cpu(ncnn::LayerType::SDPA);
    if (layer == nullptr)
        return -100;

    ncnn::ParamDict pd;
    pd.set(5, 0);
    pd.set(6, scale);
    pd.set(7, 0);
    int ret = layer->load_param(pd);
    if (ret != 0)
        return ret;

    ret = layer->load_model(ncnn::ModelBinFromMatArray(0));
    if (ret != 0)
        return ret;

    return layer->create_pipeline(opt);
}

void destroy_layer(ncnn::Layer*& layer, const ncnn::Option& opt)
{
    if (layer == nullptr)
        return;

    layer->destroy_pipeline(opt);
    delete layer;
    layer = nullptr;
}

#if NCNN_VULKAN
ncnn::Option make_safe_sdpa_vulkan_option(const ncnn::Option& opt)
{
    ncnn::Option sdpa_opt = opt;
    // AMD 880M currently trips over SDPA cooperative-matrix shader compilation.
    // Keep Vulkan enabled, but force SDPA onto the simpler non-coop path.
    sdpa_opt.use_cooperative_matrix = false;
    return sdpa_opt;
}

int create_transpose_layer_vulkan(
    ncnn::Layer*& layer,
    const ncnn::VulkanDevice* vkdev,
    const ncnn::Option& opt)
{
    layer = ncnn::create_layer_vulkan(ncnn::LayerType::Permute);
    if (layer == nullptr)
        return -100;

    layer->vkdev = vkdev;

    ncnn::ParamDict pd;
    pd.set(0, 1);
    int ret = layer->load_param(pd);
    if (ret != 0)
        return ret;

    ret = layer->load_model(ncnn::ModelBinFromMatArray(0));
    if (ret != 0)
        return ret;

    return layer->create_pipeline(opt);
}

int create_sdpa_layer_vulkan(
    ncnn::Layer*& layer,
    const ncnn::VulkanDevice* vkdev,
    float scale,
    const ncnn::Option& opt)
{
    layer = ncnn::create_layer_vulkan(ncnn::LayerType::SDPA);
    if (layer == nullptr)
        return -100;

    layer->vkdev = vkdev;

    ncnn::ParamDict pd;
    pd.set(5, 0);
    pd.set(6, scale);
    pd.set(7, 0);
    int ret = layer->load_param(pd);
    if (ret != 0)
        return ret;

    ret = layer->load_model(ncnn::ModelBinFromMatArray(0));
    if (ret != 0)
        return ret;

    return layer->create_pipeline(opt);
}
#endif

} // namespace

LadaYoloAttentionLayer::LadaYoloAttentionLayer()
    : scale_(0.f),
      permute_cpu_(nullptr),
      sdpa_cpu_(nullptr)
#if NCNN_VULKAN
      ,
      permute_vulkan_(nullptr),
      sdpa_vulkan_(nullptr)
#endif
{
    one_blob_only = false;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;
#endif
}

LadaYoloAttentionLayer::~LadaYoloAttentionLayer()
{
    ncnn::Option opt;
    opt.use_vulkan_compute = false;
    destroy_layer(permute_cpu_, opt);
    destroy_layer(sdpa_cpu_, opt);

#if NCNN_VULKAN
    ncnn::Option vk_opt;
    vk_opt.use_vulkan_compute = true;
    ncnn::Option sdpa_vk_opt = make_safe_sdpa_vulkan_option(vk_opt);
    destroy_layer(permute_vulkan_, vk_opt);
    destroy_layer(sdpa_vulkan_, sdpa_vk_opt);
#endif
}

int LadaYoloAttentionLayer::load_param(const ncnn::ParamDict& pd)
{
    scale_ = pd.get(0, 0.f);
    return 0;
}

int LadaYoloAttentionLayer::create_pipeline(const ncnn::Option& opt)
{
    yolo_attention_debug_log(
        "[yolo_attention] create_pipeline use_vulkan=%d scale=%f\n",
        opt.use_vulkan_compute ? 1 : 0,
        scale_);

    ncnn::Option cpu_opt = opt;
    cpu_opt.use_vulkan_compute = false;

    int ret = create_transpose_layer_cpu(permute_cpu_, cpu_opt);
    if (ret != 0)
        return ret;

    ret = create_sdpa_layer_cpu(sdpa_cpu_, scale_, cpu_opt);
    if (ret != 0)
        return ret;

#if NCNN_VULKAN
    if (opt.use_vulkan_compute && vkdev != nullptr)
    {
        const ncnn::Option sdpa_vulkan_opt = make_safe_sdpa_vulkan_option(opt);
        yolo_attention_debug_log(
            "[yolo_attention] create_pipeline sdpa_opt fp16(packed=%d storage=%d arithmetic=%d) coop=%d\n",
            sdpa_vulkan_opt.use_fp16_packed ? 1 : 0,
            sdpa_vulkan_opt.use_fp16_storage ? 1 : 0,
            sdpa_vulkan_opt.use_fp16_arithmetic ? 1 : 0,
            sdpa_vulkan_opt.use_cooperative_matrix ? 1 : 0);

        ret = create_transpose_layer_vulkan(permute_vulkan_, vkdev, opt);
        if (ret != 0)
            return ret;

        ret = create_sdpa_layer_vulkan(sdpa_vulkan_, vkdev, scale_, sdpa_vulkan_opt);
        if (ret != 0)
            return ret;
    }
#endif

    return 0;
}

int LadaYoloAttentionLayer::destroy_pipeline(const ncnn::Option& opt)
{
    ncnn::Option cpu_opt = opt;
    cpu_opt.use_vulkan_compute = false;

    destroy_layer(permute_cpu_, cpu_opt);
    destroy_layer(sdpa_cpu_, cpu_opt);

#if NCNN_VULKAN
    destroy_layer(permute_vulkan_, opt);
    destroy_layer(sdpa_vulkan_, make_safe_sdpa_vulkan_option(opt));
#endif

    return 0;
}

int LadaYoloAttentionLayer::forward(
    const std::vector<ncnn::Mat>& bottom_blobs,
    std::vector<ncnn::Mat>& top_blobs,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 3 || permute_cpu_ == nullptr || sdpa_cpu_ == nullptr)
        return -100;

    yolo_attention_debug_log(
        "[yolo_attention][cpu] input q=%d(%d,%d,%d,%d) k=%d(%d,%d,%d,%d) v=%d(%d,%d,%d,%d)\n",
        bottom_blobs[0].dims,
        bottom_blobs[0].w,
        bottom_blobs[0].h,
        bottom_blobs[0].d,
        bottom_blobs[0].c,
        bottom_blobs[1].dims,
        bottom_blobs[1].w,
        bottom_blobs[1].h,
        bottom_blobs[1].d,
        bottom_blobs[1].c,
        bottom_blobs[2].dims,
        bottom_blobs[2].w,
        bottom_blobs[2].h,
        bottom_blobs[2].d,
        bottom_blobs[2].c);

    std::vector<ncnn::Mat> query_blobs(1);
    std::vector<ncnn::Mat> key_blobs(1);
    std::vector<ncnn::Mat> value_blobs(1);
    std::vector<ncnn::Mat> attention_inputs(3);
    std::vector<ncnn::Mat> attention_outputs(1);
    if (top_blobs.empty())
        top_blobs.resize(1);

    int ret = permute_cpu_->forward(bottom_blobs[0], query_blobs[0], opt);
    if (ret != 0)
        return ret;
    yolo_attention_debug_log(
        "[yolo_attention][cpu] permute q -> %d(%d,%d,%d,%d)\n",
        query_blobs[0].dims,
        query_blobs[0].w,
        query_blobs[0].h,
        query_blobs[0].d,
        query_blobs[0].c);

    ret = permute_cpu_->forward(bottom_blobs[1], key_blobs[0], opt);
    if (ret != 0)
        return ret;
    yolo_attention_debug_log(
        "[yolo_attention][cpu] permute k -> %d(%d,%d,%d,%d)\n",
        key_blobs[0].dims,
        key_blobs[0].w,
        key_blobs[0].h,
        key_blobs[0].d,
        key_blobs[0].c);

    ret = permute_cpu_->forward(bottom_blobs[2], value_blobs[0], opt);
    if (ret != 0)
        return ret;
    yolo_attention_debug_log(
        "[yolo_attention][cpu] permute v -> %d(%d,%d,%d,%d)\n",
        value_blobs[0].dims,
        value_blobs[0].w,
        value_blobs[0].h,
        value_blobs[0].d,
        value_blobs[0].c);

    attention_inputs[0] = query_blobs[0];
    attention_inputs[1] = key_blobs[0];
    attention_inputs[2] = value_blobs[0];
    ret = sdpa_cpu_->forward(attention_inputs, attention_outputs, opt);
    if (ret != 0)
        return ret;
    yolo_attention_debug_log(
        "[yolo_attention][cpu] sdpa -> %d(%d,%d,%d,%d)\n",
        attention_outputs[0].dims,
        attention_outputs[0].w,
        attention_outputs[0].h,
        attention_outputs[0].d,
        attention_outputs[0].c);

    return permute_cpu_->forward(attention_outputs[0], top_blobs[0], opt);
}

#if NCNN_VULKAN
int LadaYoloAttentionLayer::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    if (permute_vulkan_ != nullptr)
    {
        int ret = permute_vulkan_->upload_model(cmd, opt);
        if (ret != 0)
            return ret;
    }

    if (sdpa_vulkan_ != nullptr)
    {
        int ret = sdpa_vulkan_->upload_model(cmd, make_safe_sdpa_vulkan_option(opt));
        if (ret != 0)
            return ret;
    }

    return 0;
}

int LadaYoloAttentionLayer::forward(
    const std::vector<ncnn::VkMat>& bottom_blobs,
    std::vector<ncnn::VkMat>& top_blobs,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 3 || permute_vulkan_ == nullptr || sdpa_vulkan_ == nullptr)
        return -100;

    yolo_attention_debug_log(
        "[yolo_attention][vk] input q=%d(%d,%d,%d,%d) k=%d(%d,%d,%d,%d) v=%d(%d,%d,%d,%d)\n",
        bottom_blobs[0].dims,
        bottom_blobs[0].w,
        bottom_blobs[0].h,
        bottom_blobs[0].d,
        bottom_blobs[0].c,
        bottom_blobs[1].dims,
        bottom_blobs[1].w,
        bottom_blobs[1].h,
        bottom_blobs[1].d,
        bottom_blobs[1].c,
        bottom_blobs[2].dims,
        bottom_blobs[2].w,
        bottom_blobs[2].h,
        bottom_blobs[2].d,
        bottom_blobs[2].c);

    std::vector<ncnn::VkMat> query_blobs(1);
    std::vector<ncnn::VkMat> key_blobs(1);
    std::vector<ncnn::VkMat> value_blobs(1);
    std::vector<ncnn::VkMat> attention_inputs(3);
    std::vector<ncnn::VkMat> attention_outputs(1);
    if (top_blobs.empty())
        top_blobs.resize(1);

    int ret = permute_vulkan_->forward(bottom_blobs[0], query_blobs[0], cmd, opt);
    if (ret != 0)
        return ret;
    yolo_attention_debug_log(
        "[yolo_attention][vk] permute q -> %d(%d,%d,%d,%d)\n",
        query_blobs[0].dims,
        query_blobs[0].w,
        query_blobs[0].h,
        query_blobs[0].d,
        query_blobs[0].c);

    ret = permute_vulkan_->forward(bottom_blobs[1], key_blobs[0], cmd, opt);
    if (ret != 0)
        return ret;
    yolo_attention_debug_log(
        "[yolo_attention][vk] permute k -> %d(%d,%d,%d,%d)\n",
        key_blobs[0].dims,
        key_blobs[0].w,
        key_blobs[0].h,
        key_blobs[0].d,
        key_blobs[0].c);

    ret = permute_vulkan_->forward(bottom_blobs[2], value_blobs[0], cmd, opt);
    if (ret != 0)
        return ret;
    yolo_attention_debug_log(
        "[yolo_attention][vk] permute v -> %d(%d,%d,%d,%d)\n",
        value_blobs[0].dims,
        value_blobs[0].w,
        value_blobs[0].h,
        value_blobs[0].d,
        value_blobs[0].c);

    attention_inputs[0] = query_blobs[0];
    attention_inputs[1] = key_blobs[0];
    attention_inputs[2] = value_blobs[0];
    ret = sdpa_vulkan_->forward(
        attention_inputs,
        attention_outputs,
        cmd,
        make_safe_sdpa_vulkan_option(opt));
    if (ret != 0)
        return ret;
    yolo_attention_debug_log(
        "[yolo_attention][vk] sdpa -> %d(%d,%d,%d,%d)\n",
        attention_outputs[0].dims,
        attention_outputs[0].w,
        attention_outputs[0].h,
        attention_outputs[0].d,
        attention_outputs[0].c);

    return permute_vulkan_->forward(attention_outputs[0], top_blobs[0], cmd, opt);
}
#endif

ncnn::Layer* lada_yolo_attention_layer_creator(void* userdata)
{
    (void)userdata;
    return new LadaYoloAttentionLayer;
}

void lada_yolo_attention_layer_destroyer(ncnn::Layer* layer, void* userdata)
{
    (void)userdata;
    delete layer;
}

int register_lada_yolo_attention_layers(ncnn::Net& net)
{
    int ret = net.register_custom_layer(
        "lada.YoloAttention",
        lada_yolo_attention_layer_creator,
        lada_yolo_attention_layer_destroyer);
    if (ret != 0)
        return ret;

    return net.register_custom_layer(
        "pnnx.custom_op.lada.YoloAttention",
        lada_yolo_attention_layer_creator,
        lada_yolo_attention_layer_destroyer);
}

} // namespace lada
