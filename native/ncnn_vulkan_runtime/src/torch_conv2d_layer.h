// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#ifndef LADA_TORCH_CONV2D_LAYER_H
#define LADA_TORCH_CONV2D_LAYER_H

#include "layer.h"

namespace ncnn {
class Net;
class Pipeline;
}

namespace lada {

class TorchConv2DLayer : public ncnn::Layer
{
public:
    TorchConv2DLayer();
    ~TorchConv2DLayer() override;

    int load_param(const ncnn::ParamDict& pd) override;
    int load_model(const ncnn::ModelBin& mb) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(
        const std::vector<ncnn::Mat>& bottom_blobs,
        std::vector<ncnn::Mat>& top_blobs,
        const ncnn::Option& opt) const override;
#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override;
    int forward(
        const std::vector<ncnn::VkMat>& bottom_blobs,
        std::vector<ncnn::VkMat>& top_blobs,
        ncnn::VkCompute& cmd,
        const ncnn::Option& opt) const override;
#endif

private:
    bool needs_padding() const;

private:
    int num_output_;
    int kernel_w_;
    int kernel_h_;
    int dilation_w_;
    int dilation_h_;
    int stride_w_;
    int stride_h_;
    int pad_left_;
    int pad_right_;
    int pad_top_;
    int pad_bottom_;
    float pad_value_;
    int bias_term_;
    int weight_data_size_;
    int group_;
    int activation_type_;
    ncnn::Mat activation_params_;
    int pad_mode_;

    ncnn::Mat weight_data_;
    ncnn::Mat bias_data_;

    ncnn::Layer* padding_cpu_;
    ncnn::Layer* conv_cpu_;

#if NCNN_VULKAN
    ncnn::VkMat weight_data_gpu_;
    ncnn::VkMat bias_data_gpu_;
    ncnn::Pipeline* pipeline_vulkan_;
#endif
};

ncnn::Layer* torch_conv2d_layer_creator(void* userdata);
void torch_conv2d_layer_destroyer(ncnn::Layer* layer, void* userdata);
int register_torch_conv2d_layers(ncnn::Net& net);

} // namespace lada

#endif // LADA_TORCH_CONV2D_LAYER_H
