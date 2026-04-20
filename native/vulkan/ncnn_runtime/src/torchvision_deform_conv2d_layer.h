// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#ifndef LADA_TORCHVISION_DEFORM_CONV2D_LAYER_H
#define LADA_TORCHVISION_DEFORM_CONV2D_LAYER_H

#include "layer.h"

namespace ncnn {
class Net;
}

namespace lada {

class TorchVisionDeformConv2DLayer : public ncnn::Layer
{
public:
    TorchVisionDeformConv2DLayer();

    int load_param(const ncnn::ParamDict& pd) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const override;
#if NCNN_VULKAN
    int forward(
        const std::vector<ncnn::VkMat>& bottom_blobs,
        std::vector<ncnn::VkMat>& top_blobs,
        ncnn::VkCompute& cmd,
        const ncnn::Option& opt) const override;
#endif

public:
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int dilation_h;
    int dilation_w;
    int groups;
    int deform_groups;
    int use_mask;

#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_deformconv;
#endif
};

ncnn::Layer* torchvision_deform_conv2d_layer_creator(void* userdata);
void torchvision_deform_conv2d_layer_destroyer(ncnn::Layer* layer, void* userdata);
int register_torchvision_deform_conv2d_layers(ncnn::Net& net);

} // namespace lada

#endif // LADA_TORCHVISION_DEFORM_CONV2D_LAYER_H
