// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#ifndef LADA_GRIDSAMPLE_LAYER_H
#define LADA_GRIDSAMPLE_LAYER_H

#include "layer.h"

namespace ncnn {
class Net;
}

namespace lada {

class LadaGridSampleLayer : public ncnn::Layer
{
public:
    LadaGridSampleLayer();

    int load_param(const ncnn::ParamDict& pd) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(
        const std::vector<ncnn::Mat>& bottom_blobs,
        std::vector<ncnn::Mat>& top_blobs,
        const ncnn::Option& opt) const override;
#if NCNN_VULKAN
    int forward(
        const std::vector<ncnn::VkMat>& bottom_blobs,
        std::vector<ncnn::VkMat>& top_blobs,
        ncnn::VkCompute& cmd,
        const ncnn::Option& opt) const override;
#endif

public:
    int sample_type;
    int padding_mode;
    int align_corner;
    int permute_fusion;

#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_gridsample;
#endif
};

ncnn::Layer* lada_gridsample_layer_creator(void* userdata);
void lada_gridsample_layer_destroyer(ncnn::Layer* layer, void* userdata);
int register_lada_gridsample_layers(ncnn::Net& net);

} // namespace lada

#endif // LADA_GRIDSAMPLE_LAYER_H
