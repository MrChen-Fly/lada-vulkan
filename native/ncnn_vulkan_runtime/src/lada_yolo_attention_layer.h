// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#ifndef LADA_YOLO_ATTENTION_LAYER_H
#define LADA_YOLO_ATTENTION_LAYER_H

#include "layer.h"

namespace ncnn {
class Net;
}

namespace lada {

class LadaYoloAttentionLayer : public ncnn::Layer
{
public:
    LadaYoloAttentionLayer();
    ~LadaYoloAttentionLayer() override;

    int load_param(const ncnn::ParamDict& pd) override;
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
    float scale_;
    ncnn::Layer* permute_cpu_;
    ncnn::Layer* sdpa_cpu_;

#if NCNN_VULKAN
    ncnn::Layer* permute_vulkan_;
    ncnn::Layer* sdpa_vulkan_;
#endif
};

ncnn::Layer* lada_yolo_attention_layer_creator(void* userdata);
void lada_yolo_attention_layer_destroyer(ncnn::Layer* layer, void* userdata);
int register_lada_yolo_attention_layers(ncnn::Net& net);

} // namespace lada

#endif // LADA_YOLO_ATTENTION_LAYER_H

