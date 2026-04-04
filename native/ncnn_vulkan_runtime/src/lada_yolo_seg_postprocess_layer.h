// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#ifndef LADA_YOLO_SEG_POSTPROCESS_LAYER_H
#define LADA_YOLO_SEG_POSTPROCESS_LAYER_H

#include "layer.h"

namespace ncnn {
class Net;
}

namespace lada {

class LadaYoloSegPostprocessLayer : public ncnn::Layer
{
public:
    LadaYoloSegPostprocessLayer();

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
    int max_det;
    int num_classes;

#if NCNN_VULKAN
public:
    ncnn::Pipeline* pipeline_candidate_decode;
    ncnn::Pipeline* pipeline_select_nms;
#endif
};

ncnn::Layer* lada_yolo_seg_postprocess_layer_creator(void* userdata);
void lada_yolo_seg_postprocess_layer_destroyer(ncnn::Layer* layer, void* userdata);
int register_lada_yolo_seg_postprocess_layers(ncnn::Net& net);

} // namespace lada

#endif // LADA_YOLO_SEG_POSTPROCESS_LAYER_H
