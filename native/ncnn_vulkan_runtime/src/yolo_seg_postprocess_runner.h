// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#ifndef LADA_YOLO_SEG_POSTPROCESS_RUNNER_H
#define LADA_YOLO_SEG_POSTPROCESS_RUNNER_H

#include <array>
#include <vector>

namespace lada {

struct YoloSegPostprocessConfig
{
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int max_det = 300;
    int num_classes = 1;
    bool agnostic_nms = false;
    int input_height = 0;
    int input_width = 0;
    int orig_height = 0;
    int orig_width = 0;
};

struct YoloSegDetection
{
    std::array<float, 6> box{};
    std::vector<unsigned char> mask;
};

struct YoloSegPostprocessResult
{
    int mask_height = 0;
    int mask_width = 0;
    std::vector<YoloSegDetection> detections;
};

struct YoloSegSelectedDetection
{
    std::array<float, 6> box{};
    std::vector<float> selected;
};

struct YoloSegSelectionResult
{
    bool proto_is_chw = true;
    int mask_dim = 0;
    std::vector<YoloSegSelectedDetection> detections;
};

YoloSegSelectionResult select_yolo_segmentation_cpu(
    const float* pred_data,
    int pred_dim0,
    int pred_dim1,
    int proto_dim0,
    int proto_dim1,
    int proto_dim2,
    const YoloSegPostprocessConfig& config);

YoloSegPostprocessResult finalize_yolo_segmentation_cpu(
    const YoloSegSelectionResult& selection,
    const float* proto_data,
    int proto_dim0,
    int proto_dim1,
    int proto_dim2,
    const YoloSegPostprocessConfig& config);

YoloSegPostprocessResult postprocess_yolo_segmentation_cpu(
    const float* pred_data,
    int pred_dim0,
    int pred_dim1,
    const float* proto_data,
    int proto_dim0,
    int proto_dim1,
    int proto_dim2,
    const YoloSegPostprocessConfig& config);

} // namespace lada

#endif // LADA_YOLO_SEG_POSTPROCESS_RUNNER_H
