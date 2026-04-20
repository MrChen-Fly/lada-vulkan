// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#include "lada_yolo_seg_postprocess_layer.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "net.h"
#include "yolo_seg_postprocess_runner.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif

namespace lada {

namespace {

struct RuntimeConfig
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

struct TensorLayout
{
    bool proto_is_chw = true;
    int mask_dim = 0;
    bool pred_features_first = true;
    int num_boxes = 0;
    int candidate_feature_dim = 0;
};

RuntimeConfig parse_runtime_config(const ncnn::Mat& config_blob)
{
    if (config_blob.dims != 1 || config_blob.w < 9) {
        throw std::runtime_error("YOLO segmentation config tensor must be a 1D tensor with 9 values.");
    }
    if (config_blob.elempack != 1 || config_blob.elemsize != sizeof(float)) {
        throw std::runtime_error("YOLO segmentation config tensor must be unpacked float32.");
    }

    const float* values = static_cast<const float*>(config_blob.data);
    RuntimeConfig config;
    config.conf_threshold = values[0];
    config.iou_threshold = values[1];
    config.max_det = std::max(static_cast<int>(values[2]), 0);
    config.num_classes = std::max(static_cast<int>(values[3]), 1);
    config.agnostic_nms = values[4] != 0.f;
    config.input_height = std::max(static_cast<int>(values[5]), 0);
    config.input_width = std::max(static_cast<int>(values[6]), 0);
    config.orig_height = std::max(static_cast<int>(values[7]), 0);
    config.orig_width = std::max(static_cast<int>(values[8]), 0);
    return config;
}

void validate_tensor(const ncnn::Mat& blob, int expected_dims, const char* name)
{
    if (blob.dims != expected_dims) {
        throw std::runtime_error(std::string(name) + " tensor has unexpected rank.");
    }
    if (blob.elempack != 1 || blob.elemsize != sizeof(float)) {
        throw std::runtime_error(std::string(name) + " tensor must be unpacked float32.");
    }
}

TensorLayout resolve_tensor_layout(
    int pred_dim0,
    int pred_dim1,
    int proto_dim0,
    int proto_dim2,
    int num_classes)
{
    const int expected_features_chw = 4 + num_classes + proto_dim0;
    const int expected_features_hwc = 4 + num_classes + proto_dim2;
    const bool pred_matches_chw = pred_dim0 == expected_features_chw || pred_dim1 == expected_features_chw;
    const bool pred_matches_hwc = pred_dim0 == expected_features_hwc || pred_dim1 == expected_features_hwc;

    TensorLayout layout;
    layout.proto_is_chw = pred_matches_chw || !pred_matches_hwc;
    layout.mask_dim = layout.proto_is_chw ? proto_dim0 : proto_dim2;
    layout.candidate_feature_dim = 6 + layout.mask_dim;
    layout.pred_features_first = pred_dim0 == (4 + num_classes + layout.mask_dim);
    layout.num_boxes = layout.pred_features_first ? pred_dim1 : pred_dim0;
    return layout;
}

int output_capacity_for(int max_det)
{
    return std::max(max_det, 1);
}

int pack_selection_outputs(
    const YoloSegSelectionResult& selection,
    int max_det,
    ncnn::Mat& boxes_blob,
    ncnn::Mat& selected_blob,
    ncnn::Mat& count_blob,
    const ncnn::Option& opt)
{
    const int output_capacity = output_capacity_for(max_det);
    const int selected_feature_dim = 6 + selection.mask_dim;

    boxes_blob.create(6, output_capacity, sizeof(float), opt.blob_allocator);
    selected_blob.create(selected_feature_dim, output_capacity, sizeof(float), opt.blob_allocator);
    count_blob.create(1, sizeof(float), opt.blob_allocator);
    if (boxes_blob.empty() || selected_blob.empty() || count_blob.empty()) {
        return -100;
    }

    std::memset(boxes_blob.data, 0, static_cast<std::size_t>(boxes_blob.total()) * sizeof(float));
    std::memset(selected_blob.data, 0, static_cast<std::size_t>(selected_blob.total()) * sizeof(float));
    static_cast<float*>(count_blob.data)[0] = static_cast<float>(selection.detections.size());

    const int copy_count = std::min<int>(static_cast<int>(selection.detections.size()), output_capacity);
    for (int detection_index = 0; detection_index < copy_count; ++detection_index) {
        const YoloSegSelectedDetection& detection = selection.detections[static_cast<std::size_t>(detection_index)];
        float* box_ptr = boxes_blob.row(detection_index);
        for (int feature_index = 0; feature_index < 6; ++feature_index) {
            box_ptr[feature_index] = detection.box[static_cast<std::size_t>(feature_index)];
        }

        float* selected_ptr = selected_blob.row(detection_index);
        for (std::size_t feature_index = 0; feature_index < detection.selected.size(); ++feature_index) {
            selected_ptr[feature_index] = detection.selected[feature_index];
        }
    }

    return 0;
}

#if NCNN_VULKAN
static const char kLadaYoloSegDecodeShader[] = R"(
#version 450

layout(binding = 0) readonly buffer pred_blob { sfp pred_blob_data[]; };
layout(binding = 1) readonly buffer config_blob { sfp config_blob_data[]; };
layout(binding = 2) writeonly buffer decoded_blob { sfp decoded_blob_data[]; };

layout(push_constant) uniform parameter
{
    int pred_dim0;
    int pred_dim1;
    int pred_features_first;
    int num_boxes;
    int num_classes;
    int mask_dim;
    int candidate_feature_dim;
} p;

float pred_value_at(int feature_index, int box_index)
{
    if (p.pred_features_first != 0)
    {
        return float(buffer_ld1(pred_blob_data, feature_index * p.pred_dim1 + box_index));
    }
    return float(buffer_ld1(pred_blob_data, box_index * p.pred_dim1 + feature_index));
}

float config_value_at(int index)
{
    return float(buffer_ld1(config_blob_data, index));
}

void decoded_store(int box_index, int feature_index, float value)
{
    buffer_st1(decoded_blob_data, box_index * p.candidate_feature_dim + feature_index, afp(value));
}

void main()
{
    const int box_index = int(gl_GlobalInvocationID.x);
    if (box_index >= p.num_boxes)
        return;

    float best_confidence = 0.0;
    int best_class = 0;
    for (int class_index = 0; class_index < p.num_classes; class_index++)
    {
        const float class_score = pred_value_at(4 + class_index, box_index);
        if (class_score > best_confidence)
        {
            best_confidence = class_score;
            best_class = class_index;
        }
    }

    const float conf_threshold = config_value_at(0);
    if (best_confidence <= conf_threshold)
    {
        for (int feature_index = 0; feature_index < p.candidate_feature_dim; feature_index++)
        {
            decoded_store(box_index, feature_index, 0.0);
        }
        return;
    }

    const float center_x = pred_value_at(0, box_index);
    const float center_y = pred_value_at(1, box_index);
    const float width = pred_value_at(2, box_index);
    const float height = pred_value_at(3, box_index);

    decoded_store(box_index, 0, center_x - width * 0.5);
    decoded_store(box_index, 1, center_y - height * 0.5);
    decoded_store(box_index, 2, center_x + width * 0.5);
    decoded_store(box_index, 3, center_y + height * 0.5);
    decoded_store(box_index, 4, best_confidence);
    decoded_store(box_index, 5, float(best_class));

    for (int mask_index = 0; mask_index < p.mask_dim; mask_index++)
    {
        decoded_store(
            box_index,
            6 + mask_index,
            pred_value_at(4 + p.num_classes + mask_index, box_index));
    }
}
)";

static const char kLadaYoloSegSelectShader[] = R"(
#version 450

layout(binding = 0) readonly buffer decoded_blob { sfp decoded_blob_data[]; };
layout(binding = 1) readonly buffer config_blob { sfp config_blob_data[]; };
layout(binding = 2) buffer suppressed_blob { sfp suppressed_blob_data[]; };
layout(binding = 3) writeonly buffer boxes_blob { sfp boxes_blob_data[]; };
layout(binding = 4) writeonly buffer selected_blob { sfp selected_blob_data[]; };
layout(binding = 5) writeonly buffer count_blob { sfp count_blob_data[]; };

layout(push_constant) uniform parameter
{
    int num_boxes;
    int candidate_feature_dim;
    int output_capacity;
} p;

float decoded_value_at(int box_index, int feature_index)
{
    return float(buffer_ld1(decoded_blob_data, box_index * p.candidate_feature_dim + feature_index));
}

float config_value_at(int index)
{
    return float(buffer_ld1(config_blob_data, index));
}

void selected_store(int row, int feature, float value)
{
    buffer_st1(selected_blob_data, row * p.candidate_feature_dim + feature, afp(value));
}

void boxes_store(int row, int feature, float value)
{
    buffer_st1(boxes_blob_data, row * 6 + feature, afp(value));
}

float suppressed_value_at(int index)
{
    return float(buffer_ld1(suppressed_blob_data, index));
}

void suppressed_store(int index, float value)
{
    buffer_st1(suppressed_blob_data, index, afp(value));
}

float intersection_over_union(int left_index, int right_index)
{
    const float left_x1 = decoded_value_at(left_index, 0);
    const float left_y1 = decoded_value_at(left_index, 1);
    const float left_x2 = decoded_value_at(left_index, 2);
    const float left_y2 = decoded_value_at(left_index, 3);
    const float right_x1 = decoded_value_at(right_index, 0);
    const float right_y1 = decoded_value_at(right_index, 1);
    const float right_x2 = decoded_value_at(right_index, 2);
    const float right_y2 = decoded_value_at(right_index, 3);

    const float inter_x1 = max(left_x1, right_x1);
    const float inter_y1 = max(left_y1, right_y1);
    const float inter_x2 = min(left_x2, right_x2);
    const float inter_y2 = min(left_y2, right_y2);
    const float inter_width = max(inter_x2 - inter_x1, 0.0);
    const float inter_height = max(inter_y2 - inter_y1, 0.0);
    const float inter_area = inter_width * inter_height;
    if (inter_area <= 0.0)
        return 0.0;

    const float left_area = max(left_x2 - left_x1, 0.0) * max(left_y2 - left_y1, 0.0);
    const float right_area = max(right_x2 - right_x1, 0.0) * max(right_y2 - right_y1, 0.0);
    const float union_area = left_area + right_area - inter_area;
    if (union_area <= 0.0)
        return 0.0;
    return inter_area / union_area;
}

void main()
{
    if (gl_GlobalInvocationID.x != 0 || gl_GlobalInvocationID.y != 0 || gl_GlobalInvocationID.z != 0)
        return;

    const float conf_threshold = config_value_at(0);
    const float iou_threshold = config_value_at(1);
    const bool agnostic_nms = config_value_at(4) != 0.0;
    const float input_height = config_value_at(5);
    const float input_width = config_value_at(6);
    const float orig_height = config_value_at(7);
    const float orig_width = config_value_at(8);

    for (int box_index = 0; box_index < p.num_boxes; box_index++)
    {
        suppressed_store(box_index, 0.0);
    }
    for (int row = 0; row < p.output_capacity; row++)
    {
        for (int feature = 0; feature < 6; feature++)
        {
            boxes_store(row, feature, 0.0);
        }
        for (int feature = 0; feature < p.candidate_feature_dim; feature++)
        {
            selected_store(row, feature, 0.0);
        }
    }
    buffer_st1(count_blob_data, 0, afp(0.0));

    const float gain = min(input_height / orig_height, input_width / orig_width);
    const float pad_x = float(int(roundEven((input_width - orig_width * gain) * 0.5 - 0.1)));
    const float pad_y = float(int(roundEven((input_height - orig_height * gain) * 0.5 - 0.1)));

    int kept_count = 0;
    for (int det_index = 0; det_index < p.output_capacity; det_index++)
    {
        int best_index = -1;
        float best_confidence = conf_threshold;
        for (int box_index = 0; box_index < p.num_boxes; box_index++)
        {
            if (suppressed_value_at(box_index) != 0.0)
                continue;

            const float confidence = decoded_value_at(box_index, 4);
            if (confidence <= best_confidence)
                continue;

            best_confidence = confidence;
            best_index = box_index;
        }

        if (best_index < 0)
            break;

        const float class_id = decoded_value_at(best_index, 5);
        const float x1 = decoded_value_at(best_index, 0);
        const float y1 = decoded_value_at(best_index, 1);
        const float x2 = decoded_value_at(best_index, 2);
        const float y2 = decoded_value_at(best_index, 3);

        for (int feature = 0; feature < p.candidate_feature_dim; feature++)
        {
            selected_store(det_index, feature, decoded_value_at(best_index, feature));
        }

        boxes_store(det_index, 0, clamp((x1 - pad_x) / gain, 0.0, orig_width));
        boxes_store(det_index, 1, clamp((y1 - pad_y) / gain, 0.0, orig_height));
        boxes_store(det_index, 2, clamp((x2 - pad_x) / gain, 0.0, orig_width));
        boxes_store(det_index, 3, clamp((y2 - pad_y) / gain, 0.0, orig_height));
        boxes_store(det_index, 4, best_confidence);
        boxes_store(det_index, 5, class_id);
        kept_count = det_index + 1;

        suppressed_store(best_index, 1.0);
        for (int box_index = 0; box_index < p.num_boxes; box_index++)
        {
            if (suppressed_value_at(box_index) != 0.0)
                continue;

            const float confidence = decoded_value_at(box_index, 4);
            if (confidence <= conf_threshold)
                continue;
            if (!agnostic_nms && decoded_value_at(box_index, 5) != class_id)
                continue;
            if (intersection_over_union(box_index, best_index) > iou_threshold)
            {
                suppressed_store(box_index, 1.0);
            }
        }
    }

    buffer_st1(count_blob_data, 0, afp(float(kept_count)));
}
)";
#endif

} // namespace

LadaYoloSegPostprocessLayer::LadaYoloSegPostprocessLayer()
{
    one_blob_only = false;
    support_inplace = false;
    max_det = 300;
    num_classes = 1;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
    pipeline_candidate_decode = 0;
    pipeline_select_nms = 0;
#else
    support_vulkan = false;
#endif
}

int LadaYoloSegPostprocessLayer::load_param(const ncnn::ParamDict& pd)
{
    max_det = std::max(pd.get(0, 300), 0);
    num_classes = std::max(pd.get(1, 1), 1);
    return 0;
}

int LadaYoloSegPostprocessLayer::create_pipeline(const ncnn::Option& opt)
{
#if NCNN_VULKAN
    if (!opt.use_vulkan_compute) {
        return 0;
    }

    std::vector<uint32_t> decode_spirv;
    int ret = ncnn::compile_spirv_module(
        kLadaYoloSegDecodeShader,
        static_cast<int>(sizeof(kLadaYoloSegDecodeShader) - 1),
        opt,
        decode_spirv);
    if (ret != 0) {
        return ret;
    }

    pipeline_candidate_decode = new ncnn::Pipeline(vkdev);
    pipeline_candidate_decode->set_optimal_local_size_xyz(64, 1, 1);
    ret = pipeline_candidate_decode->create(
        decode_spirv.data(),
        decode_spirv.size() * sizeof(uint32_t),
        std::vector<ncnn::vk_specialization_type>());
    if (ret != 0) {
        return ret;
    }

    std::vector<uint32_t> select_spirv;
    ret = ncnn::compile_spirv_module(
        kLadaYoloSegSelectShader,
        static_cast<int>(sizeof(kLadaYoloSegSelectShader) - 1),
        opt,
        select_spirv);
    if (ret != 0) {
        return ret;
    }

    pipeline_select_nms = new ncnn::Pipeline(vkdev);
    pipeline_select_nms->set_optimal_local_size_xyz(1, 1, 1);
    return pipeline_select_nms->create(
        select_spirv.data(),
        select_spirv.size() * sizeof(uint32_t),
        std::vector<ncnn::vk_specialization_type>());
#else
    (void)opt;
    return 0;
#endif
}

int LadaYoloSegPostprocessLayer::destroy_pipeline(const ncnn::Option& opt)
{
#if NCNN_VULKAN
    (void)opt;
    delete pipeline_candidate_decode;
    delete pipeline_select_nms;
    pipeline_candidate_decode = 0;
    pipeline_select_nms = 0;
#else
    (void)opt;
#endif
    return 0;
}

int LadaYoloSegPostprocessLayer::forward(
    const std::vector<ncnn::Mat>& bottom_blobs,
    std::vector<ncnn::Mat>& top_blobs,
    const ncnn::Option& opt) const
{
    (void)opt;

    if (bottom_blobs.size() != 3) {
        return -100;
    }

    try {
        const ncnn::Mat& pred = bottom_blobs[0];
        const ncnn::Mat& proto = bottom_blobs[1];
        const ncnn::Mat& config_blob = bottom_blobs[2];
        validate_tensor(pred, 2, "prediction");
        validate_tensor(proto, 3, "proto");

        RuntimeConfig runtime_config = parse_runtime_config(config_blob);
        runtime_config.max_det = max_det;
        runtime_config.num_classes = num_classes;

        YoloSegPostprocessConfig config;
        config.conf_threshold = runtime_config.conf_threshold;
        config.iou_threshold = runtime_config.iou_threshold;
        config.max_det = runtime_config.max_det;
        config.num_classes = runtime_config.num_classes;
        config.agnostic_nms = runtime_config.agnostic_nms;
        config.input_height = runtime_config.input_height;
        config.input_width = runtime_config.input_width;
        config.orig_height = runtime_config.orig_height;
        config.orig_width = runtime_config.orig_width;

        const YoloSegSelectionResult result = select_yolo_segmentation_cpu(
            static_cast<const float*>(pred.data),
            pred.h,
            pred.w,
            proto.c,
            proto.h,
            proto.w,
            config);

        if (top_blobs.size() < 3) {
            top_blobs.resize(3);
        }

        return pack_selection_outputs(
            result,
            max_det,
            top_blobs[0],
            top_blobs[1],
            top_blobs[2],
            opt);
    } catch (const std::exception&) {
        return -100;
    }
}

#if NCNN_VULKAN
int LadaYoloSegPostprocessLayer::forward(
    const std::vector<ncnn::VkMat>& bottom_blobs,
    std::vector<ncnn::VkMat>& top_blobs,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 3 || pipeline_candidate_decode == 0 || pipeline_select_nms == 0) {
        return -100;
    }

    const ncnn::VkMat& pred = bottom_blobs[0];
    const ncnn::VkMat& proto = bottom_blobs[1];
    const ncnn::VkMat& config_blob = bottom_blobs[2];
    if (pred.dims != 2 || proto.dims != 3 || config_blob.dims != 1) {
        return -100;
    }
    if (pred.elempack != 1 || proto.elempack != 1 || config_blob.elempack != 1) {
        return -100;
    }

    const TensorLayout layout = resolve_tensor_layout(
        pred.h,
        pred.w,
        proto.c,
        proto.w,
        num_classes);
    const int output_capacity = output_capacity_for(max_det);

    if (top_blobs.size() < 3) {
        top_blobs.resize(3);
    }

    ncnn::VkMat& boxes_blob = top_blobs[0];
    ncnn::VkMat& selected_blob = top_blobs[1];
    ncnn::VkMat& count_blob = top_blobs[2];
    boxes_blob.create(6, output_capacity, pred.elemsize, 1, opt.blob_vkallocator);
    selected_blob.create(layout.candidate_feature_dim, output_capacity, pred.elemsize, 1, opt.blob_vkallocator);
    count_blob.create(1, pred.elemsize, 1, opt.blob_vkallocator);
    if (boxes_blob.empty() || selected_blob.empty() || count_blob.empty()) {
        return -100;
    }

    ncnn::VkMat decoded_candidates;
    decoded_candidates.create(
        layout.candidate_feature_dim,
        layout.num_boxes,
        pred.elemsize,
        1,
        opt.workspace_vkallocator);
    ncnn::VkMat suppressed_flags;
    suppressed_flags.create(layout.num_boxes, pred.elemsize, 1, opt.workspace_vkallocator);
    if (decoded_candidates.empty() || suppressed_flags.empty()) {
        return -100;
    }

    {
        std::vector<ncnn::VkMat> bindings(3);
        bindings[0] = pred;
        bindings[1] = config_blob;
        bindings[2] = decoded_candidates;

        std::vector<ncnn::vk_constant_type> constants(7);
        constants[0].i = pred.h;
        constants[1].i = pred.w;
        constants[2].i = layout.pred_features_first ? 1 : 0;
        constants[3].i = layout.num_boxes;
        constants[4].i = num_classes;
        constants[5].i = layout.mask_dim;
        constants[6].i = layout.candidate_feature_dim;
        // Dispatch decode over the candidate axis only; the storage tensor stays row-major
        // as [num_boxes, candidate_feature_dim] for the downstream select shader.
        cmd.record_pipeline(pipeline_candidate_decode, bindings, constants, suppressed_flags);
    }

    {
        std::vector<ncnn::VkMat> bindings(6);
        bindings[0] = decoded_candidates;
        bindings[1] = config_blob;
        bindings[2] = suppressed_flags;
        bindings[3] = boxes_blob;
        bindings[4] = selected_blob;
        bindings[5] = count_blob;

        std::vector<ncnn::vk_constant_type> constants(3);
        constants[0].i = layout.num_boxes;
        constants[1].i = layout.candidate_feature_dim;
        constants[2].i = output_capacity;
        cmd.record_pipeline(pipeline_select_nms, bindings, constants, count_blob);
    }

    return 0;
}
#endif

ncnn::Layer* lada_yolo_seg_postprocess_layer_creator(void* userdata)
{
    (void)userdata;
    return new LadaYoloSegPostprocessLayer;
}

void lada_yolo_seg_postprocess_layer_destroyer(ncnn::Layer* layer, void* userdata)
{
    (void)userdata;
    delete layer;
}

int register_lada_yolo_seg_postprocess_layers(ncnn::Net& net)
{
    int ret = net.register_custom_layer(
        "lada.YoloSegPostprocess",
        lada_yolo_seg_postprocess_layer_creator,
        lada_yolo_seg_postprocess_layer_destroyer);
    if (ret != 0) {
        return ret;
    }

    return net.register_custom_layer(
        "pnnx.custom_op.lada.YoloSegPostprocess",
        lada_yolo_seg_postprocess_layer_creator,
        lada_yolo_seg_postprocess_layer_destroyer);
}

} // namespace lada
