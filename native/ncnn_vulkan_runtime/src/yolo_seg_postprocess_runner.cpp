// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#include "yolo_seg_postprocess_runner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace lada {

namespace {

struct PredictionAccessor
{
    const float* data = nullptr;
    int dim0 = 0;
    int dim1 = 0;
    bool features_first = true;

    float at(int feature_index, int box_index) const
    {
        if (features_first) {
            return data[static_cast<std::size_t>(feature_index) * dim1 + box_index];
        }
        return data[static_cast<std::size_t>(box_index) * dim1 + feature_index];
    }

    int num_boxes() const
    {
        return features_first ? dim1 : dim0;
    }
};

struct ProtoAccessor
{
    const float* data = nullptr;
    int channels = 0;
    int height = 0;
    int width = 0;
    bool chw_layout = true;

    float at(int channel, int y, int x) const
    {
        if (chw_layout) {
            return data[(static_cast<std::size_t>(channel) * height + y) * width + x];
        }
        return data[(static_cast<std::size_t>(y) * width + x) * channels + channel];
    }
};

struct Candidate
{
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
    float confidence = 0.f;
    int class_id = 0;
    int source_index = 0;
    std::vector<float> mask_coeffs;
};

struct TensorLayout
{
    bool proto_is_chw = true;
    int mask_dim = 0;
    int expected_features = 0;
};

float clamp_float(float value, float lower, float upper)
{
    return std::max(lower, std::min(value, upper));
}

int python_round_to_int(float value)
{
    // Python's round() uses ties-to-even; std::round() does not.
    return static_cast<int>(std::nearbyint(static_cast<double>(value)));
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
    layout.expected_features = 4 + num_classes + layout.mask_dim;
    return layout;
}

PredictionAccessor make_prediction_accessor(
    const float* pred_data,
    int pred_dim0,
    int pred_dim1,
    int expected_features)
{
    if (pred_dim0 == expected_features) {
        return PredictionAccessor{pred_data, pred_dim0, pred_dim1, true};
    }
    if (pred_dim1 == expected_features) {
        return PredictionAccessor{pred_data, pred_dim0, pred_dim1, false};
    }
    throw std::runtime_error("Unexpected YOLO prediction tensor shape.");
}

ProtoAccessor make_proto_accessor(
    const float* proto_data,
    int proto_dim0,
    int proto_dim1,
    int proto_dim2,
    int mask_dim,
    bool chw_layout)
{
    if (proto_dim0 > 0 && proto_dim1 > 0 && proto_dim2 > 0) {
        if (chw_layout) {
            return ProtoAccessor{proto_data, mask_dim, proto_dim1, proto_dim2, true};
        }
        return ProtoAccessor{proto_data, mask_dim, proto_dim0, proto_dim1, false};
    }
    throw std::runtime_error("Unexpected YOLO proto tensor shape.");
}

std::vector<Candidate> collect_candidates(
    const PredictionAccessor& pred,
    int num_classes,
    int mask_dim,
    const YoloSegPostprocessConfig& config)
{
    std::vector<Candidate> candidates;
    candidates.reserve(pred.num_boxes());

    for (int box_index = 0; box_index < pred.num_boxes(); ++box_index) {
        float best_confidence = 0.f;
        int best_class = 0;
        for (int class_index = 0; class_index < num_classes; ++class_index) {
            const float class_score = pred.at(4 + class_index, box_index);
            if (class_score > best_confidence) {
                best_confidence = class_score;
                best_class = class_index;
            }
        }
        const bool class_allowed = config.allowed_classes.empty()
            || (
                best_class >= 0
                && best_class < static_cast<int>(config.allowed_classes.size())
                && config.allowed_classes[static_cast<std::size_t>(best_class)] != 0);
        if (best_confidence <= config.conf_threshold || !class_allowed) {
            continue;
        }

        const float center_x = pred.at(0, box_index);
        const float center_y = pred.at(1, box_index);
        const float width = pred.at(2, box_index);
        const float height = pred.at(3, box_index);

        Candidate candidate;
        candidate.x1 = center_x - width * 0.5f;
        candidate.y1 = center_y - height * 0.5f;
        candidate.x2 = center_x + width * 0.5f;
        candidate.y2 = center_y + height * 0.5f;
        candidate.confidence = best_confidence;
        candidate.class_id = best_class;
        candidate.source_index = box_index;
        candidate.mask_coeffs.resize(mask_dim);
        for (int mask_index = 0; mask_index < mask_dim; ++mask_index) {
            candidate.mask_coeffs[mask_index] = pred.at(4 + num_classes + mask_index, box_index);
        }
        candidates.push_back(std::move(candidate));
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            if (left.confidence == right.confidence) {
                return left.source_index < right.source_index;
            }
            return left.confidence > right.confidence;
        });
    if (config.max_nms <= 0) {
        candidates.clear();
    } else if (static_cast<int>(candidates.size()) > config.max_nms) {
        candidates.resize(static_cast<std::size_t>(config.max_nms));
    }
    return candidates;
}

float intersection_over_union(const Candidate& left, const Candidate& right)
{
    const float inter_x1 = std::max(left.x1, right.x1);
    const float inter_y1 = std::max(left.y1, right.y1);
    const float inter_x2 = std::min(left.x2, right.x2);
    const float inter_y2 = std::min(left.y2, right.y2);

    const float inter_width = std::max(inter_x2 - inter_x1, 0.f);
    const float inter_height = std::max(inter_y2 - inter_y1, 0.f);
    const float inter_area = inter_width * inter_height;
    if (inter_area <= 0.f) {
        return 0.f;
    }

    const float left_area = std::max(left.x2 - left.x1, 0.f) * std::max(left.y2 - left.y1, 0.f);
    const float right_area = std::max(right.x2 - right.x1, 0.f) * std::max(right.y2 - right.y1, 0.f);
    const float union_area = left_area + right_area - inter_area;
    if (union_area <= 0.f) {
        return 0.f;
    }
    return inter_area / union_area;
}

std::vector<Candidate> apply_nms(
    const std::vector<Candidate>& candidates,
    float iou_threshold,
    int max_det,
    bool agnostic_nms)
{
    std::vector<Candidate> kept;
    kept.reserve(std::min<int>(static_cast<int>(candidates.size()), max_det));

    for (const Candidate& candidate : candidates) {
        bool suppressed = false;
        for (const Candidate& accepted : kept) {
            if (!agnostic_nms && accepted.class_id != candidate.class_id) {
                continue;
            }
            if (intersection_over_union(candidate, accepted) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) {
            continue;
        }
        kept.push_back(candidate);
        if (static_cast<int>(kept.size()) >= max_det) {
            break;
        }
    }

    return kept;
}

std::array<float, 4> scale_box(
    const Candidate& candidate,
    const YoloSegPostprocessConfig& config)
{
    const float gain = std::min(
        static_cast<float>(config.input_height) / static_cast<float>(config.orig_height),
        static_cast<float>(config.input_width) / static_cast<float>(config.orig_width));
    const float pad_x = static_cast<float>(python_round_to_int(
        (static_cast<float>(config.input_width) - static_cast<float>(config.orig_width) * gain) * 0.5f - 0.1f));
    const float pad_y = static_cast<float>(python_round_to_int(
        (static_cast<float>(config.input_height) - static_cast<float>(config.orig_height) * gain) * 0.5f - 0.1f));

    std::array<float, 4> box = {
        (candidate.x1 - pad_x) / gain,
        (candidate.y1 - pad_y) / gain,
        (candidate.x2 - pad_x) / gain,
        (candidate.y2 - pad_y) / gain,
    };
    box[0] = clamp_float(box[0], 0.f, static_cast<float>(config.orig_width));
    box[1] = clamp_float(box[1], 0.f, static_cast<float>(config.orig_height));
    box[2] = clamp_float(box[2], 0.f, static_cast<float>(config.orig_width));
    box[3] = clamp_float(box[3], 0.f, static_cast<float>(config.orig_height));
    return box;
}

std::vector<float> decode_mask(
    const Candidate& candidate,
    const ProtoAccessor& proto)
{
    std::vector<float> mask(static_cast<std::size_t>(proto.height) * proto.width, 0.f);
    for (int channel = 0; channel < proto.channels; ++channel) {
        const float coeff = candidate.mask_coeffs[channel];
        for (int y = 0; y < proto.height; ++y) {
            for (int x = 0; x < proto.width; ++x) {
                mask[static_cast<std::size_t>(y) * proto.width + x] += coeff * proto.at(channel, y, x);
            }
        }
    }
    return mask;
}

void crop_mask_inplace(
    std::vector<float>& mask,
    int mask_height,
    int mask_width,
    const Candidate& candidate,
    const YoloSegPostprocessConfig& config)
{
    const float width_ratio = static_cast<float>(mask_width) / static_cast<float>(config.input_width);
    const float height_ratio = static_cast<float>(mask_height) / static_cast<float>(config.input_height);

    const float x1 = candidate.x1 * width_ratio;
    const float y1 = candidate.y1 * height_ratio;
    const float x2 = candidate.x2 * width_ratio;
    const float y2 = candidate.y2 * height_ratio;

    for (int y = 0; y < mask_height; ++y) {
        for (int x = 0; x < mask_width; ++x) {
            if (
                static_cast<float>(y) < y1
                || static_cast<float>(y) >= y2
                || static_cast<float>(x) < x1
                || static_cast<float>(x) >= x2) {
                mask[static_cast<std::size_t>(y) * mask_width + x] = 0.f;
            }
        }
    }
}

std::vector<float> resize_bilinear_float(
    const std::vector<float>& source,
    int source_height,
    int source_width,
    int target_height,
    int target_width)
{
    if (source_height == target_height && source_width == target_width) {
        return source;
    }

    std::vector<float> resized(static_cast<std::size_t>(target_height) * target_width, 0.f);
    const float scale_y = static_cast<float>(source_height) / static_cast<float>(target_height);
    const float scale_x = static_cast<float>(source_width) / static_cast<float>(target_width);

    for (int y = 0; y < target_height; ++y) {
        const float source_y = std::max((y + 0.5f) * scale_y - 0.5f, 0.f);
        const int y0 = std::min(static_cast<int>(std::floor(source_y)), source_height - 1);
        const int y1 = std::min(y0 + 1, source_height - 1);
        const float ly = source_y - y0;
        const float hy = 1.f - ly;

        for (int x = 0; x < target_width; ++x) {
            const float source_x = std::max((x + 0.5f) * scale_x - 0.5f, 0.f);
            const int x0 = std::min(static_cast<int>(std::floor(source_x)), source_width - 1);
            const int x1 = std::min(x0 + 1, source_width - 1);
            const float lx = source_x - x0;
            const float hx = 1.f - lx;

            const float top =
                source[static_cast<std::size_t>(y0) * source_width + x0] * hx +
                source[static_cast<std::size_t>(y0) * source_width + x1] * lx;
            const float bottom =
                source[static_cast<std::size_t>(y1) * source_width + x0] * hx +
                source[static_cast<std::size_t>(y1) * source_width + x1] * lx;
            resized[static_cast<std::size_t>(y) * target_width + x] = top * hy + bottom * ly;
        }
    }

    return resized;
}

std::vector<unsigned char> threshold_mask(
    const std::vector<float>& mask,
    float threshold,
    unsigned char active_value)
{
    std::vector<unsigned char> thresholded(mask.size(), 0);
    for (std::size_t index = 0; index < mask.size(); ++index) {
        thresholded[index] = mask[index] > threshold ? active_value : static_cast<unsigned char>(0);
    }
    return thresholded;
}

bool mask_has_active_pixel(const std::vector<unsigned char>& mask)
{
    return std::any_of(mask.begin(), mask.end(), [](unsigned char value) { return value != 0; });
}

std::vector<unsigned char> scale_and_unpad_mask(
    const std::vector<unsigned char>& input_mask,
    const YoloSegPostprocessConfig& config)
{
    if (config.input_height == config.orig_height && config.input_width == config.orig_width) {
        return input_mask;
    }

    const float gain = std::min(
        static_cast<float>(config.input_height) / static_cast<float>(config.orig_height),
        static_cast<float>(config.input_width) / static_cast<float>(config.orig_width));
    const float pad_x = (static_cast<float>(config.input_width) - static_cast<float>(config.orig_width) * gain) * 0.5f;
    const float pad_y = (static_cast<float>(config.input_height) - static_cast<float>(config.orig_height) * gain) * 0.5f;

    const int top = std::max(0, python_round_to_int(pad_y - 0.1f));
    const int left = std::max(0, python_round_to_int(pad_x - 0.1f));
    const int bottom = std::min(config.input_height, config.input_height - python_round_to_int(pad_y + 0.1f));
    const int right = std::min(config.input_width, config.input_width - python_round_to_int(pad_x + 0.1f));

    const int crop_height = std::max(bottom - top, 1);
    const int crop_width = std::max(right - left, 1);
    std::vector<float> cropped(static_cast<std::size_t>(crop_height) * crop_width, 0.f);
    for (int y = 0; y < crop_height; ++y) {
        for (int x = 0; x < crop_width; ++x) {
            const int source_y = std::min(top + y, config.input_height - 1);
            const int source_x = std::min(left + x, config.input_width - 1);
            cropped[static_cast<std::size_t>(y) * crop_width + x] = static_cast<float>(
                input_mask[static_cast<std::size_t>(source_y) * config.input_width + source_x]);
        }
    }

    const std::vector<float> resized = resize_bilinear_float(
        cropped,
        crop_height,
        crop_width,
        config.orig_height,
        config.orig_width);
    std::vector<unsigned char> final_mask(resized.size(), 0);
    for (std::size_t index = 0; index < resized.size(); ++index) {
        final_mask[index] = resized[index] > 127.f ? static_cast<unsigned char>(255) : static_cast<unsigned char>(0);
    }
    return final_mask;
}

} // namespace

YoloSegSelectionResult select_yolo_segmentation_cpu(
    const float* pred_data,
    int pred_dim0,
    int pred_dim1,
    int proto_dim0,
    int proto_dim1,
    int proto_dim2,
    const YoloSegPostprocessConfig& config)
{
    if (config.input_height <= 0 || config.input_width <= 0) {
        throw std::runtime_error("Input shape must be positive.");
    }
    if (config.orig_height <= 0 || config.orig_width <= 0) {
        throw std::runtime_error("Original image shape must be positive.");
    }
    if (config.num_classes <= 0) {
        throw std::runtime_error("YOLO postprocess requires at least one class.");
    }

    const TensorLayout layout = resolve_tensor_layout(
        pred_dim0,
        pred_dim1,
        proto_dim0,
        proto_dim2,
        config.num_classes);
    const PredictionAccessor pred = make_prediction_accessor(
        pred_data,
        pred_dim0,
        pred_dim1,
        layout.expected_features);

    std::vector<Candidate> candidates = collect_candidates(
        pred,
        config.num_classes,
        layout.mask_dim,
        config);
    std::vector<Candidate> detections = apply_nms(
        candidates,
        config.iou_threshold,
        config.max_det,
        config.agnostic_nms);

    YoloSegSelectionResult result;
    result.proto_is_chw = layout.proto_is_chw;
    result.mask_dim = layout.mask_dim;
    result.detections.reserve(detections.size());

    for (const Candidate& detection : detections) {
        YoloSegSelectedDetection output_detection;
        const std::array<float, 4> scaled_box = scale_box(detection, config);
        output_detection.box = {scaled_box[0], scaled_box[1], scaled_box[2], scaled_box[3], detection.confidence, static_cast<float>(detection.class_id)};
        output_detection.selected.reserve(static_cast<std::size_t>(6 + layout.mask_dim));
        output_detection.selected.push_back(detection.x1);
        output_detection.selected.push_back(detection.y1);
        output_detection.selected.push_back(detection.x2);
        output_detection.selected.push_back(detection.y2);
        output_detection.selected.push_back(detection.confidence);
        output_detection.selected.push_back(static_cast<float>(detection.class_id));
        output_detection.selected.insert(
            output_detection.selected.end(),
            detection.mask_coeffs.begin(),
            detection.mask_coeffs.end());
        result.detections.push_back(std::move(output_detection));
    }

    return result;
}

YoloSegPostprocessResult finalize_yolo_segmentation_cpu(
    const YoloSegSelectionResult& selection,
    const float* proto_data,
    int proto_dim0,
    int proto_dim1,
    int proto_dim2,
    const YoloSegPostprocessConfig& config)
{
    if (config.input_height <= 0 || config.input_width <= 0) {
        throw std::runtime_error("Input shape must be positive.");
    }
    if (config.orig_height <= 0 || config.orig_width <= 0) {
        throw std::runtime_error("Original image shape must be positive.");
    }
    if (selection.mask_dim <= 0) {
        throw std::runtime_error("YOLO selection result is missing mask coefficients.");
    }

    const ProtoAccessor proto = make_proto_accessor(
        proto_data,
        proto_dim0,
        proto_dim1,
        proto_dim2,
        selection.mask_dim,
        selection.proto_is_chw);

    YoloSegPostprocessResult result;
    result.mask_height = config.orig_height;
    result.mask_width = config.orig_width;
    result.detections.reserve(selection.detections.size());

    for (const YoloSegSelectedDetection& selected_detection : selection.detections) {
        if (selected_detection.selected.size() != static_cast<std::size_t>(6 + selection.mask_dim)) {
            throw std::runtime_error("YOLO selection row has an unexpected feature dimension.");
        }

        Candidate detection;
        detection.x1 = selected_detection.selected[0];
        detection.y1 = selected_detection.selected[1];
        detection.x2 = selected_detection.selected[2];
        detection.y2 = selected_detection.selected[3];
        detection.confidence = selected_detection.selected[4];
        detection.class_id = static_cast<int>(selected_detection.selected[5]);
        detection.mask_coeffs.assign(
            selected_detection.selected.begin() + 6,
            selected_detection.selected.end());

        std::vector<float> mask = decode_mask(detection, proto);
        crop_mask_inplace(mask, proto.height, proto.width, detection, config);
        const std::vector<float> upsampled_mask = resize_bilinear_float(
            mask,
            proto.height,
            proto.width,
            config.input_height,
            config.input_width);
        const std::vector<unsigned char> input_space_mask = threshold_mask(upsampled_mask, 0.f, 255);
        if (!mask_has_active_pixel(input_space_mask)) {
            continue;
        }

        YoloSegDetection output_detection;
        output_detection.box = selected_detection.box;
        output_detection.mask = scale_and_unpad_mask(input_space_mask, config);
        result.detections.push_back(std::move(output_detection));
    }

    return result;
}

YoloSegPostprocessResult postprocess_yolo_segmentation_cpu(
    const float* pred_data,
    int pred_dim0,
    int pred_dim1,
    const float* proto_data,
    int proto_dim0,
    int proto_dim1,
    int proto_dim2,
    const YoloSegPostprocessConfig& config)
{
    const YoloSegSelectionResult selection = select_yolo_segmentation_cpu(
        pred_data,
        pred_dim0,
        pred_dim1,
        proto_dim0,
        proto_dim1,
        proto_dim2,
        config);
    return finalize_yolo_segmentation_cpu(
        selection,
        proto_data,
        proto_dim0,
        proto_dim1,
        proto_dim2,
        config);
}

} // namespace lada
