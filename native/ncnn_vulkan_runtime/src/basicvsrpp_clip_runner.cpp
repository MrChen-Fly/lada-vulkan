// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#include "basicvsrpp_clip_runner.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "command.h"
#include "gpu.h"
#include "lada_gridsample_layer.h"
#include "mat.h"
#include "native_op_profile.h"
#include "net.h"
#include "torchvision_deform_conv2d_layer.h"

namespace py = pybind11;

#if NCNN_VULKAN
namespace {

using ModuleValue = std::variant<ncnn::Mat, ncnn::VkMat>;
using NamedValue = std::pair<std::string, ModuleValue>;
using NamedValues = std::vector<NamedValue>;
using SteadyClock = std::chrono::steady_clock;

double elapsed_seconds(SteadyClock::time_point started_at);

bool clip_trace_enabled()
{
    static const bool enabled = []() {
        const char* value = std::getenv("LADA_NCNN_CLIP_TRACE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

void clip_trace(const char* format, ...)
{
    if (!clip_trace_enabled()) {
        return;
    }

    std::va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fflush(stderr);
}

struct ModuleArtifacts
{
    std::string param_path;
    std::string bin_path;
};

const char* kQuarterDownsample = "quarter_downsample";
const char* kFeatExtract = "feat_extract";
const char* kSpyNet = "spynet";
const char* kBackward1 = "backward_1";
const char* kForward1 = "forward_1";
const char* kBackward2 = "backward_2";
const char* kForward2 = "forward_2";
const char* kOutputFrame = "output_frame";

const std::vector<std::string> kRequiredModules = {
    kQuarterDownsample,
    kFeatExtract,
    kSpyNet,
    "backward_1_backbone",
    "forward_1_backbone",
    "backward_2_backbone",
    "forward_2_backbone",
    "backward_1_step",
    "forward_1_step",
    "backward_2_step",
    "forward_2_step",
    kOutputFrame,
};

#if NCNN_BENCHMARK
constexpr std::size_t kOutputFrameDownloadChunkSize = 2;
#else
constexpr std::size_t kOutputFrameDownloadChunkSize = 8;
#endif

static const char kBasicVsrppFramePreprocessShader[] = R"(
#version 450

layout(binding = 0) readonly buffer input_blob { uint input_blob_data[]; };
layout(binding = 1) writeonly buffer output_blob { sfp output_blob_data[]; };

layout(push_constant) uniform parameter
{
    int src_w;
    int src_h;
    int output_cstep;
} p;

uint load_input_byte(int byte_index)
{
    const int word_index = byte_index >> 2;
    const int byte_offset = (byte_index & 3) * 8;
    return (input_blob_data[word_index] >> uint(byte_offset)) & 255u;
}

float read_input_value(int y, int x, int channel)
{
    const int byte_index = ((y * p.src_w + x) * 3) + channel;
    return float(load_input_byte(byte_index)) / 255.0;
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int c = int(gl_GlobalInvocationID.z);
    if (x >= p.src_w || y >= p.src_h || c >= 3)
        return;

    const float value = read_input_value(y, x, c);
    buffer_st1(output_blob_data, c * p.output_cstep + y * p.src_w + x, afp(value));
}
)";

static const char kBasicVsrppResizePreprocessShader[] = R"(
#version 450

layout(binding = 0) readonly buffer input_blob { uint input_blob_data[]; };
layout(binding = 1) writeonly buffer output_blob { sfp output_blob_data[]; };

layout(push_constant) uniform parameter
{
    int src_w;
    int src_h;
    int dst_w;
    int dst_h;
    int resized_w;
    int resized_h;
    int pad_left;
    int pad_top;
    int pad_mode;
    int output_cstep;
} p;

uint load_input_byte(int byte_index)
{
    const int word_index = byte_index >> 2;
    const int byte_offset = (byte_index & 3) * 8;
    return (input_blob_data[word_index] >> uint(byte_offset)) & 255u;
}

float read_input_value(int y, int x, int channel)
{
    const int byte_index = ((y * p.src_w + x) * 3) + channel;
    return float(load_input_byte(byte_index));
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

float sample_resized_value(int y, int x, int channel)
{
    int inner_x = x - p.pad_left;
    int inner_y = y - p.pad_top;

    if (p.pad_mode == 0)
    {
        if (inner_x < 0 || inner_x >= p.resized_w || inner_y < 0 || inner_y >= p.resized_h)
            return 0.0;
    }
    else
    {
        inner_x = reflect_coord(inner_x, p.resized_w);
        inner_y = reflect_coord(inner_y, p.resized_h);
    }

    const float src_x = (float(inner_x) + 0.5) * float(p.src_w) / float(p.resized_w) - 0.5;
    const float src_y = (float(inner_y) + 0.5) * float(p.src_h) / float(p.resized_h) - 0.5;
    const int x0 = clamp(int(floor(src_x)), 0, p.src_w - 1);
    const int y0 = clamp(int(floor(src_y)), 0, p.src_h - 1);
    const int x1 = min(x0 + 1, p.src_w - 1);
    const int y1 = min(y0 + 1, p.src_h - 1);
    const float lx = src_x - float(x0);
    const float ly = src_y - float(y0);
    const float hx = 1.0 - lx;
    const float hy = 1.0 - ly;

    const float v00 = read_input_value(y0, x0, channel);
    const float v01 = read_input_value(y0, x1, channel);
    const float v10 = read_input_value(y1, x0, channel);
    const float v11 = read_input_value(y1, x1, channel);
    const float top = v00 * hx + v01 * lx;
    const float bottom = v10 * hx + v11 * lx;
    return (top * hy + bottom * ly) / 255.0;
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int c = int(gl_GlobalInvocationID.z);
    if (x >= p.dst_w || y >= p.dst_h || c >= 3)
        return;

    const float value = sample_resized_value(y, x, c);
    buffer_st1(output_blob_data, c * p.output_cstep + y * p.dst_w + x, afp(value));
}
)";

struct PackedBgrFrame
{
    int width = 0;
    int height = 0;
    ncnn::Mat packed_input;
};

struct PackedBgrResizeFrame
{
    int src_w = 0;
    int src_h = 0;
    int dst_w = 0;
    int dst_h = 0;
    int resized_w = 0;
    int resized_h = 0;
    int pad_left = 0;
    int pad_top = 0;
    int pad_mode = 0;
    ncnn::Mat packed_input;
};

ModuleArtifacts parse_module_artifacts(const py::handle& value)
{
    if (py::isinstance<py::dict>(value)) {
        const py::dict dict_value = py::reinterpret_borrow<py::dict>(value);
        return {
            py::cast<std::string>(dict_value["param_path"]),
            py::cast<std::string>(dict_value["bin_path"]),
        };
    }

    if (py::isinstance<py::tuple>(value) || py::isinstance<py::list>(value)) {
        const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
        if (py::len(sequence) != 2) {
            throw std::runtime_error(
                "BasicVsrppClipRunner module paths must be a pair of (param_path, bin_path).");
        }
        return {
            py::cast<std::string>(sequence[0]),
            py::cast<std::string>(sequence[1]),
        };
    }

    throw std::runtime_error(
        "BasicVsrppClipRunner module paths must be a dict or a pair of strings.");
}

std::unordered_map<std::string, ModuleArtifacts> parse_module_artifacts_map(const py::dict& module_paths)
{
    std::unordered_map<std::string, ModuleArtifacts> parsed;
    parsed.reserve(py::len(module_paths));
    for (const auto& item : module_paths) {
        parsed.emplace(
            py::cast<std::string>(item.first),
            parse_module_artifacts(item.second));
    }
    return parsed;
}

struct ModuleContext
{
    explicit ModuleContext(std::unique_ptr<ncnn::Net> owned_net)
        : net(std::move(owned_net))
    {
        vkdev = net->vulkan_device();
        if (vkdev == nullptr || !vkdev->is_valid()) {
            throw std::runtime_error("ncnn.Net does not expose a valid Vulkan device.");
        }

        blob_vkallocator = vkdev->acquire_blob_allocator();
        staging_vkallocator = vkdev->acquire_staging_allocator();
        if (blob_vkallocator == nullptr || staging_vkallocator == nullptr) {
            throw std::runtime_error("Failed to acquire ncnn Vulkan allocators.");
        }
    }

    ~ModuleContext()
    {
        if (vkdev != nullptr && blob_vkallocator != nullptr) {
            vkdev->reclaim_blob_allocator(blob_vkallocator);
        }
        if (vkdev != nullptr && staging_vkallocator != nullptr) {
            vkdev->reclaim_staging_allocator(staging_vkallocator);
        }
    }

    ncnn::Extractor create_extractor(bool light_mode) const
    {
        ncnn::Extractor extractor = net->create_extractor();
        extractor.set_light_mode(light_mode);
        extractor.set_blob_vkallocator(blob_vkallocator);
        extractor.set_workspace_vkallocator(blob_vkallocator);
        extractor.set_staging_vkallocator(staging_vkallocator);
        return extractor;
    }

    std::unique_ptr<ncnn::Net> net;
    const ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::VkAllocator* blob_vkallocator = nullptr;
    ncnn::VkAllocator* staging_vkallocator = nullptr;
};

class ModuleRunner
{
public:
    ModuleRunner(
        std::string module_name,
        const ModuleArtifacts& artifacts,
        bool fp16,
        int num_threads)
        : module_name_(std::move(module_name))
    {
        auto net = std::make_unique<ncnn::Net>();
        int ret = lada::register_torchvision_deform_conv2d_layers(*net);
        if (ret != 0) {
            throw std::runtime_error(
                "Failed to register deformconv custom layer for module '" + module_name_ + "'.");
        }
        ret = lada::register_lada_gridsample_layers(*net);
        if (ret != 0) {
            throw std::runtime_error(
                "Failed to register gridsample custom layer for module '" + module_name_ + "'.");
        }

        net->set_vulkan_device(0);
        net->opt.use_vulkan_compute = true;
        net->opt.use_fp16_storage = fp16;
        net->opt.use_fp16_packed = fp16;
        net->opt.use_fp16_arithmetic = fp16;
        net->opt.num_threads = std::max(num_threads, 1);

        if (net->load_param(artifacts.param_path.c_str()) != 0) {
            throw std::runtime_error(
                "Failed to load ncnn param for module '" + module_name_ + "': " + artifacts.param_path);
        }
        if (net->load_model(artifacts.bin_path.c_str()) != 0) {
            throw std::runtime_error(
                "Failed to load ncnn weights for module '" + module_name_ + "': " + artifacts.bin_path);
        }

        context_ = std::make_shared<ModuleContext>(std::move(net));
    }

    ncnn::VkMat run_to_gpu(const NamedValues& inputs, const std::string& output_name = "out0") const
    {
        clip_trace("[clip] run_to_gpu module=%s output=%s begin\n", module_name_.c_str(), output_name.c_str());
        ncnn::VkCompute extract_cmd(context_->vkdev);
        std::vector<ncnn::Extractor> extractors;
        extractors.reserve(1);
        ncnn::VkMat output = record_to_gpu(inputs, extract_cmd, extractors, output_name);
        const int ret = extract_cmd.submit_and_wait();
        if (ret == 0) {
            lada::finalize_native_op_gpu_profile(extract_cmd, context_->vkdev);
        }
        if (ret != 0 || output.empty()) {
            throw std::runtime_error(
                "Failed to produce Vulkan tensor for module '" + module_name_ + "'.");
        }
        clip_trace("[clip] run_to_gpu module=%s done\n", module_name_.c_str());
        return output;
    }

    ncnn::VkMat record_to_gpu(
        const NamedValues& inputs,
        ncnn::VkCompute& extract_cmd,
        std::vector<ncnn::Extractor>& extractors,
        const std::string& output_name = "out0") const
    {
        clip_trace(
            "[clip] record_to_gpu module=%s output=%s inputs=%zu\n",
            module_name_.c_str(),
            output_name.c_str(),
            inputs.size());
        extractors.push_back(context_->create_extractor(true));
        ncnn::Extractor& extractor = extractors.back();
        feed_inputs(extractor, inputs);

        ncnn::VkMat extracted_output;
        const int ret = extractor.extract(output_name.c_str(), extracted_output, extract_cmd);
        if (ret != 0) {
            throw std::runtime_error(
                "Failed to record Vulkan extractor output for module '" + module_name_ + "'.");
        }

        // The caller keeps `extractors` alive until command submission so the recorded graph
        // and its intermediates remain valid. `VkMat` itself is refcounted, so returning the
        // extracted output avoids an extra device-side clone without dropping ownership.
        return extracted_output;
    }

    ncnn::Mat run_to_cpu(const NamedValues& inputs, const std::string& output_name = "out0") const
    {
        ncnn::Extractor extractor = context_->create_extractor(true);
        feed_inputs(extractor, inputs);

        return extract_cpu_output(extractor, output_name);
    }

    std::vector<ncnn::Mat> run_many_to_cpu(
        const std::vector<NamedValues>& inputs_batch,
        const std::string& output_name = "out0") const
    {
        clip_trace(
            "[clip] run_many_to_cpu module=%s batch=%zu begin\n",
            module_name_.c_str(),
            inputs_batch.size());
        if (inputs_batch.empty()) {
            return {};
        }

        ncnn::Option opt = make_gpu_option();
        std::vector<ncnn::Mat> cpu_outputs(inputs_batch.size());
        for (std::size_t chunk_begin = 0; chunk_begin < inputs_batch.size(); chunk_begin += kOutputFrameDownloadChunkSize) {
            const std::size_t chunk_end = std::min(
                inputs_batch.size(),
                chunk_begin + kOutputFrameDownloadChunkSize);

            ncnn::VkCompute extract_cmd(context_->vkdev);
            std::vector<ncnn::Extractor> extractors;
            extractors.reserve(chunk_end - chunk_begin);

            std::vector<ncnn::VkMat> outputs;
            outputs.reserve(chunk_end - chunk_begin);
            for (std::size_t input_index = chunk_begin; input_index < chunk_end; ++input_index) {
                outputs.push_back(
                    record_to_gpu(inputs_batch[input_index], extract_cmd, extractors, output_name));
            }

            for (std::size_t output_index = 0; output_index < outputs.size(); ++output_index) {
                extract_cmd.record_download(
                    outputs[output_index],
                    cpu_outputs[chunk_begin + output_index],
                    opt);
            }

            const int ret = extract_cmd.submit_and_wait();
            if (ret == 0) {
                lada::finalize_native_op_gpu_profile(extract_cmd, context_->vkdev);
            }
            if (ret != 0) {
                throw std::runtime_error(
                    "Failed to batch-download CPU tensors from module '" + module_name_ + "'.");
            }
        }

        for (std::size_t output_index = 0; output_index < cpu_outputs.size(); ++output_index) {
            cpu_outputs[output_index] = unpack_cpu_output(std::move(cpu_outputs[output_index]));
        }
        clip_trace("[clip] run_many_to_cpu module=%s done\n", module_name_.c_str());
        return cpu_outputs;
    }

    const ncnn::VulkanDevice* vkdev() const
    {
        return context_->vkdev;
    }

    ncnn::Option gpu_option() const
    {
        return make_gpu_option();
    }

private:
    ncnn::Option make_gpu_option() const
    {
        ncnn::Option opt = context_->net->opt;
        opt.use_vulkan_compute = true;
        opt.blob_vkallocator = context_->blob_vkallocator;
        opt.workspace_vkallocator = context_->blob_vkallocator;
        opt.staging_vkallocator = context_->staging_vkallocator;
        return opt;
    }

    ncnn::Mat extract_cpu_output(
        ncnn::Extractor& extractor,
        const std::string& output_name) const
    {
        ncnn::Mat output;
        const int ret = extractor.extract(output_name.c_str(), output);
        if (ret != 0) {
            throw std::runtime_error(
                "Failed to extract CPU tensor from module '" + module_name_ + "'.");
        }

        return unpack_cpu_output(std::move(output));
    }

    ncnn::Mat unpack_cpu_output(ncnn::Mat output) const
    {
        if (output.elempack != 1) {
            ncnn::Option unpack_opt = context_->net->opt;
            unpack_opt.use_vulkan_compute = false;
            unpack_opt.use_packing_layout = false;

            ncnn::Mat unpacked_output;
            ncnn::convert_packing(output, unpacked_output, 1, unpack_opt);
            output = std::move(unpacked_output);
        }

        return output.clone();
    }

    void feed_inputs(ncnn::Extractor& extractor, const NamedValues& inputs) const
    {
        for (const auto& entry : inputs) {
            const std::string& blob_name = entry.first;
            int ret = 0;
            if (std::holds_alternative<ncnn::Mat>(entry.second)) {
                ret = extractor.input(blob_name.c_str(), std::get<ncnn::Mat>(entry.second));
            } else {
                ret = extractor.input(blob_name.c_str(), std::get<ncnn::VkMat>(entry.second));
            }

            if (ret != 0) {
                throw std::runtime_error(
                    "Failed to feed '" + blob_name + "' into module '" + module_name_ + "'.");
            }
        }
    }

    std::string module_name_;
    std::shared_ptr<ModuleContext> context_;
};

ncnn::Mat py_array_to_ncnn_mat(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& array)
{
    const py::buffer_info info = array.request();
    if (info.ndim != 3) {
        throw std::runtime_error("BasicVsrppClipRunner expects CHW float32 tensors.");
    }

    const int channels = static_cast<int>(info.shape[0]);
    const int height = static_cast<int>(info.shape[1]);
    const int width = static_cast<int>(info.shape[2]);
    ncnn::Mat mat(width, height, channels, static_cast<size_t>(4u), 1);
    std::memcpy(mat.data, info.ptr, static_cast<size_t>(channels) * height * width * sizeof(float));
    return mat;
}

py::array_t<float> ncnn_mat_to_py_array(const ncnn::Mat& mat)
{
    if (mat.dims != 3 || mat.elempack != 1) {
        throw std::runtime_error("BasicVsrppClipRunner can only return unpacked 3D tensors.");
    }

    py::array_t<float> output({mat.c, mat.h, mat.w});
    float* dst = output.mutable_data();
    const size_t plane_size = static_cast<size_t>(mat.w) * mat.h;
    for (int channel = 0; channel < mat.c; ++channel) {
        const float* src = reinterpret_cast<const float*>(mat.channel(channel).data);
        std::memcpy(
            dst + static_cast<size_t>(channel) * plane_size,
            src,
            plane_size * sizeof(float));
    }
    return output;
}

ncnn::Mat zeros_like_vkmat(const ncnn::VkMat& value)
{
    if (value.dims != 3) {
        throw std::runtime_error("BasicVsrppClipRunner expects 3D feature tensors.");
    }

    ncnn::Mat zero(value.w, value.h, value.c * value.elempack, static_cast<size_t>(4u), 1);
    zero.fill(0.f);
    return zero;
}

NamedValues build_backbone_inputs(
    const std::string& module_name,
    const ncnn::VkMat& feat_current,
    const ModuleValue& feat_prop,
    const std::unordered_map<std::string, std::vector<ncnn::VkMat>>& branch_feats,
    int frame_index)
{
    if (module_name == kBackward1) {
        return {
            {"in0", feat_current},
            {"in1", feat_prop},
        };
    }

    if (module_name == kForward1) {
        return {
            {"in0", feat_current},
            {"in1", branch_feats.at(kBackward1)[frame_index]},
            {"in2", feat_prop},
        };
    }

    if (module_name == kBackward2) {
        return {
            {"in0", feat_current},
            {"in1", branch_feats.at(kBackward1)[frame_index]},
            {"in2", branch_feats.at(kForward1)[frame_index]},
            {"in3", feat_prop},
        };
    }

    return {
        {"in0", feat_current},
        {"in1", branch_feats.at(kBackward1)[frame_index]},
        {"in2", branch_feats.at(kForward1)[frame_index]},
        {"in3", branch_feats.at(kBackward2)[frame_index]},
        {"in4", feat_prop},
    };
}

NamedValues build_step_inputs(
    const std::string& module_name,
    const ModuleValue& feat_prop,
    const ncnn::VkMat& feat_current,
    const std::unordered_map<std::string, std::vector<ncnn::VkMat>>& branch_feats,
    int frame_index,
    const ModuleValue& feat_n2,
    const ncnn::VkMat& flow_n1,
    const ModuleValue& prev_flow_n2)
{
    if (module_name == kBackward1) {
        return {
            {"in0", feat_prop},
            {"in1", feat_current},
            {"in2", feat_n2},
            {"in3", flow_n1},
            {"in4", prev_flow_n2},
        };
    }

    if (module_name == kForward1) {
        return {
            {"in0", feat_prop},
            {"in1", feat_current},
            {"in2", branch_feats.at(kBackward1)[frame_index]},
            {"in3", feat_n2},
            {"in4", flow_n1},
            {"in5", prev_flow_n2},
        };
    }

    if (module_name == kBackward2) {
        return {
            {"in0", feat_prop},
            {"in1", feat_current},
            {"in2", branch_feats.at(kBackward1)[frame_index]},
            {"in3", branch_feats.at(kForward1)[frame_index]},
            {"in4", feat_n2},
            {"in5", flow_n1},
            {"in6", prev_flow_n2},
        };
    }

    return {
        {"in0", feat_prop},
        {"in1", feat_current},
        {"in2", branch_feats.at(kBackward1)[frame_index]},
        {"in3", branch_feats.at(kForward1)[frame_index]},
        {"in4", branch_feats.at(kBackward2)[frame_index]},
        {"in5", feat_n2},
        {"in6", flow_n1},
        {"in7", prev_flow_n2},
    };
}

NamedValues build_output_frame_inputs(
    const std::vector<ModuleValue>& lqs,
    const std::vector<ncnn::VkMat>& spatial_feats,
    const std::unordered_map<std::string, std::vector<ncnn::VkMat>>& branch_feats,
    int frame_index)
{
    return {
        {"in0", lqs[frame_index]},
        {"in1", spatial_feats[frame_index]},
        {"in2", branch_feats.at(kBackward1)[frame_index]},
        {"in3", branch_feats.at(kForward1)[frame_index]},
        {"in4", branch_feats.at(kBackward2)[frame_index]},
        {"in5", branch_feats.at(kForward2)[frame_index]},
    };
}

class BasicVsrppClipRunner
{
public:
    BasicVsrppClipRunner(
        const std::unordered_map<std::string, ModuleArtifacts>& module_artifacts,
        bool fp16,
        int num_threads)
    {
        for (const std::string& module_name : kRequiredModules) {
            const auto it = module_artifacts.find(module_name);
            if (it == module_artifacts.end()) {
                throw std::runtime_error(
                    "BasicVsrppClipRunner is missing required module artifact '" + module_name + "'.");
            }
            runners_.emplace(
                module_name,
                std::make_unique<ModuleRunner>(module_name, it->second, fp16, num_threads));
        }
    }

    ~BasicVsrppClipRunner()
    {
        delete frame_preprocess_pipeline_;
        frame_preprocess_pipeline_ = nullptr;
        delete frame_resize_preprocess_pipeline_;
        frame_resize_preprocess_pipeline_ = nullptr;
    }

    std::vector<ncnn::Mat> restore(const std::vector<ncnn::Mat>& lqs) const
    {
        clip_trace("[clip] restore float begin frames=%zu\n", lqs.size());
        reset_last_profile();
        set_last_profile_duration("input_preprocess_s", 0.0);
        std::vector<ModuleValue> runtime_lqs;
        runtime_lqs.reserve(lqs.size());
        for (const ncnn::Mat& lq : lqs) {
            runtime_lqs.emplace_back(lq);
        }
        std::vector<ncnn::Mat> outputs = restore_impl(runtime_lqs);
        clip_trace("[clip] restore float done\n");
        return outputs;
    }

    std::vector<ncnn::Mat> restore_packed_bgr_u8(const std::vector<PackedBgrFrame>& frames) const
    {
        clip_trace("[clip] restore_bgr_u8 begin frames=%zu\n", frames.size());
        reset_last_profile();
        const auto preprocess_started_at = SteadyClock::now();
        const std::vector<ncnn::VkMat> preprocessed_frames = preprocess_packed_bgr_frames(frames);
        set_last_profile_duration("input_preprocess_s", elapsed_seconds(preprocess_started_at));
        clip_trace("[clip] restore_bgr_u8 preprocessed=%zu\n", preprocessed_frames.size());
        std::vector<ModuleValue> runtime_lqs;
        runtime_lqs.reserve(preprocessed_frames.size());
        for (const ncnn::VkMat& frame : preprocessed_frames) {
            runtime_lqs.emplace_back(frame);
        }
        std::vector<ncnn::Mat> outputs = restore_impl(runtime_lqs);
        clip_trace("[clip] restore_bgr_u8 done\n");
        return outputs;
    }

    std::vector<ncnn::Mat> restore_packed_bgr_u8_resized(
        const std::vector<PackedBgrResizeFrame>& frames) const
    {
        reset_last_profile();
        const auto preprocess_started_at = SteadyClock::now();
        const std::vector<ncnn::VkMat> preprocessed_frames = preprocess_packed_bgr_frames_resized(frames);
        set_last_profile_duration("input_preprocess_s", elapsed_seconds(preprocess_started_at));
        std::vector<ModuleValue> runtime_lqs;
        runtime_lqs.reserve(preprocessed_frames.size());
        for (const ncnn::VkMat& frame : preprocessed_frames) {
            runtime_lqs.emplace_back(frame);
        }
        return restore_impl(runtime_lqs);
    }

    py::dict get_last_profile() const
    {
        py::dict profile;
        for (const auto& entry : last_profile_) {
            profile[py::str(entry.first)] = py::float_(entry.second);
        }
        return profile;
    }

private:
    std::vector<ncnn::Mat> restore_impl(const std::vector<ModuleValue>& lqs) const
    {
        if (lqs.size() < 2) {
            throw std::runtime_error("BasicVsrppClipRunner expects at least two frames.");
        }

        const auto restore_started_at = SteadyClock::now();
        std::vector<ncnn::VkMat> lqs_downsampled;
        std::vector<ncnn::VkMat> spatial_feats;
        lqs_downsampled.reserve(lqs.size());
        spatial_feats.reserve(lqs.size());

        std::vector<ncnn::VkMat> flows_backward;
        std::vector<ncnn::VkMat> flows_forward;
        flows_backward.reserve(lqs.size() - 1);
        flows_forward.reserve(lqs.size() - 1);

        {
            const auto preprocess_started_at = SteadyClock::now();
            ncnn::VkCompute preprocess_cmd(runner(kQuarterDownsample).vkdev());
            std::vector<ncnn::Extractor> preprocess_extractors;
            preprocess_extractors.reserve(lqs.size() * 2 + (lqs.size() - 1) * 2);

            for (const ModuleValue& current_lq : lqs) {
                lqs_downsampled.push_back(
                    runner(kQuarterDownsample).record_to_gpu(
                        {{"in0", current_lq}},
                        preprocess_cmd,
                        preprocess_extractors));
                spatial_feats.push_back(
                    runner(kFeatExtract).record_to_gpu(
                        {{"in0", current_lq}},
                        preprocess_cmd,
                        preprocess_extractors));
            }

            for (size_t index = 0; index + 1 < lqs_downsampled.size(); ++index) {
                flows_backward.push_back(
                    runner(kSpyNet).record_to_gpu(
                        {{"in0", lqs_downsampled[index]}, {"in1", lqs_downsampled[index + 1]}},
                        preprocess_cmd,
                        preprocess_extractors));
                flows_forward.push_back(
                    runner(kSpyNet).record_to_gpu(
                        {{"in0", lqs_downsampled[index + 1]}, {"in1", lqs_downsampled[index]}},
                        preprocess_cmd,
                        preprocess_extractors));
            }

            if (preprocess_cmd.submit_and_wait() != 0) {
                throw std::runtime_error("Failed to submit preprocess Vulkan subgraph.");
            }
            lada::finalize_native_op_gpu_profile(preprocess_cmd, runner(kQuarterDownsample).vkdev());
            set_last_profile_duration("graph_preprocess_s", elapsed_seconds(preprocess_started_at));
        }

        std::unordered_map<std::string, std::vector<ncnn::VkMat>> branch_feats;
        {
            const auto started_at = SteadyClock::now();
            branch_feats.emplace(kBackward1, run_branch(kBackward1, spatial_feats, branch_feats, flows_backward));
            set_last_profile_duration("branch_backward_1_s", elapsed_seconds(started_at));
        }
        {
            const auto started_at = SteadyClock::now();
            branch_feats.emplace(kForward1, run_branch(kForward1, spatial_feats, branch_feats, flows_forward));
            set_last_profile_duration("branch_forward_1_s", elapsed_seconds(started_at));
        }
        {
            const auto started_at = SteadyClock::now();
            branch_feats.emplace(kBackward2, run_branch(kBackward2, spatial_feats, branch_feats, flows_backward));
            set_last_profile_duration("branch_backward_2_s", elapsed_seconds(started_at));
        }
        {
            const auto started_at = SteadyClock::now();
            branch_feats.emplace(kForward2, run_branch(kForward2, spatial_feats, branch_feats, flows_forward));
            set_last_profile_duration("branch_forward_2_s", elapsed_seconds(started_at));
        }

        std::vector<NamedValues> output_inputs;
        output_inputs.reserve(lqs.size());
        for (size_t frame_index = 0; frame_index < lqs.size(); ++frame_index) {
            output_inputs.push_back(
                build_output_frame_inputs(
                    lqs,
                    spatial_feats,
                    branch_feats,
                    static_cast<int>(frame_index)));
        }
        const auto output_started_at = SteadyClock::now();
        std::vector<ncnn::Mat> outputs = runner(kOutputFrame).run_many_to_cpu(output_inputs);
        set_last_profile_duration("output_frame_s", elapsed_seconds(output_started_at));
        set_last_profile_duration("total_s", elapsed_seconds(restore_started_at));
        merge_native_op_profile();
        return outputs;
    }

    void reset_last_profile() const
    {
        last_profile_.clear();
        lada::reset_native_op_profile();
    }

    void set_last_profile_duration(const std::string& key, double duration_s) const
    {
        last_profile_[key] = duration_s;
    }

    void merge_native_op_profile() const
    {
        const lada::NativeOpProfileSnapshot snapshot = lada::snapshot_native_op_profile();
        last_profile_["custom_deformconv_cpu_s"] = snapshot.deformconv_cpu_s;
        last_profile_["custom_deformconv_cpu_count"] = static_cast<double>(snapshot.deformconv_cpu_count);
        last_profile_["custom_deformconv_vulkan_s"] = snapshot.deformconv_vulkan_s;
        last_profile_["custom_deformconv_vulkan_gpu_s"] = snapshot.deformconv_vulkan_gpu_s;
        last_profile_["custom_deformconv_vulkan_count"] = static_cast<double>(snapshot.deformconv_vulkan_count);
        last_profile_["custom_gridsample_cpu_s"] = snapshot.gridsample_cpu_s;
        last_profile_["custom_gridsample_cpu_count"] = static_cast<double>(snapshot.gridsample_cpu_count);
        last_profile_["custom_gridsample_vulkan_s"] = snapshot.gridsample_vulkan_s;
        last_profile_["custom_gridsample_vulkan_gpu_s"] = snapshot.gridsample_vulkan_gpu_s;
        last_profile_["custom_gridsample_vulkan_count"] = static_cast<double>(snapshot.gridsample_vulkan_count);
    }

    void ensure_frame_preprocess_pipeline() const
    {
        if (frame_preprocess_pipeline_ != nullptr) {
            return;
        }

        ncnn::Option opt = runner(kQuarterDownsample).gpu_option();
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = false;

        std::vector<std::uint32_t> spirv;
        const int compile_ret = ncnn::compile_spirv_module(
            kBasicVsrppFramePreprocessShader,
            static_cast<int>(sizeof(kBasicVsrppFramePreprocessShader) - 1),
            opt,
            spirv);
        if (compile_ret != 0) {
            throw std::runtime_error("Failed to compile BasicVSR++ frame preprocess shader.");
        }

        frame_preprocess_pipeline_ = new ncnn::Pipeline(runner(kQuarterDownsample).vkdev());
        frame_preprocess_pipeline_->set_optimal_local_size_xyz(8, 8, 1);
        if (
            frame_preprocess_pipeline_->create(
                spirv.data(),
                spirv.size() * sizeof(std::uint32_t),
                std::vector<ncnn::vk_specialization_type>()) != 0) {
            delete frame_preprocess_pipeline_;
            frame_preprocess_pipeline_ = nullptr;
            throw std::runtime_error("Failed to create BasicVSR++ frame preprocess pipeline.");
        }
    }

    void ensure_frame_resize_preprocess_pipeline() const
    {
        if (frame_resize_preprocess_pipeline_ != nullptr) {
            return;
        }

        ncnn::Option opt = runner(kQuarterDownsample).gpu_option();
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = false;

        std::vector<std::uint32_t> spirv;
        const int compile_ret = ncnn::compile_spirv_module(
            kBasicVsrppResizePreprocessShader,
            static_cast<int>(sizeof(kBasicVsrppResizePreprocessShader) - 1),
            opt,
            spirv);
        if (compile_ret != 0) {
            throw std::runtime_error("Failed to compile BasicVSR++ resize preprocess shader.");
        }

        frame_resize_preprocess_pipeline_ = new ncnn::Pipeline(runner(kQuarterDownsample).vkdev());
        frame_resize_preprocess_pipeline_->set_optimal_local_size_xyz(8, 8, 1);
        if (
            frame_resize_preprocess_pipeline_->create(
                spirv.data(),
                spirv.size() * sizeof(std::uint32_t),
                std::vector<ncnn::vk_specialization_type>()) != 0) {
            delete frame_resize_preprocess_pipeline_;
            frame_resize_preprocess_pipeline_ = nullptr;
            throw std::runtime_error("Failed to create BasicVSR++ resize preprocess pipeline.");
        }
    }

    std::vector<ncnn::VkMat> preprocess_packed_bgr_frames(const std::vector<PackedBgrFrame>& frames) const
    {
        clip_trace("[clip] preprocess_bgr_u8 begin frames=%zu\n", frames.size());
        std::vector<ncnn::VkMat> outputs;
        outputs.reserve(frames.size());
        if (frames.empty()) {
            return outputs;
        }

        ensure_frame_preprocess_pipeline();

        ncnn::Option opt = runner(kQuarterDownsample).gpu_option();
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = false;

        ncnn::VkCompute cmd(runner(kQuarterDownsample).vkdev());
        for (const PackedBgrFrame& frame : frames) {
            if (frame.width <= 0 || frame.height <= 0 || frame.packed_input.empty()) {
                throw std::runtime_error("BasicVSR++ frame preprocess received an invalid packed frame.");
            }

            ncnn::VkMat input_blob;
            input_blob.create(frame.packed_input.w, frame.packed_input.elemsize, 1, opt.blob_vkallocator);
            if (input_blob.empty()) {
                throw std::runtime_error("Failed to allocate BasicVSR++ preprocess Vulkan input tensor.");
            }
            cmd.record_upload(frame.packed_input, input_blob, opt);

            ncnn::VkMat output_blob;
            output_blob.create(
                frame.width,
                frame.height,
                3,
                static_cast<size_t>(4u),
                1,
                opt.blob_vkallocator);
            if (output_blob.empty()) {
                throw std::runtime_error("Failed to allocate BasicVSR++ preprocess Vulkan output tensor.");
            }

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = input_blob;
            bindings[1] = output_blob;
            std::vector<ncnn::vk_constant_type> constants(3);
            constants[0].i = frame.width;
            constants[1].i = frame.height;
            constants[2].i = static_cast<int>(output_blob.cstep);
            cmd.record_pipeline(frame_preprocess_pipeline_, bindings, constants, output_blob);
            outputs.push_back(output_blob);
        }

        if (cmd.submit_and_wait() != 0) {
            throw std::runtime_error("Failed to execute BasicVSR++ frame preprocess Vulkan pipeline.");
        }
        lada::finalize_native_op_gpu_profile(cmd, runner(kQuarterDownsample).vkdev());
        clip_trace("[clip] preprocess_bgr_u8 done outputs=%zu\n", outputs.size());
        return outputs;
    }

    std::vector<ncnn::VkMat> preprocess_packed_bgr_frames_resized(
        const std::vector<PackedBgrResizeFrame>& frames) const
    {
        std::vector<ncnn::VkMat> outputs;
        outputs.reserve(frames.size());
        if (frames.empty()) {
            return outputs;
        }

        ensure_frame_resize_preprocess_pipeline();

        ncnn::Option opt = runner(kQuarterDownsample).gpu_option();
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = false;

        ncnn::VkCompute cmd(runner(kQuarterDownsample).vkdev());
        for (const PackedBgrResizeFrame& frame : frames) {
            if (
                frame.src_w <= 0 || frame.src_h <= 0 || frame.dst_w <= 0 || frame.dst_h <= 0
                || frame.resized_w <= 0 || frame.resized_h <= 0 || frame.packed_input.empty()) {
                throw std::runtime_error("BasicVSR++ resized preprocess received an invalid packed frame.");
            }

            ncnn::VkMat input_blob;
            input_blob.create(frame.packed_input.w, frame.packed_input.elemsize, 1, opt.blob_vkallocator);
            if (input_blob.empty()) {
                throw std::runtime_error("Failed to allocate BasicVSR++ resize preprocess Vulkan input tensor.");
            }
            cmd.record_upload(frame.packed_input, input_blob, opt);

            ncnn::VkMat output_blob;
            output_blob.create(
                frame.dst_w,
                frame.dst_h,
                3,
                static_cast<size_t>(4u),
                1,
                opt.blob_vkallocator);
            if (output_blob.empty()) {
                throw std::runtime_error("Failed to allocate BasicVSR++ resize preprocess Vulkan output tensor.");
            }

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = input_blob;
            bindings[1] = output_blob;
            std::vector<ncnn::vk_constant_type> constants(10);
            constants[0].i = frame.src_w;
            constants[1].i = frame.src_h;
            constants[2].i = frame.dst_w;
            constants[3].i = frame.dst_h;
            constants[4].i = frame.resized_w;
            constants[5].i = frame.resized_h;
            constants[6].i = frame.pad_left;
            constants[7].i = frame.pad_top;
            constants[8].i = frame.pad_mode;
            constants[9].i = static_cast<int>(output_blob.cstep);
            cmd.record_pipeline(frame_resize_preprocess_pipeline_, bindings, constants, output_blob);
            outputs.push_back(output_blob);
        }

        if (cmd.submit_and_wait() != 0) {
            throw std::runtime_error("Failed to execute BasicVSR++ resize preprocess Vulkan pipeline.");
        }
        lada::finalize_native_op_gpu_profile(cmd, runner(kQuarterDownsample).vkdev());

        return outputs;
    }

    const ModuleRunner& runner(const std::string& module_name) const
    {
        return *runners_.at(module_name);
    }

    std::vector<ncnn::VkMat> run_branch(
        const std::string& module_name,
        const std::vector<ncnn::VkMat>& spatial_feats,
        const std::unordered_map<std::string, std::vector<ncnn::VkMat>>& branch_feats,
        const std::vector<ncnn::VkMat>& flows) const
    {
        clip_trace("[clip] run_branch begin module=%s frames=%zu\n", module_name.c_str(), spatial_feats.size());
        std::vector<int> frame_indices;
        frame_indices.reserve(spatial_feats.size());
        for (size_t index = 0; index < spatial_feats.size(); ++index) {
            frame_indices.push_back(static_cast<int>(index));
        }
        if (module_name.rfind("backward", 0) == 0) {
            std::reverse(frame_indices.begin(), frame_indices.end());
        }

        ModuleValue feat_prop = zeros_like_vkmat(spatial_feats.front());
        const ncnn::Mat zero_feat = zeros_like_vkmat(spatial_feats.front());
        const ncnn::Mat zero_flow = zeros_like_vkmat(flows.front());
        ModuleValue previous_raw_flow = zero_flow;
        std::vector<ncnn::VkMat> outputs;
        outputs.reserve(spatial_feats.size());
        ncnn::VkCompute branch_cmd(runner(module_name + "_backbone").vkdev());
        std::vector<ncnn::Extractor> branch_extractors;
        branch_extractors.reserve(frame_indices.size());

        for (size_t step_index = 0; step_index < frame_indices.size(); ++step_index) {
            const int frame_index = frame_indices[step_index];
            const ncnn::VkMat& feat_current = spatial_feats[frame_index];

            if (step_index == 0) {
                feat_prop = runner(module_name + "_backbone").record_to_gpu(
                    build_backbone_inputs(
                        module_name,
                        feat_current,
                        feat_prop,
                        branch_feats,
                        frame_index),
                    branch_cmd,
                    branch_extractors);
                outputs.push_back(std::get<ncnn::VkMat>(feat_prop));
                continue;
            }

            const int adjacent_index = frame_indices[step_index - 1];
            const ncnn::VkMat& raw_flow_n1 = flows[std::min(frame_index, adjacent_index)];
            const ModuleValue feat_n2 =
                step_index > 1 ? ModuleValue(outputs[outputs.size() - 2]) : ModuleValue(zero_feat);
            const ModuleValue prev_flow_n2 =
                step_index > 1 ? previous_raw_flow : ModuleValue(zero_flow);
            feat_prop = runner(module_name + "_step").record_to_gpu(
                build_step_inputs(
                    module_name,
                    feat_prop,
                    feat_current,
                    branch_feats,
                    frame_index,
                    feat_n2,
                    raw_flow_n1,
                    prev_flow_n2),
                branch_cmd,
                branch_extractors);
            outputs.push_back(std::get<ncnn::VkMat>(feat_prop));
            previous_raw_flow = raw_flow_n1;
        }

        const int submit_ret = branch_cmd.submit_and_wait();
        if (submit_ret != 0) {
            throw std::runtime_error(
                "Failed to submit branch Vulkan subgraph for module '" + module_name + "'.");
        }
        lada::finalize_native_op_gpu_profile(branch_cmd, runner(module_name + "_backbone").vkdev());

        if (module_name.rfind("backward", 0) == 0) {
            std::reverse(outputs.begin(), outputs.end());
        }
        clip_trace("[clip] run_branch done module=%s outputs=%zu\n", module_name.c_str(), outputs.size());
        return outputs;
    }

    std::unordered_map<std::string, std::unique_ptr<ModuleRunner>> runners_;
    mutable std::unordered_map<std::string, double> last_profile_;
    mutable ncnn::Pipeline* frame_preprocess_pipeline_ = nullptr;
    mutable ncnn::Pipeline* frame_resize_preprocess_pipeline_ = nullptr;
};

PackedBgrFrame pack_bgr_u8_frame(const py::handle& value)
{
    const auto array = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw std::runtime_error("BasicVsrppClipRunner uint8 inputs must be numpy-compatible arrays.");
    }

    const py::buffer_info info = array.request();
    if (info.ndim != 3 || info.shape[2] != 3) {
        throw std::runtime_error("BasicVsrppClipRunner uint8 inputs must be HWC uint8 tensors.");
    }

    const int height = static_cast<int>(info.shape[0]);
    const int width = static_cast<int>(info.shape[1]);
    const std::size_t byte_count =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 3u;
    const std::size_t word_count = (byte_count + 3u) / 4u;

    ncnn::Mat packed_input(static_cast<int>(word_count), static_cast<size_t>(4u), 1);
    if (packed_input.empty()) {
        throw std::runtime_error("Failed to allocate BasicVSR++ packed uint8 input buffer.");
    }
    std::memset(packed_input.data, 0, word_count * sizeof(std::uint32_t));
    std::memcpy(packed_input.data, info.ptr, byte_count);

    PackedBgrFrame frame;
    frame.width = width;
    frame.height = height;
    frame.packed_input = std::move(packed_input);
    return frame;
}

int parse_basicvsrpp_pad_mode(const std::string& pad_mode)
{
    if (pad_mode == "zero") {
        return 0;
    }
    if (pad_mode == "reflect") {
        return 1;
    }
    throw std::runtime_error("BasicVsrppClipRunner only supports 'zero' and 'reflect' pad modes.");
}

double elapsed_seconds(const SteadyClock::time_point started_at)
{
    return std::chrono::duration<double>(SteadyClock::now() - started_at).count();
}

PackedBgrResizeFrame pack_bgr_u8_resize_frame(
    const py::handle& value,
    int target_size,
    int resize_reference_width,
    int resize_reference_height,
    int pad_mode)
{
    PackedBgrFrame packed_frame = pack_bgr_u8_frame(value);
    if (target_size <= 0 || resize_reference_width <= 0 || resize_reference_height <= 0) {
        throw std::runtime_error("BasicVsrppClipRunner resized input received invalid target metadata.");
    }

    const float scale_width = static_cast<float>(target_size) / static_cast<float>(resize_reference_width);
    const float scale_height = static_cast<float>(target_size) / static_cast<float>(resize_reference_height);
    const int resized_w = std::max(1, static_cast<int>(std::floor(packed_frame.width * scale_width)));
    const int resized_h = std::max(1, static_cast<int>(std::floor(packed_frame.height * scale_height)));
    if (resized_w > target_size || resized_h > target_size) {
        throw std::runtime_error("BasicVsrppClipRunner resized input exceeds the target frame size.");
    }

    const int pad_w = target_size - resized_w;
    const int pad_h = target_size - resized_h;
    PackedBgrResizeFrame frame;
    frame.src_w = packed_frame.width;
    frame.src_h = packed_frame.height;
    frame.dst_w = target_size;
    frame.dst_h = target_size;
    frame.resized_w = resized_w;
    frame.resized_h = resized_h;
    frame.pad_left = (pad_w + 1) / 2;
    frame.pad_top = (pad_h + 1) / 2;
    frame.pad_mode = pad_mode;
    frame.packed_input = std::move(packed_frame.packed_input);
    return frame;
}

} // namespace
#endif

namespace lada {

void bind_basicvsrpp_clip_runner(py::module_& m)
{
#if NCNN_VULKAN
    py::class_<BasicVsrppClipRunner>(m, "BasicVsrppClipRunner")
        .def(
            py::init([](const py::dict& module_paths, bool fp16, int num_threads) {
                return std::make_unique<BasicVsrppClipRunner>(
                    parse_module_artifacts_map(module_paths),
                    fp16,
                    num_threads);
            }),
            py::arg("module_paths"),
            py::arg("fp16") = false,
            py::arg("num_threads") = 1)
        .def(
            "restore",
            [](const BasicVsrppClipRunner& self, const py::list& lqs) {
                std::vector<ncnn::Mat> input_mats;
                input_mats.reserve(py::len(lqs));
                for (const py::handle& item : lqs) {
                    auto array = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(item);
                    if (!array) {
                        throw std::runtime_error(
                            "BasicVsrppClipRunner inputs must be float32 numpy-compatible arrays.");
                    }
                    input_mats.push_back(py_array_to_ncnn_mat(array));
                }

                std::vector<ncnn::Mat> outputs;
                {
                    py::gil_scoped_release release;
                    outputs = self.restore(input_mats);
                }
                py::list result;
                for (const ncnn::Mat& output : outputs) {
                    result.append(ncnn_mat_to_py_array(output));
                }
                return result;
            },
            py::arg("lqs"))
        .def(
            "restore_bgr_u8",
            [](const BasicVsrppClipRunner& self, const py::list& lqs) {
                std::vector<PackedBgrFrame> packed_frames;
                packed_frames.reserve(py::len(lqs));
                for (const py::handle& item : lqs) {
                    packed_frames.push_back(pack_bgr_u8_frame(item));
                }

                std::vector<ncnn::Mat> outputs;
                {
                    py::gil_scoped_release release;
                    outputs = self.restore_packed_bgr_u8(packed_frames);
                }
                py::list result;
                for (const ncnn::Mat& output : outputs) {
                    result.append(ncnn_mat_to_py_array(output));
                }
                return result;
            },
            py::arg("lqs"))
        .def(
            "restore_bgr_u8_resized",
            [](
                const BasicVsrppClipRunner& self,
                const py::list& lqs,
                int target_size,
                const std::vector<int>& resize_reference_shape,
                const std::string& pad_mode) {
                if (resize_reference_shape.size() != 2) {
                    throw std::runtime_error(
                        "BasicVsrppClipRunner resize_reference_shape must be [max_width, max_height].");
                }

                const int parsed_pad_mode = parse_basicvsrpp_pad_mode(pad_mode);
                std::vector<PackedBgrResizeFrame> packed_frames;
                packed_frames.reserve(py::len(lqs));
                for (const py::handle& item : lqs) {
                    packed_frames.push_back(
                        pack_bgr_u8_resize_frame(
                            item,
                            target_size,
                            resize_reference_shape[0],
                            resize_reference_shape[1],
                            parsed_pad_mode));
                }

                std::vector<ncnn::Mat> outputs;
                {
                    py::gil_scoped_release release;
                    outputs = self.restore_packed_bgr_u8_resized(packed_frames);
                }
                py::list result;
                for (const ncnn::Mat& output : outputs) {
                    result.append(ncnn_mat_to_py_array(output));
                }
                return result;
            },
            py::arg("lqs"),
            py::arg("target_size"),
            py::arg("resize_reference_shape"),
            py::arg("pad_mode"))
        .def("get_last_profile", &BasicVsrppClipRunner::get_last_profile);

    m.attr("has_lada_basicvsrpp_clip_runner") = py::bool_(true);
#else
    m.attr("has_lada_basicvsrpp_clip_runner") = py::bool_(false);
#endif
}

} // namespace lada
