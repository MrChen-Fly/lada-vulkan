// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#include "torchvision_deform_conv2d_layer.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

#if NCNN_VULKAN
#include "gpu.h"
#endif

namespace {

void fill_blob(ncnn::Mat& blob, float scale)
{
    float* ptr = blob;
    for (size_t i = 0; i < blob.total(); i++)
    {
        ptr[i] = static_cast<float>(static_cast<int>(i % 97) - 48) * scale;
    }
}

float max_abs_diff(const ncnn::Mat& a, const ncnn::Mat& b)
{
    if (a.dims != b.dims || a.w != b.w || a.h != b.h || a.d != b.d || a.c != b.c)
        return std::numeric_limits<float>::infinity();

    const float* a_ptr = a;
    const float* b_ptr = b;
    const size_t total = a.total();
    float diff = 0.f;
    for (size_t i = 0; i < total; i++)
    {
        diff = std::max(diff, std::fabs(a_ptr[i] - b_ptr[i]));
    }
    return diff;
}

struct CpuBenchmarkResult
{
    int ret = 0;
    ncnn::Mat output;
    long long elapsed_ms = 0;
};

CpuBenchmarkResult run_cpu_benchmark(
    const ncnn::ParamDict& pd,
    const std::vector<ncnn::Mat>& bottoms)
{
    lada::TorchVisionDeformConv2DLayer layer;
    CpuBenchmarkResult result;

    result.ret = layer.load_param(pd);
    if (result.ret != 0)
        return result;

    ncnn::Option opt;
    opt.num_threads = 1;

    result.ret = layer.create_pipeline(opt);
    if (result.ret != 0)
        return result;

    std::vector<ncnn::Mat> tops(1);
    const auto start = std::chrono::steady_clock::now();
    result.ret = layer.forward(bottoms, tops, opt);
    result.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start)
                            .count();

    layer.destroy_pipeline(opt);

    if (result.ret == 0)
        result.output = tops[0];

    return result;
}

#if NCNN_VULKAN
struct VulkanBenchmarkResult
{
    int ret = 0;
    ncnn::Mat output;
    long long upload_ms = 0;
    long long compute_ms = 0;
    long long download_ms = 0;
    long long total_ms = 0;
};

VulkanBenchmarkResult run_vulkan_benchmark(
    const ncnn::ParamDict& pd,
    const std::vector<ncnn::Mat>& bottoms)
{
    VulkanBenchmarkResult result;
    bool gpu_instance_ready = false;
    ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::VkAllocator* blob_vkallocator = nullptr;
    ncnn::VkAllocator* staging_vkallocator = nullptr;
    lada::TorchVisionDeformConv2DLayer layer;
    bool pipeline_created = false;
    std::vector<ncnn::VkMat> bottoms_uploaded;
    std::vector<ncnn::VkMat> bottoms_gpu;
    std::vector<ncnn::VkMat> tops_gpu;

    const auto cleanup = [&]() {
        tops_gpu.clear();
        bottoms_gpu.clear();
        bottoms_uploaded.clear();
        if (pipeline_created)
            layer.destroy_pipeline(ncnn::Option());
        if (vkdev && blob_vkallocator)
            vkdev->reclaim_blob_allocator(blob_vkallocator);
        if (vkdev && staging_vkallocator)
            vkdev->reclaim_staging_allocator(staging_vkallocator);
        if (gpu_instance_ready)
            ncnn::destroy_gpu_instance();
    };

    std::cerr << "[vulkan] create_gpu_instance" << std::endl;
    result.ret = ncnn::create_gpu_instance();
    if (result.ret != 0)
    {
        cleanup();
        return result;
    }
    gpu_instance_ready = true;

    if (ncnn::get_gpu_count() <= 0)
    {
        result.ret = -1;
        cleanup();
        return result;
    }

    vkdev = ncnn::get_gpu_device();
    if (vkdev == nullptr || !vkdev->is_valid())
    {
        result.ret = -1;
        cleanup();
        return result;
    }

    blob_vkallocator = vkdev->acquire_blob_allocator();
    staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    std::cerr << "[vulkan] load_param" << std::endl;
    layer.vkdev = vkdev;
    result.ret = layer.load_param(pd);
    if (result.ret != 0)
    {
        cleanup();
        return result;
    }

    std::cerr << "[vulkan] create_pipeline" << std::endl;
    result.ret = layer.create_pipeline(opt);
    if (result.ret != 0)
    {
        cleanup();
        return result;
    }
    pipeline_created = true;

    bottoms_uploaded.resize(bottoms.size());
    bottoms_gpu.resize(bottoms.size());
    tops_gpu.resize(1);

    const auto total_start = std::chrono::steady_clock::now();
    {
        std::cerr << "[vulkan] upload" << std::endl;
        ncnn::VkCompute cmd(vkdev);
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < bottoms.size(); i++)
        {
            cmd.record_upload(bottoms[i], bottoms_uploaded[i], opt);
        }
        for (size_t i = 0; i < bottoms.size(); i++)
        {
            if (bottoms_uploaded[i].elempack != 1)
            {
                vkdev->convert_packing(bottoms_uploaded[i], bottoms_gpu[i], 1, cmd, opt);
            }
            else
            {
                bottoms_gpu[i] = bottoms_uploaded[i];
            }
        }
        result.ret = cmd.submit_and_wait();
        result.upload_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
                               .count();
        if (result.ret != 0)
        {
            cleanup();
            return result;
        }
    }

    {
        std::cerr << "[vulkan] compute" << std::endl;
        ncnn::VkCompute cmd(vkdev);
        const auto start = std::chrono::steady_clock::now();
        result.ret = layer.forward(bottoms_gpu, tops_gpu, cmd, opt);
        if (result.ret == 0)
            result.ret = cmd.submit_and_wait();
        result.compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
                                .count();
        if (result.ret != 0)
        {
            cleanup();
            return result;
        }
    }

    {
        std::cerr << "[vulkan] download" << std::endl;
        ncnn::VkCompute cmd(vkdev);
        const auto start = std::chrono::steady_clock::now();
        cmd.record_download(tops_gpu[0], result.output, opt);
        result.ret = cmd.submit_and_wait();
        result.download_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
                                 .count();
        if (result.ret != 0)
        {
            cleanup();
            return result;
        }
    }

    result.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - total_start)
                          .count();

    cleanup();
    return result;
}
#endif

} // namespace

int main()
{
    ncnn::ParamDict pd;
    pd.set(5, 1);
    pd.set(6, 1);
    pd.set(7, 1);
    pd.set(8, 1);
    pd.set(9, 1);
    pd.set(10, 1);
    pd.set(11, 1);
    pd.set(12, 16);
    pd.set(13, 1);

    ncnn::Mat input(64, 64, 128);
    ncnn::Mat weight(3, 3, 128, 64);
    ncnn::Mat offset(64, 64, 288);
    ncnn::Mat mask(64, 64, 144);
    ncnn::Mat bias(64);

    fill_blob(input, 0.0025f);
    fill_blob(weight, 0.001f);
    fill_blob(offset, 0.01f);
    fill_blob(mask, 0.01f);
    fill_blob(bias, 0.001f);

    std::vector<ncnn::Mat> bottoms = {input, weight, offset, mask, bias};
    const CpuBenchmarkResult cpu = run_cpu_benchmark(pd, bottoms);
    if (cpu.ret != 0)
    {
        std::cerr << "cpu forward failed with code " << cpu.ret << "\n";
        return cpu.ret;
    }

    const float* cpu_output_ptr = cpu.output;
    const float cpu_first_value = cpu_output_ptr ? cpu_output_ptr[0] : 0.f;

    std::cout
        << "TorchVisionDeformConv2DLayer benchmark\n"
        << "cpu_output_shape=" << cpu.output.c << "x" << cpu.output.h << "x" << cpu.output.w << "\n"
        << "cpu_elapsed_ms=" << cpu.elapsed_ms << "\n"
        << "cpu_first_value=" << cpu_first_value << "\n";
    std::cout.flush();

#if NCNN_VULKAN
    const VulkanBenchmarkResult vulkan = run_vulkan_benchmark(pd, bottoms);
    if (vulkan.ret != 0)
    {
        std::cerr << "vulkan forward failed with code " << vulkan.ret << "\n";
        return vulkan.ret;
    }

    const float* vulkan_output_ptr = vulkan.output;
    const float vulkan_first_value = vulkan_output_ptr ? vulkan_output_ptr[0] : 0.f;

    std::cout
        << "vulkan_output_shape=" << vulkan.output.c << "x" << vulkan.output.h << "x" << vulkan.output.w << "\n"
        << "vulkan_upload_ms=" << vulkan.upload_ms << "\n"
        << "vulkan_compute_ms=" << vulkan.compute_ms << "\n"
        << "vulkan_download_ms=" << vulkan.download_ms << "\n"
        << "vulkan_total_ms=" << vulkan.total_ms << "\n"
        << "vulkan_first_value=" << vulkan_first_value << "\n"
        << "max_abs_diff=" << max_abs_diff(cpu.output, vulkan.output) << "\n";
#else
    std::cout << "vulkan_status=disabled_at_build_time\n";
#endif

    return 0;
}
