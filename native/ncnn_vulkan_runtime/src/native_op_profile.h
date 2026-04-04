#ifndef LADA_NATIVE_OP_PROFILE_H
#define LADA_NATIVE_OP_PROFILE_H

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#if NCNN_VULKAN && NCNN_BENCHMARK
#include "command.h"
#include "gpu.h"
#endif

namespace lada {

enum class NativeOpKind
{
    DeformConv,
    GridSample,
};

enum class NativeOpBackend
{
    Cpu,
    Vulkan,
};

struct NativeOpProfileSnapshot
{
    double deformconv_cpu_s = 0.0;
    double deformconv_vulkan_s = 0.0;
    double deformconv_vulkan_gpu_s = 0.0;
    double gridsample_cpu_s = 0.0;
    double gridsample_vulkan_s = 0.0;
    double gridsample_vulkan_gpu_s = 0.0;
    std::uint64_t deformconv_cpu_count = 0;
    std::uint64_t deformconv_vulkan_count = 0;
    std::uint64_t gridsample_cpu_count = 0;
    std::uint64_t gridsample_vulkan_count = 0;
};

namespace detail {

inline std::atomic<std::uint64_t> g_deformconv_cpu_time_ns{0};
inline std::atomic<std::uint64_t> g_deformconv_vulkan_time_ns{0};
inline std::atomic<std::uint64_t> g_deformconv_vulkan_gpu_time_ns{0};
inline std::atomic<std::uint64_t> g_gridsample_cpu_time_ns{0};
inline std::atomic<std::uint64_t> g_gridsample_vulkan_time_ns{0};
inline std::atomic<std::uint64_t> g_gridsample_vulkan_gpu_time_ns{0};
inline std::atomic<std::uint64_t> g_deformconv_cpu_count{0};
inline std::atomic<std::uint64_t> g_deformconv_vulkan_count{0};
inline std::atomic<std::uint64_t> g_gridsample_cpu_count{0};
inline std::atomic<std::uint64_t> g_gridsample_vulkan_count{0};

inline std::atomic<std::uint64_t>& time_counter(NativeOpKind kind, NativeOpBackend backend)
{
    if (kind == NativeOpKind::DeformConv) {
        return backend == NativeOpBackend::Cpu ? g_deformconv_cpu_time_ns : g_deformconv_vulkan_time_ns;
    }
    return backend == NativeOpBackend::Cpu ? g_gridsample_cpu_time_ns : g_gridsample_vulkan_time_ns;
}

inline std::atomic<std::uint64_t>& count_counter(NativeOpKind kind, NativeOpBackend backend)
{
    if (kind == NativeOpKind::DeformConv) {
        return backend == NativeOpBackend::Cpu ? g_deformconv_cpu_count : g_deformconv_vulkan_count;
    }
    return backend == NativeOpBackend::Cpu ? g_gridsample_cpu_count : g_gridsample_vulkan_count;
}

inline std::atomic<std::uint64_t>& gpu_time_counter(NativeOpKind kind)
{
    return kind == NativeOpKind::DeformConv ? g_deformconv_vulkan_gpu_time_ns : g_gridsample_vulkan_gpu_time_ns;
}

inline double ns_to_seconds(std::uint64_t value)
{
    return static_cast<double>(value) / 1'000'000'000.0;
}

inline bool native_gpu_timestamps_enabled()
{
    static const bool enabled = []() {
        const char* value = std::getenv("LADA_NCNN_NATIVE_GPU_TIMESTAMPS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

inline std::uint32_t native_gpu_query_capacity()
{
    static const std::uint32_t capacity = []() {
        constexpr std::uint32_t kDefaultCapacity = 4096;
        constexpr std::uint32_t kMinCapacity = 2;
        constexpr std::uint32_t kMaxCapacity = 65536;

        const char* value = std::getenv("LADA_NCNN_NATIVE_GPU_QUERY_CAPACITY");
        if (value == nullptr || value[0] == '\0') {
            return kDefaultCapacity;
        }

        char* end = nullptr;
        unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || (end != nullptr && *end != '\0')) {
            return kDefaultCapacity;
        }

        std::uint32_t clamped = static_cast<std::uint32_t>(
            std::min<unsigned long>(std::max<unsigned long>(parsed, kMinCapacity), kMaxCapacity));
        if ((clamped & 1u) != 0u) {
            clamped -= 1u;
        }
        return clamped < kMinCapacity ? kMinCapacity : clamped;
    }();
    return capacity;
}

#if NCNN_VULKAN && NCNN_BENCHMARK
struct CommandQueryRange
{
    NativeOpKind kind;
    std::uint32_t start_query = 0;
};

struct CommandQueryState
{
    std::uint32_t query_capacity = 0;
    std::uint32_t next_query = 0;
    std::vector<CommandQueryRange> ranges;
};

inline std::mutex g_command_query_mutex;
inline std::unordered_map<const ncnn::VkCompute*, CommandQueryState> g_command_queries;

inline std::uint32_t reserve_query_range(ncnn::VkCompute& cmd, NativeOpKind kind)
{
    std::lock_guard<std::mutex> lock(g_command_query_mutex);
    CommandQueryState& state = g_command_queries[&cmd];
    if (state.query_capacity == 0) {
        const std::uint32_t query_capacity = native_gpu_query_capacity();
        if (cmd.create_query_pool(query_capacity) != 0) {
            g_command_queries.erase(&cmd);
            return UINT32_MAX;
        }
        state.query_capacity = query_capacity;
    }

    if (state.next_query + 2 > state.query_capacity) {
        return UINT32_MAX;
    }

    const std::uint32_t start_query = state.next_query;
    state.next_query += 2;
    state.ranges.push_back(CommandQueryRange{kind, start_query});
    return start_query;
}

inline void finalize_command_queries(ncnn::VkCompute& cmd, const ncnn::VulkanDevice* vkdev)
{
    CommandQueryState state;
    {
        std::lock_guard<std::mutex> lock(g_command_query_mutex);
        const auto it = g_command_queries.find(&cmd);
        if (it == g_command_queries.end()) {
            return;
        }
        state = std::move(it->second);
        g_command_queries.erase(it);
    }

    if (state.next_query == 0) {
        return;
    }

    std::vector<std::uint64_t> results(state.next_query);
    if (cmd.get_query_pool_results(0, state.next_query, results) != 0) {
        return;
    }

    const double timestamp_period_ns = static_cast<double>(vkdev->info.timestamp_period());
    for (const CommandQueryRange& range : state.ranges) {
        const std::uint64_t start = results[range.start_query];
        const std::uint64_t end = results[range.start_query + 1];
        if (start == 0 || end == 0 || end < start) {
            continue;
        }

        const auto elapsed_ns = static_cast<std::uint64_t>(
            static_cast<double>(end - start) * timestamp_period_ns);
        gpu_time_counter(range.kind).fetch_add(elapsed_ns, std::memory_order_relaxed);
    }
}
#endif

} // namespace detail

inline void reset_native_op_profile()
{
    detail::g_deformconv_cpu_time_ns.store(0, std::memory_order_relaxed);
    detail::g_deformconv_vulkan_time_ns.store(0, std::memory_order_relaxed);
    detail::g_deformconv_vulkan_gpu_time_ns.store(0, std::memory_order_relaxed);
    detail::g_gridsample_cpu_time_ns.store(0, std::memory_order_relaxed);
    detail::g_gridsample_vulkan_time_ns.store(0, std::memory_order_relaxed);
    detail::g_gridsample_vulkan_gpu_time_ns.store(0, std::memory_order_relaxed);
    detail::g_deformconv_cpu_count.store(0, std::memory_order_relaxed);
    detail::g_deformconv_vulkan_count.store(0, std::memory_order_relaxed);
    detail::g_gridsample_cpu_count.store(0, std::memory_order_relaxed);
    detail::g_gridsample_vulkan_count.store(0, std::memory_order_relaxed);
#if NCNN_VULKAN && NCNN_BENCHMARK
    std::lock_guard<std::mutex> lock(detail::g_command_query_mutex);
    detail::g_command_queries.clear();
#endif
}

inline NativeOpProfileSnapshot snapshot_native_op_profile()
{
    NativeOpProfileSnapshot snapshot;
    snapshot.deformconv_cpu_s = detail::ns_to_seconds(
        detail::g_deformconv_cpu_time_ns.load(std::memory_order_relaxed));
    snapshot.deformconv_vulkan_s = detail::ns_to_seconds(
        detail::g_deformconv_vulkan_time_ns.load(std::memory_order_relaxed));
    snapshot.deformconv_vulkan_gpu_s = detail::ns_to_seconds(
        detail::g_deformconv_vulkan_gpu_time_ns.load(std::memory_order_relaxed));
    snapshot.gridsample_cpu_s = detail::ns_to_seconds(
        detail::g_gridsample_cpu_time_ns.load(std::memory_order_relaxed));
    snapshot.gridsample_vulkan_s = detail::ns_to_seconds(
        detail::g_gridsample_vulkan_time_ns.load(std::memory_order_relaxed));
    snapshot.gridsample_vulkan_gpu_s = detail::ns_to_seconds(
        detail::g_gridsample_vulkan_gpu_time_ns.load(std::memory_order_relaxed));
    snapshot.deformconv_cpu_count = detail::g_deformconv_cpu_count.load(std::memory_order_relaxed);
    snapshot.deformconv_vulkan_count = detail::g_deformconv_vulkan_count.load(std::memory_order_relaxed);
    snapshot.gridsample_cpu_count = detail::g_gridsample_cpu_count.load(std::memory_order_relaxed);
    snapshot.gridsample_vulkan_count = detail::g_gridsample_vulkan_count.load(std::memory_order_relaxed);
    return snapshot;
}

class ScopedNativeOpTimer
{
public:
    ScopedNativeOpTimer(NativeOpKind kind, NativeOpBackend backend)
        : kind_(kind), backend_(backend), started_at_(Clock::now())
    {
    }

    ~ScopedNativeOpTimer()
    {
        const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - started_at_);
        detail::time_counter(kind_, backend_).fetch_add(
            static_cast<std::uint64_t>(elapsed_ns.count()),
            std::memory_order_relaxed);
        detail::count_counter(kind_, backend_).fetch_add(1, std::memory_order_relaxed);
    }

private:
    using Clock = std::chrono::steady_clock;

    NativeOpKind kind_;
    NativeOpBackend backend_;
    Clock::time_point started_at_;
};

class ScopedNativeOpGpuTimestampQuery
{
public:
#if NCNN_VULKAN && NCNN_BENCHMARK
    ScopedNativeOpGpuTimestampQuery(NativeOpKind kind, ncnn::VkCompute& cmd)
        : cmd_(nullptr), query_index_(UINT32_MAX)
    {
        if (!detail::native_gpu_timestamps_enabled()) {
            return;
        }

        cmd_ = &cmd;
        query_index_ = detail::reserve_query_range(cmd, kind);
        if (query_index_ != UINT32_MAX) {
            cmd_->record_write_timestamp(query_index_);
        }
    }

    ~ScopedNativeOpGpuTimestampQuery()
    {
        if (cmd_ != nullptr && query_index_ != UINT32_MAX) {
            cmd_->record_write_timestamp(query_index_ + 1);
        }
    }
#else
    ScopedNativeOpGpuTimestampQuery(NativeOpKind kind, void* cmd = nullptr)
    {
        (void)kind;
        (void)cmd;
    }
#endif

private:
#if NCNN_VULKAN && NCNN_BENCHMARK
    ncnn::VkCompute* cmd_ = nullptr;
    std::uint32_t query_index_ = UINT32_MAX;
#endif
};

#if NCNN_VULKAN && NCNN_BENCHMARK
inline void finalize_native_op_gpu_profile(ncnn::VkCompute& cmd, const ncnn::VulkanDevice* vkdev)
{
    detail::finalize_command_queries(cmd, vkdev);
}
#else
inline void finalize_native_op_gpu_profile(void* cmd, const void* vkdev)
{
    (void)cmd;
    (void)vkdev;
}
#endif

} // namespace lada

#endif // LADA_NATIVE_OP_PROFILE_H
