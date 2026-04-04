if(NOT DEFINED LADA_NCNN_NET_CPP)
    message(FATAL_ERROR "LADA_NCNN_NET_CPP must point at ncnn/src/net.cpp.")
endif()

if(NOT EXISTS "${LADA_NCNN_NET_CPP}")
    message(FATAL_ERROR "ncnn net.cpp not found: ${LADA_NCNN_NET_CPP}")
endif()

file(READ "${LADA_NCNN_NET_CPP}" _lada_ncnn_net_cpp)

if(_lada_ncnn_net_cpp MATCHES "Shared external VkCompute commands may record multiple extractor graphs")
    message(STATUS "ncnn shared VkCompute benchmark patch already applied: ${LADA_NCNN_NET_CPP}")
    return()
endif()

set(_lada_ncnn_old_submit_block [=[#if NCNN_BENCHMARK
        std::vector<uint64_t> results(layer_index * 2);
        cmd.get_query_pool_results(0, layer_index * 2, results);
        for (int i = 0; i < layer_index; i++)
        {
            uint64_t start = results[i * 2];
            uint64_t end = results[i * 2 + 1];
            if (start == 0 || end == 0)
                continue;

            double duration_us = (end - start) * vkdev->info.timestamp_period() / 1000;
            NCNN_LOGE("%-24s %-30s %8.2lfus    |", layers[i]->type.c_str(), layers[i]->name.c_str(), duration_us);
        }
#endif // NCNN_BENCHMARK]=])

set(_lada_ncnn_new_submit_block [=[#if NCNN_BENCHMARK
        // Shared external VkCompute commands may record multiple extractor graphs into the same
        // command buffer. The built-in benchmark path assumes one net/query-pool per command and
        // becomes invalid on this path, so keep the mid-command benchmark logic disabled here.
#endif // NCNN_BENCHMARK]=])

set(_lada_ncnn_old_timestamp_begin [=[#if NCNN_BENCHMARK
        cmd.record_write_timestamp(layer_index * 2);
#endif]=])

set(_lada_ncnn_new_timestamp_begin [=[#if NCNN_BENCHMARK
        // See note above: external shared VkCompute paths disable net-level benchmark timestamps.
#endif]=])

set(_lada_ncnn_old_timestamp_end [=[#if NCNN_BENCHMARK
        cmd.record_write_timestamp(layer_index * 2 + 1);
#endif]=])

set(_lada_ncnn_new_timestamp_end [=[#if NCNN_BENCHMARK
        // See note above: external shared VkCompute paths disable net-level benchmark timestamps.
#endif]=])

string(FIND "${_lada_ncnn_net_cpp}" "${_lada_ncnn_old_submit_block}" _lada_ncnn_submit_idx)
if(_lada_ncnn_submit_idx EQUAL -1)
    message(FATAL_ERROR "Failed to locate ncnn shared VkCompute benchmark submit block in ${LADA_NCNN_NET_CPP}")
endif()
string(REPLACE "${_lada_ncnn_old_submit_block}" "${_lada_ncnn_new_submit_block}" _lada_ncnn_net_cpp "${_lada_ncnn_net_cpp}")

string(FIND "${_lada_ncnn_net_cpp}" "${_lada_ncnn_old_timestamp_begin}" _lada_ncnn_begin_idx)
if(_lada_ncnn_begin_idx EQUAL -1)
    message(FATAL_ERROR "Failed to locate ncnn shared VkCompute benchmark begin timestamp block in ${LADA_NCNN_NET_CPP}")
endif()
string(REPLACE "${_lada_ncnn_old_timestamp_begin}" "${_lada_ncnn_new_timestamp_begin}" _lada_ncnn_net_cpp "${_lada_ncnn_net_cpp}")

string(FIND "${_lada_ncnn_net_cpp}" "${_lada_ncnn_old_timestamp_end}" _lada_ncnn_end_idx)
if(_lada_ncnn_end_idx EQUAL -1)
    message(FATAL_ERROR "Failed to locate ncnn shared VkCompute benchmark end timestamp block in ${LADA_NCNN_NET_CPP}")
endif()
string(REPLACE "${_lada_ncnn_old_timestamp_end}" "${_lada_ncnn_new_timestamp_end}" _lada_ncnn_net_cpp "${_lada_ncnn_net_cpp}")

file(WRITE "${LADA_NCNN_NET_CPP}" "${_lada_ncnn_net_cpp}")
message(STATUS "Applied ncnn shared VkCompute benchmark patch: ${LADA_NCNN_NET_CPP}")
