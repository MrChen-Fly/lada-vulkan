cmake_minimum_required(VERSION 3.24)

project(lada_ncnn_vulkan_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LADA_NCNN_WITH_VULKAN "Build bundled ncnn with Vulkan enabled." OFF)
option(LADA_NCNN_BUILD_LOCAL_RUNTIME "Build a local ncnn Python runtime with the custom layer built into the same module." ON)
set(LADA_NCNN_SOURCE_DIR "" CACHE PATH "Path to an existing ncnn source tree.")
set(_lada_ncnn_already_added OFF)

set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_PYTHON OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN ${LADA_NCNN_WITH_VULKAN} CACHE BOOL "" FORCE)

if(LADA_NCNN_SOURCE_DIR)
    set(_lada_ncnn_source_dir "${LADA_NCNN_SOURCE_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(
        ncnn_src
        GIT_REPOSITORY https://github.com/Tencent/ncnn.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(ncnn_src)
    set(_lada_ncnn_source_dir "${ncnn_src_SOURCE_DIR}")
    set(_lada_ncnn_already_added ON)
endif()

if(NCNN_BENCHMARK AND NCNN_VULKAN)
    set(LADA_NCNN_NET_CPP "${_lada_ncnn_source_dir}/src/net.cpp")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_ncnn_shared_vulkan_benchmark.cmake")
endif()

if(NOT _lada_ncnn_already_added)
    add_subdirectory("${_lada_ncnn_source_dir}" "${CMAKE_BINARY_DIR}/ncnn" EXCLUDE_FROM_ALL)
endif()

add_library(lada_torchvision_deformconv STATIC
    src/lada_gridsample_layer.cpp
    src/torch_conv2d_layer.cpp
    src/lada_yolo_seg_postprocess_layer.cpp
    src/lada_yolo_attention_layer.cpp
    src/torchvision_deform_conv2d_layer.cpp
    src/yolo_seg_postprocess_runner.cpp
)
target_include_directories(lada_torchvision_deformconv PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(lada_torchvision_deformconv PUBLIC ncnn)

add_executable(lada_ncnn_deformconv_bench
    src/benchmark_main.cpp
)
target_link_libraries(lada_ncnn_deformconv_bench PRIVATE
    lada_torchvision_deformconv
    ncnn
)

if(LADA_NCNN_BUILD_LOCAL_RUNTIME)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    add_subdirectory("${_lada_ncnn_source_dir}/python/pybind11" "${CMAKE_BINARY_DIR}/pybind11" EXCLUDE_FROM_ALL)
endif()

if(LADA_NCNN_BUILD_LOCAL_RUNTIME)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set(LADA_LOCAL_PYNCNN_MAIN_INPUT "${_lada_ncnn_source_dir}/python/src/main.cpp")
    set(LADA_LOCAL_PYNCNN_MAIN_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/ncnn_local_main.cpp")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_local_pyncnn_main.cmake")

    pybind11_add_module(lada_ncnn_local_runtime MODULE
        "${LADA_LOCAL_PYNCNN_MAIN_OUTPUT}"
        src/basicvsrpp_clip_runner.cpp
        src/python_local_runtime_bindings.cpp
        src/vulkan_blend_runtime.cpp
    )
    target_link_libraries(lada_ncnn_local_runtime PRIVATE
        lada_torchvision_deformconv
        ncnn
    )
    target_include_directories(lada_ncnn_local_runtime PRIVATE
        "${_lada_ncnn_source_dir}/python/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    set_target_properties(lada_ncnn_local_runtime PROPERTIES
        PREFIX ""
        OUTPUT_NAME "ncnn"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/local_runtime"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/local_runtime"
    )
endif()
