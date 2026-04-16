cmake_minimum_required(VERSION 3.24)
project(resonance_net LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

find_package(CUDAToolkit REQUIRED)

option(RESONANCE_NCCL "Build with NCCL multi-GPU data-parallel support" OFF)

set(RNET_SOURCES
    src/main.cpp
    src/resonance_net.cpp
    src/dist.cpp
    src/rms_norm.cu
    src/embedding.cu
    src/causal_conv.cu
    src/min_gru.cu
    src/slot_memory.cu
    src/swiglu_ffn.cu
)

add_executable(resonance_net ${RNET_SOURCES})

target_include_directories(resonance_net PRIVATE include)
target_link_libraries(resonance_net PRIVATE
    CUDA::cudart
    CUDA::cublas
)

if(RESONANCE_NCCL)
    if(NOT NCCL_ROOT)
        set(NCCL_ROOT "$ENV{HOME}/.local/nccl" CACHE PATH "Path to NCCL install")
    endif()
    find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_ROOT}/include REQUIRED)
    find_library(NCCL_LIBRARY NAMES nccl HINTS ${NCCL_ROOT}/lib REQUIRED)
    message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
    message(STATUS "NCCL library: ${NCCL_LIBRARY}")
    target_include_directories(resonance_net PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(resonance_net PRIVATE ${NCCL_LIBRARY})
    target_compile_definitions(resonance_net PRIVATE RNET_NCCL=1)
    set_target_properties(resonance_net PROPERTIES
        INSTALL_RPATH "${NCCL_ROOT}/lib"
        BUILD_RPATH   "${NCCL_ROOT}/lib"
    )
endif()

# Allow .cpp to include CUDA headers
set_target_properties(resonance_net PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
