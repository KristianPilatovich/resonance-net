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

# Main executable
add_executable(resonance_net
    src/main.cpp
    src/resonance_net.cpp
    src/rms_norm.cu
    src/embedding.cu
    src/causal_conv.cu
    src/min_gru.cu
    src/slot_memory.cu
    src/swiglu_ffn.cu
    src/optimizer.cu
)

target_include_directories(resonance_net PRIVATE include)
target_link_libraries(resonance_net PRIVATE
    CUDA::cudart
    CUDA::cublas
)

# Allow .cpp to include CUDA headers
set_target_properties(resonance_net PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
