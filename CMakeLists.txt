cmake_minimum_required(VERSION 3.25)
project(KernelImageProcessing LANGUAGES CUDA CXX)

find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(KernelImageProcessing
        main.cpp
        Image/PPMImage.cpp
        Image/PPMImage.h
        sequential/SequentialGaussianBlur.cpp
        sequential/SequentialGaussianBlur.h Results/Result.cpp Results/Result.h parallel/ParallelGaussianBlur.cu parallel/ParallelGaussianBlur.cuh)

target_link_libraries(KernelImageProcessing PRIVATE CUDA::cudart)
set_target_properties(KernelImageProcessing PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        )

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_86,code=sm_86")