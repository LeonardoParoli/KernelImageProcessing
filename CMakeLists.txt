cmake_minimum_required(VERSION 3.25)
project(KernelImageProcessing)

set(CMAKE_CXX_STANDARD 17)

add_executable(KernelImageProcessing main.cpp Image/PPMImage.cpp Image/PPMImage.h sequential/SequentialGaussianBlur.cpp sequential/SequentialGaussianBlur.h)