cmake_minimum_required (VERSION 3.18)
project(renderer LANGUAGES CXX CUDA)

create_library(
    TARGET ${PROJECT_NAME}
    SOURCES NativeRender.cpp BaseRender.cpp Utils.cpp CudaRender.cu
            Camera.cu
    HEADERS CudaUtils.cuh
)
