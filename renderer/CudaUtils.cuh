#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gtx/string_cast.hpp>


namespace utils { namespace cuda {

static void handle_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (utils::cuda::handle_error(err, __FILE__, __LINE__ ))


template<typename T>
static inline T* cuda_allocate(T* buffer, const size_t len) {
    HANDLE_ERROR(cudaMalloc((void**)&buffer, len * sizeof(T)));
    return buffer;
}


template<typename T, typename ... Args>
static inline T* cuda_construct(T* dev_ptr, Args ... args) {
    HANDLE_ERROR(cudaMallocManaged(&dev_ptr, sizeof (T)));
    T temp(args...);
    *dev_ptr = temp;
    return dev_ptr;
}


template<typename T>
static inline T* cuda_copy(T* dev_ptr, T&& temp) {
    HANDLE_ERROR(cudaMallocManaged(&dev_ptr, sizeof (T)));
    *dev_ptr = temp;
    return dev_ptr;
}

template<typename T>
inline T* cuda_copy(T* dev_ptr, const T& temp) {
    HANDLE_ERROR(cudaMallocManaged(&dev_ptr, sizeof (T)));
    *dev_ptr = temp;
    return dev_ptr;
}

template<typename T> __device__ inline T max(T lhs, T rhs) {
    return lhs > rhs ? lhs : rhs;
}

__device__ inline glm::vec3 reflect(const glm::vec3& income, const glm::vec3& normal) {
    return income - normal * 2.f * glm::dot(income, normal) * normal;
}

}}