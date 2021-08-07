#pragma once

#include <vector>

#include <glm.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define ATTRIBS __device__
#define __ATTRIBS__ __host__ __device__


namespace utils { namespace cuda {

static void handle_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (utils::cuda::handle_error(err, __FILE__, __LINE__ ))


template <typename T>
inline T* cuda_allocate_buffer(T* buffer, const size_t len) {
    HANDLE_ERROR(cudaMalloc((void**)&buffer, len * sizeof(T)));
    return buffer;
}

template <typename T>
inline void cuda_allocate_copy_buffer(T* dev_ptr, T* host_ptr, size_t len) {
    const size_t width = len * sizeof (T);
    HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, width));
    HANDLE_ERROR(cudaMemcpy(dev_ptr, host_ptr, width,
                            cudaMemcpyHostToDevice));
}

template <typename T>
inline T* cuda_managed_array(T* dev_ptr, T* source, const size_t len) {
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMallocManaged(&dev_ptr, len * sizeof (T)));
    std::copy(source, source + len, dev_ptr);
    cudaDeviceSynchronize();
    return dev_ptr;
}

template<typename T, typename ... Args>
inline T* cuda_construct(T* dev_ptr, Args ... args) {
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMallocManaged(&dev_ptr, sizeof (T)));
    T temp(args...);
    *dev_ptr = temp;
    cudaDeviceSynchronize();
    return dev_ptr;
}


template<typename T>
inline T* cuda_copy(T* dev_ptr, T&& temp) {
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMallocManaged(&dev_ptr, sizeof (T)));
    *dev_ptr = temp;
    cudaDeviceSynchronize();
    return dev_ptr;
}

template<typename T>
inline T* cuda_copy(T* dev_ptr, const T& temp, size_t len = 1) {
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMallocManaged(&dev_ptr, len * sizeof (T)));
    *dev_ptr = temp;
    cudaDeviceSynchronize();
    return dev_ptr;
}

template <typename T> struct Collection {
    T* list;
    size_t len;
};

template <typename T>
Collection<T> convert_to_cuda_managed(const std::vector<T>& values) {
    cudaDeviceSynchronize();
    Collection<T> collection;
    collection.len = values.size();
    HANDLE_ERROR(cudaMallocManaged(&collection.list, collection.len * sizeof (T)));
    for (size_t index = 0; index < collection.len; ++index) {
        collection.list[index] = values[index];
    }
    cudaDeviceSynchronize();
    return collection;
}


template<typename T> __host__ __device__ inline T max(T lhs, T rhs) {
    return lhs > rhs ? lhs : rhs;
}

__host__ __device__ inline glm::vec3 reflect(const glm::vec3& income, const glm::vec3& normal) {
    return income - normal * 2.f * glm::dot(income, normal) * normal;
}

__host__ __device__ inline float positive_infinite() {
    return 3.402823466e+38F;
}

}}