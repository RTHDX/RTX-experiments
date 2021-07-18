#include <gtx/string_cast.hpp>
#include <gtest/gtest.h>

#include "Utils.hpp"
#include "CudaUtils.cuh"
#include "renderer/CudaRender.cuh"

#include "cuda_test.cuh"

using namespace utils;
using namespace render;
using namespace render::cuda;


__global__ void camera_status(Camera* camera) {
    assert(camera);

    const auto& pos = camera->position();
    printf("<Camera: position - (%f;%f;%f), WxH - %dx%d>",
           pos.x, pos.y, pos.z, camera->width(), camera->height());
}


int main(int argc, char** argv) {
    Camera native_cam(Point(0.0, 0.0, 0.0), 4, 34, 34);
    Camera* device_cam = nullptr;
    device_cam = utils::cuda::cuda_copy(device_cam, native_cam);
    camera_status<<<1, 1>>>(device_cam);

    return 0;
}