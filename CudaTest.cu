#include <gtx/string_cast.hpp>
#include <gtest/gtest.h>

#include <thrust/device_vector.h>

#include "Utils.hpp"
#include "CudaUtils.cuh"
#include "renderer/CudaRender.cuh"
#include "CudaTest.cuh"

using namespace utils;
using namespace utils::cuda;
using namespace render;
using namespace render::cuda;


__global__ void kernel(int len) {
    static constexpr char TEMPLATE[] =
        "<(%4d/%4d) - "
        " (B.x: %2d, B.y: %2d, B.z: %2d)"
        " (T.x: %2d, T.y: %2d, T.z: %2d)"
        " (D.x: %2d, D.y: %2d, D.z: %2d)"
        " (GD.x: %2d, GD.y: %2d, GD.z: %2d)"
        ">\n";

    const int index = blockIdx.x + (gridDim.x * threadIdx.x);
    printf(TEMPLATE, index, len,
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockDim.x, blockIdx.y, blockIdx.z,
           gridDim.x, gridDim.y, gridDim.z);
}


int main(int argc, char** argv) {
    atexit(post_processing);
    const int width = 860;
    const int height = 640;

    GLFWwindow* window = load_glfw("cuda raytracer", width, height);
    load_opengl();

    CudaRender render = make_render(width, height);
    while (!glfwWindowShouldClose(window)) {
        render.render();
        glfwSwapBuffers(window);
        render.draw();
        glfwPollEvents();
    }

    exit(EXIT_SUCCESS);
}
