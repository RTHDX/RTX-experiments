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
