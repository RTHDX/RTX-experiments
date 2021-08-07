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

std::ostream& operator << (std::ostream& os, const Ray& ray) {
    return os << "<Ray: origin(" << ray.origin.x << ";"
              << ray.origin.y << ";" << ray.origin.z << "), direction("
              << ray.direction.x << ";" << ray.direction.y << ";"
              << ray.direction.z << ")>";
}

CameraTest::CameraTest()
    : testing::Test()
    , _camera(Point(0.0, 0.0, 20.0), to_radian(50), 20, 20)
{}


TEST_F(CameraTest, emit_ray) {
    Ray ray_1 = _camera.emit_ray(0, 0);
    Ray ray_2 = _camera.emit_world_ray(0, 0);
    EXPECT_EQ(ray_1.origin, ray_2.origin);
    EXPECT_EQ(ray_1.direction, ray_2.direction);
}


static CudaRender& inst_render(int width = 0, int height = 0) {
    static CudaRender _render = make_render(width, height);
    return _render;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);


int main(int argc, char** argv) {
    atexit(post_processing);
    testing::InitGoogleTest(&argc, argv);
    int test_run_result = RUN_ALL_TESTS();

    const int width = 860;
    const int height = 640;

    GLFWwindow* window = load_glfw("cuda raytracer", width, height);
    glfwSetKeyCallback(window, key_callback);
    load_opengl();

    CudaRender& render = inst_render(width, height);
    while (!glfwWindowShouldClose(window)) {
        render.render();
        glfwSwapBuffers(window);
        render.draw();
        glfwPollEvents();
    }

    exit(test_run_result);
}


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    switch (key) {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, true);
        break;
    case GLFW_KEY_W:
        inst_render().camera()->move_forward();
        break;
    case GLFW_KEY_S:
        inst_render().camera()->move_backward();
        break;
    case GLFW_KEY_A:
        inst_render().camera()->move_left();
        break;
    case GLFW_KEY_D:
        inst_render().camera()->move_right();
        break;
    case GLFW_KEY_UP:
        inst_render().camera()->move_up();
        break;
    case GLFW_KEY_DOWN:
        inst_render().camera()->move_down();
        break;
    default:;
    }
}
