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

void dump_rays(const Ray& ray_1, const Ray& ray_2) {
    std::cout << " - " << ray_1 << std::endl << " - " << ray_2 << std::endl;
}

std::ostream& operator << (std::ostream& os, const Hit& hit) {
    static constexpr char MISSED_TEMPLATE[] = "<Hit: is_hitted - 0>";
    static constexpr char HITTED_TEMPLATE[] =
        "<Hit: t_near - %3.5f, t_far - %3.5f, point - (%.3f;%.3f;%.3f)"
        ", normal - (%.3f; %.3f; %.3f)>";
    static char BUFFER[256];

    size_t count = 0;
    if (!hit.is_hitted()) {
        count = sprintf(BUFFER, MISSED_TEMPLATE);
    } else {
        const Point& point = hit.point();
        const Vector& normal = hit.normal();
        count = sprintf(BUFFER, HITTED_TEMPLATE, hit.t_near(), hit.t_far(),
                        point.x, point.y, point.z, normal.x, normal.y, normal.z);
    }
    os.write(BUFFER, count);
    return os;
}

void dump_hits(const Hit& hit_1, const Hit& hit_2) {
    std::cout << " - " << hit_1 << std::endl << " - " << hit_2 << std::endl;
}

Point CameraTest::z_positive(0.0, 0.0, 20.0);
Point CameraTest::z_negative(0.0, 0.0, -20.0);
Point CameraTest::x_positive(20.0, 0.0, 0.0);
Point CameraTest::x_negative(-20.0, 0.0, 0.0);
Point CameraTest::y_negative(0.0, -20.0, 0.0);
Point CameraTest::y_positive(0.0, 20.0, 0.0);

CameraTest::CameraTest()
    : testing::Test()
    , _camera(Point(0.0, 0.0, 20.0), Point(0.0, 0.0, 0.0), to_radian(50), 20, 20)
{}


TEST_F(CameraTest, emit_ray) {
    Ray ray_1 = _camera.emit_ray(0, 0);
    std::cout << ray_1 << std::endl;
}

TEST_F(CameraTest, hit_sphere) {
    Sphere sphere(Point(0.0, 0.0, 0.0), 5.0, Material());
    Ray ray = _camera.emit_ray(0, 0);
    std::cout << ray << std::endl;
    Hit hit = sphere.hit(ray);
    std::cout << hit << std::endl;
    EXPECT_FALSE(hit.is_hitted());

    std::cout << std::endl;
    ray = _camera.emit_ray(_camera.width() / 2, _camera.height() / 2);
    std::cout << ray << std::endl;
    hit = sphere.hit(ray);
    std::cout << hit << std::endl;
    EXPECT_TRUE(hit.is_hitted());
}

TEST_F(CameraTest, update_position) {
    Sphere sphere(Point(0.0, 0.0, 0.0), 5.0, Material());
    int w_pos = _camera.width() / 2;
    int h_pos = _camera.height() / 2;
    
    Ray ray = _camera.emit_ray(w_pos, h_pos);
    EXPECT_TRUE(sphere.hit(ray).is_hitted());

    _camera.update_position(x_positive);
    ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());

    _camera.update_position(x_negative);
    ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());

    _camera.update_position(y_positive);
    ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());

    _camera.update_position(y_negative);
    ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());

    _camera.update_position(z_positive);
    ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());
    
    _camera.update_position(z_negative);
    ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());

    _camera.update_position(x_positive + y_positive + z_positive);
    _camera.dump();
    ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());
}

TEST_F(CameraTest, move_camera_near_z) {
    Sphere sphere(Point(0.0, 0.0, 0.0), 5.0, Material());
    int h_pos = _camera.height() / 2;
    int w_pos = _camera.width() / 2;

    _camera.move_left();
    _camera.dump();
    Ray ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());

    _camera.move_right();
    _camera.move_right();
    _camera.dump();
    ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());

    _camera.update_position(z_positive);
    _camera.move_up();
    _camera.dump();
    ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());

    for (size_t i = 0; i < 100; ++i) { _camera.move_down(); }
    EXPECT_TRUE(std::fabs(std::fabs(_camera.position().y) - 49.5) < 0.0000001);
    ray = _camera.emit_ray(w_pos, h_pos);
    std::cout << ray << std::endl;
    EXPECT_TRUE(sphere.hit(ray).is_hitted());
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
    case GLFW_KEY_KP_8:
        inst_render().camera()->update_position(Point(0.0, 100.0, 0.0));
        break;
    case GLFW_KEY_KP_2:
        inst_render().camera()->update_position(Point(0.0, -100.0, 0.0));
        break;
    case GLFW_KEY_KP_5: {
        if (action != GLFW_PRESS) return;

        static bool is_front = true;
        is_front ?
            inst_render().camera()->update_position(Point(0.0, 0.0, -100.0)) :
            inst_render().camera()->update_position(Point(0.0, 0.0, 100.0));
        is_front = !is_front;
        break;
    } case GLFW_KEY_KP_4:
        inst_render().camera()->update_position(Point(-100.0, 0.0, 0.0));
        break;
    case GLFW_KEY_KP_6:
        inst_render().camera()->update_position(Point(100.0, 0.0, 0.0));
        break;
    default:
        std::cout << key << std::endl;
    }
}
