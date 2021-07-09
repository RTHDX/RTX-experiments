#include <gtx/string_cast.hpp>
#include <gtest/gtest.h>

#include "utils.hpp"
#include "test.hpp"

using namespace utils;
using namespace render;

constexpr int DEFAULT_DIM = 20;

static inline Scene make_scene(int width = DEFAULT_DIM,
                               int height = DEFAULT_DIM) {
    Material material_1(Color(1.0, 1.0, 1.0), Albedo(0.1, 0.1, 0.1), 20.0);
    Material material_2(Color(0.9, 0.9, 0.9), Albedo(0.1, 0.1, 0.1), 30.0);
    Scene scene;
    scene.objects = {
        std::make_shared<Sphere>(Point(0.0, 0.0, 0.0), 1.5f, material_1),
        std::make_shared<Sphere>(Point(0.0, -100005.0, 0.0), 100000.0,
                                 material_2)
    };
    scene.background = utils::BLACK;
    scene.height = height;
    scene.width = width;
    scene.lights = {
        Light(Point(0.0, 40.0, 0.0), 10.0f),
    };
    return scene;
}

Camera make_camera(int widht = DEFAULT_DIM, int height = DEFAULT_DIM) {
    return Camera(Point(0.0, 0.0, 10.0), to_radian(60),
                  widht, height);
}

Render make_render(int width = DEFAULT_DIM, int height = DEFAULT_DIM) {
    return Render(make_scene(width, height),
                  make_camera(width, height));
}


ShadowTest::ShadowTest()
    : testing::Test()
    , scene(make_scene())
    , camera(make_camera())
    , render(scene, camera)
{}


TEST_F(ShadowTest, light_direction) {
    Point p(0.0f, 0.0f, 0.0f);
    Vector dir = scene.lights[0].direction(p);
    EXPECT_EQ(dir, Vector(0.0f, -1.0f, 0.0f));
}

TEST_F(ShadowTest, hit_info) {
    Ray ray = camera.emit_ray(DEFAULT_DIM / 2, DEFAULT_DIM / 2);
    Hit hit_1 = scene.objects[0]->hit(ray);
    EXPECT_TRUE(hit_1.is_hitted);
    std::cout << hit_1 << std::endl;
    Hit hit_2 = scene.objects[1]->hit(ray);
    EXPECT_FALSE(hit_2.is_hitted);
}

TEST_F(ShadowTest, shadow_ray) {
    //Ray primary = camera.emit_ray(8, 18);
    //Color mid = render.trace(primary);
    //std::cout << glm::to_string(mid) << std::endl;
    //std::cout << primary << std::endl;
    //render.render();
}



int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    atexit(post_processing);
    int test_run_results = RUN_ALL_TESTS();

    const int width = 300;
    const int height = 300;
    GLFWwindow* main_window = load_glfw("TEST", width, height);
    load_opengl();
    Render render = make_render(width, height);
    auto frame = render.render();
    while (!glfwWindowShouldClose(main_window)) {
        glfwSwapBuffers(main_window);
        draw(width, height, fast_convert(frame));
        glfwPollEvents();
    }

    exit(test_run_results);
}