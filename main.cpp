#include <stdio.h>
#include <stdlib.h>

#include "render.hpp"
#include "utils.hpp"

constexpr int WIDTH = 860;
constexpr int HEIGHT = 640;

using namespace render;
using namespace utils;


Objects create_objects() {
    Material red(Color(0.9, 0.1, 0.1),
                 Albedo(0.09, 0.03, 0.0), 50.0f);

    Material green(Color(0.0, 0.9, 0.0),
                   Albedo(0.05, 0.1, 0.05), 40.0f);

    Material ground(Color(0.8, 0.95, 0.9),
                    Albedo(0.03f, 0.001f, 1.5f), 0.5f);

    Material white(Color(0.95, 0.95, 0.95),
                   Albedo(0.1, 0.05, 0.005), 20.0f);

    Material mint(Color(0.1, 0.95, 0.65),
                  Albedo(0.3, 0.1, 0.01), 80.0f);

    return {
        std::make_shared<Sphere>(Point(40.0f, 0.0f, 40.0f), 5.0f,
                                 red),
        std::make_shared<Sphere>(Point(-40.0f, 8.0f, 40.0f), 5.0f,
                                 red),
        std::make_shared<Sphere>(Point(0.0f, 8.0f, 0.0f), 10.0f,
                                 green),
        std::make_shared<Sphere>(Point(25.0, 10.0f, 0.0f), 10.0f,
                                 white),
        std::make_shared<Sphere>(Point(-25.0, 10.0f, 0.0f), 10.0f,
                                 white),
        std::make_shared<Sphere>(Point(0.0f, 10.0f, -50.0f), 40.0f,
                                 mint),
        std::make_shared<Sphere>(Point(0.0f, -10000005.0f, 0.0f),
                                 10000000.0f, ground),
    };
}

const Point camera_location(0.0f, 0.0f, 120.0f);

render::Lights create_lights() {
    return {
        Light(Point(0.0f, 50.0f, 0.0f), 25.0f),
        Light(Point(45.0f, 50.0f, 45.0f), 5.0f),
        Light(Point(-45.0f, 40.0f, 45.0f), 5.0f),
    };
}

Scene create_scene(int width, int height) {
    Scene scene;
    scene.objects = create_objects();
    scene.background = Color(0.05, 0.05, 0.05);
    scene.lights = create_lights();
    scene.width = width;
    scene.height = height;
    return scene;
}

Camera create_camera(int widht, int height) {
    return Camera(camera_location, to_radian(60), widht, height);
}

Render create_render(int width, int height) {
    return Render(
        create_scene(width, height),
        create_camera(width, height)
    );
}

int main(int argc, char** argv) {
    atexit(post_processing);

    GLFWwindow* main_window = load_glfw("pixel buffer render",
                                        WIDTH, HEIGHT);
    load_opengl();

    auto renderer = create_render(WIDTH, HEIGHT);
    std::vector<Color> frame = renderer.render();

    while (!glfwWindowShouldClose(main_window)) {
        glfwSwapBuffers(main_window);
        draw(WIDTH, HEIGHT, fast_convert(frame));
        glfwPollEvents();
    }

    exit(EXIT_SUCCESS);
}