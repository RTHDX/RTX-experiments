#include <stdio.h>
#include <stdlib.h>

#include "render.hpp"
#include "utils.hpp"

constexpr int WIDTH = 860;
constexpr int HEIGHT = 680;

using namespace render;
using namespace utils;


Objects create_objects() {
    Material red(Color(0.9, 0.0, 0.0),
                       Albedo(0.04, 0.3, 0.5), 59.0f);

    Material green(Color(0.0, 0.9, 0.0),
                         Albedo(0.05, 0.1, 0.5), 40.0f);

    Material ground(Color(0.3, 0.3, 0.3),
                    Albedo(0.01, 0.01, 0.5), 10.0f);

    return {
        //std::make_shared<Sphere>(Point(0.0f, 0.0f, 1.0f), 3.0f,
        //                         red),
        std::make_shared<Sphere>(Point(0.0f, 5.0f, -12.0f), 10.0f,
                                 green),
        std::make_shared<Sphere>(Point(0.0f, -10000000.0f, 0.0f),
                                 10000000.0f, ground),
    };
}

render::Lights create_lights() {
    return {
        Light(Point(0.0f, 100.0f, 50.0f), 30.0f),
        Light(Point(40.0f, 40.0f, 50.0f), 5.0f),
        //Light(Point(-40.0f, 40.0f, 50.0f), 5.0f)
    };
}

Scene create_scene(int width, int height) {
    Scene scene;
    scene.objects = create_objects();
    scene.background = Color(0.8, 0.85, 0.8);
    scene.lights = create_lights();
    scene.width = width;
    scene.height = height;
    return scene;
}

Render create_render(int width, int height) {
    return Render(
        create_scene(width, height),
        Camera(Point(0.0, 0.0, 30.0), to_radian(50), width, height)
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