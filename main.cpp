#include <stdio.h>
#include <stdlib.h>

#include "render.hpp"
#include "utils.hpp"

constexpr int WIDTH = 860;
constexpr int HEIGHT = 640;

render::Objects create_objects() {
    render::Material red(render::Color(0.9, 0.0, 0.0), render::Albedo(0.3, 0.5, 0.5));
    render::Material green(render::Color(0.0, 0.9, 0.0), render::Albedo(0.3, 0.5, 0.5));
    render::Material blue(render::Color(0.0, 0.0, 0.9), render::Albedo(0.95, 0.5, 0.5));

    return {
        std::make_shared<render::Sphere>(render::Point(0.0f, 0.0f, 0.0f), 3.0f,
                                         red),
        std::make_shared<render::Sphere>(render::Point(0.0f, 5.0f, -12.0f), 10.0f,
                                         green),
        std::make_shared<render::Sphere>(render::Point(0.0f, -100000.0f, 0.0f), 100000.0f,
                                         blue),
    };
}

render::Lights create_lights() {
    return {
        render::Light(render::Point(40.0f, 40.0f, 40.0f), 2.0f),
        render::Light(render::Point(-40.0f, 40.0f, 40.0f), 2.0f)
    };
}

render::Scene create_scene(int width, int height) {
    render::Scene scene;
    scene.objects = create_objects();
    scene.background = render::Color(0.8, 0.85, 0.8);
    scene.lights = create_lights();
    scene.width = width;
    scene.height = height;
    return scene;
}

render::Render create_render(int width, int height) {
    return render::Render(
        create_scene(width, height),
        render::Camera(render::Point(0.0, 0.0, 30.0), utils::to_radian(50),
                       width, height)
    );
}

int main(int argc, char** argv) {
    atexit(utils::post_processing);

    GLFWwindow* main_window = utils::load_glfw("pixel buffer render",
                                               WIDTH, HEIGHT);
    utils::load_opengl();

    auto renderer = create_render(WIDTH, HEIGHT);
    std::vector<render::Color> frame = renderer.render();

    while (!glfwWindowShouldClose(main_window)) {
        glfwSwapBuffers(main_window);
        utils::draw(WIDTH, HEIGHT, utils::fast_convert(frame));
        glfwPollEvents();
    }

    exit(EXIT_SUCCESS);
}