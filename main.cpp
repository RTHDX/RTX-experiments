#include <stdio.h>
#include <stdlib.h>

#include "glm/glm/gtc/type_ptr.hpp"

#include "render.hpp"
#include "utils.hpp"

constexpr int WIDTH = 860;
constexpr int HEIGHT = 640;

render::Scene create_scene(int width, int height) {
    render::Scene scene;
    scene.objects = {
        std::make_shared<render::Sphere>(render::Point(0.0f, 0.0f, 0.0f), 3.0f)
    };
    scene.background = render::Color(0.8, 0.85, 0.8);
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
    static constexpr size_t BUFF_SIZE = WIDTH * HEIGHT * 3;
    atexit(utils::post_processing);

    GLFWwindow* main_window = utils::load_glfw("pixel buffer render",
                                               WIDTH, HEIGHT);
    utils::load_opengl();

    auto renderer = create_render(WIDTH, HEIGHT);
    float* buffer = utils::covert(renderer.render());

    while (!glfwWindowShouldClose(main_window)) {
        glfwSwapBuffers(main_window);
        utils::draw(WIDTH, HEIGHT, buffer);
        glfwPollEvents();
    }

    delete [] buffer;
    exit(EXIT_SUCCESS);
}