#include "Utils.hpp"


namespace utils {

void fill_frame(float* buffer, size_t size, float r, float g, float b) {
    for (size_t i = 0; i < size; i += 3) {
        buffer[i] = r;
        buffer[i + 1] = g;
        buffer[i + 2] = b;
    }
}


GLFWwindow* load_glfw(const char* title, int width, int height) {
    if (glfwInit() == GLFW_FALSE) {
        fprintf(stderr, "[Window] Unable to load glfw\n");
        exit(EXIT_FAILURE);
    }

    GLFWwindow* window = glfwCreateWindow(width, height, title,
                                          nullptr, nullptr);
    if (window == nullptr) {
        fprintf(stderr, "[Window] Unable to create window\n");
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwMakeContextCurrent(window);
    return window;
}


void load_opengl() {
    if (!gladLoadGL()) {
        fprintf(stderr, "[Window] Unnable to load OpenGL function\n");
        exit(EXIT_FAILURE);
    }

    int major, minor, profile_mask;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile_mask);
    printf("[Window] OpenGL contenx loaded:\n");
    printf(" - major %d\n", major);
    printf(" - minor %d\n", minor);
    printf(" - vedor %s\n", glGetString(GL_VENDOR));
    printf(" - renderer %s\n", glGetString(GL_RENDERER));
    printf(" - shading language %s\n",
           glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf(" - profile mask %d\n", profile_mask);
}


void post_processing() {
    printf("At exit");
    glfwTerminate();
}


void draw(int width, int height, float* frame) {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(width, height, GL_RGB, GL_FLOAT, frame);
}


float to_radian(float degree) {
    return degree * float(M_PI / 180);
}


float* covert(const std::vector<glm::vec3>& frame) {
    const size_t len = frame.size() * 3;
    float* buffer = new float[len];
    size_t idx = 0;
    for (size_t i = 0; i < len; i += 3) {
        buffer[i]     = frame[idx].r;
        buffer[i + 1] = frame[idx].g;
        buffer[i + 2] = frame[idx].b;
        idx++;
    }
    return buffer;
}


float* fast_convert(std::vector<glm::vec3>& frame) {
    return glm::value_ptr(*frame.data());
}

}