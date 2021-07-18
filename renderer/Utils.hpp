#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <vector>

#include <gtc/type_ptr.hpp>
#include <glm.hpp>

#include "glad/glad.h"
#include "GLFW/glfw3.h"


namespace utils {

void fill_frame(float* buffer, size_t size, float r, float g, float b);
GLFWwindow* load_glfw(const char* title, int width, int height);
void load_opengl();
void post_processing();
void draw(int width, int height, float* frame);
float to_radian(float degree);
float* covert(const std::vector<glm::vec3>& frame);
float* fast_convert(std::vector<glm::vec3>& frame);

}
