#include <glad/glad.h>

#include "BaseRender.hpp"

namespace render {

void BaseRender::draw() {
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

}