#pragma once

#include <vector>

#include <glm.hpp>

namespace render {

using Point  = glm::vec3;
using Color  = glm::vec3;
using Albedo = glm::vec3;
using Vector = glm::vec3;

class BaseRender {
public:
    BaseRender() = default;
    virtual ~BaseRender() = default;

    virtual void render() = 0;
    virtual float* frame() = 0;
    virtual void draw();
};

}
