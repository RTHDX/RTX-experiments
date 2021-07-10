#pragma once

#include <glm.hpp>

#include "Aliases.hpp"


namespace render {

struct Material {
    Color color;
    Albedo albedo;
    float specular_exponent;

public:
    Material() = default;
    Material(Color color, Albedo albedo, float specular_exponent);
};

}
