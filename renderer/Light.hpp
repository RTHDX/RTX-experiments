#pragma once

#include <vector>

#include "Aliases.hpp"


namespace render {

struct Light {
    Point position;
    float intensity;

public:
    Light(Point position, float intensity);

    Vector direction(const Point& point) const;
    float diffuce_factor(const Point& point, const Vector& normal) const;
    float specular_factor(const Point& point, const Vector& normal, float exp) const;
};
using Lights = std::vector<Light>;

}