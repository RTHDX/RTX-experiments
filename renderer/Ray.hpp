#pragma once

#include <iostream>

#include "Aliases.hpp"


namespace render {

struct Ray {
public:
    Point origin;
    Vector direction;

public:
    Ray(Point origin, Vector direction);

    Point at(float n) const;
};
std::ostream& operator << (std::ostream& out, const Ray& ray);

}