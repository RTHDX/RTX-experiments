#include <gtx/string_cast.hpp>

#include "Ray.hpp"

namespace render {

Ray::Ray(Point point, Vector direction)
    : origin(std::move(point))
    , direction(std::move(direction))
{}

Point Ray::at(float n) const {
    return origin + n * direction;
}

std::ostream& operator << (std::ostream& out, const Ray& ray) {
    return out << "<Ray. origin: " << glm::to_string(ray.origin)
        << ", direction: " << glm::to_string(ray.direction) << ">";
}


}