#include "Light.hpp"


namespace render {

Light::Light(Point position, float intensity)
    : position(std::move(position))
    , intensity(intensity)
{}

Vector Light::direction(const Point& point) const {
    return glm::normalize(point - position);
}

float Light::diffuce_factor(const Point& point, const Vector& normal) const {
    float dot = glm::dot(-direction(point), normal);
    return intensity * std::max(0.0f, dot);
}

float Light::specular_factor(const Point& point, const Vector& normal, float exp) const {
    float dot = glm::dot(normal, -direction(point));
    return powf(std::max(0.0f, dot), exp) * intensity;
}

}