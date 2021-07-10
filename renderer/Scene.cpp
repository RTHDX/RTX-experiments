#include "Scene.hpp"


namespace render {

Sphere::Sphere(Point center, float radius, Material material)
    : IObject(std::move(material))
    , _center(std::move(center))
    , _radius(radius)
{}

Hit Sphere::hit(const Ray& ray) {
    Vector origin_to_center = _center - ray.origin;
    float tca = glm::dot(origin_to_center, ray.direction);
    if (tca < 0) { return Hit(); } // ray direction mismatches

    float d = sqrt(glm::dot(origin_to_center, origin_to_center) - tca * tca);
    if (d > _radius) { return Hit(); } // ray misses sphere

    float thc = sqrt(_radius * _radius - d * d);
    Hit hit(this);
    hit.t_near = tca - thc;
    hit.t_far = tca + thc;
    hit.point = ray.at(hit.t_near);
    hit.normal = glm::normalize(hit.point - _center);
    return hit;
}

}