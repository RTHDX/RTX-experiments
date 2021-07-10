#pragma once

#include <vector>

#include "Material.hpp"
#include "Light.hpp"
#include "Hit.hpp"
#include "Ray.hpp"

namespace render {

class IObject {
public:
    IObject(Material material)
        : _material(std::move(material))
    {}
    virtual ~IObject() = default;

    const Material& material() const { return _material; }

    virtual Hit hit(const Ray& ray) = 0;

private:
    Material _material;
};
using Object = std::shared_ptr<IObject>;
using Objects = std::vector<Object>;


class Sphere : public IObject {
public:
    Sphere(Point center, float radius, Material material);

    Hit hit(const Ray& ray) override;

private:
    Point _center;
    float _radius;
};

struct Scene {
    static constexpr float BIAS = 1e-4;
    static constexpr int MAX_DEPTH = 5;

    Objects objects;
    Color background;
    Lights lights;
    size_t width;
    size_t height;
};

}