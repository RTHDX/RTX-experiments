#pragma once

#include "Aliases.hpp"
#include "Material.hpp"
#include "Light.hpp"
#include "Hit.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Camera.hpp"


namespace render {

class IRender {
public:
    IRender(Scene scene, Camera camera);
    virtual ~IRender() = default;

    virtual std::vector<Color> render() const = 0;
    virtual Color trace(const Ray& ray, int depth) const = 0;
    virtual Hit intersects(const Ray& ray) const = 0;
    virtual bool is_shaded(const Ray& ray, IObject* current) const = 0;

    const Scene& scene() const { return _scene; }
    const Camera& camera() const { return _camera; }

private:
    Scene _scene;
    Camera _camera;
};


class NativeRender final : public IRender {
public:
    NativeRender(Scene scene, Camera camera);
    ~NativeRender() override = default;

    std::vector<Color> render() const override;
    Color trace(const Ray& ray, int depth) const override;
    Hit intersects(const Ray& ray) const override;
    bool is_shaded(const Ray& ray, IObject* current) const override;
};

}
