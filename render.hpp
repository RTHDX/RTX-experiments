#pragma once

#include <vector>
#include <memory>
#include <array>
#include <limits>

#include "glm/glm/glm.hpp"


namespace render {

using Point  = glm::vec3;
using Color  = glm::vec3;
using Albedo = glm::vec3;
using Vector = glm::vec3;


struct Material {
    Color color;
    Albedo albedo;

public:
    Material() = default;
    Material(Color color, Albedo albedo);
};

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


struct Hit {
    float t_near;
    float t_far;
    bool  is_hitted;
    Point point;
    Vector normal;
    Material material;

public:
    Hit(bool is_hitted);
};

struct Ray {
public:
    Point origin;
    Vector direction;

public:
    Point at(float n) const;
};


class IOBject {
public:
    IOBject(Material material)
        : _material(std::move(material))
    {}
    virtual ~IOBject() = default;

    const Material& material() const { return _material; }

    virtual Hit hit(const Ray& ray) const = 0;

private:
    Material _material;
};
using Objects = std::vector<std::shared_ptr<IOBject>>;


class Sphere : public IOBject {
public:
    Sphere(Point center, float radius, Material material);

    Hit hit(const Ray& ray) const override;

private:
    Point _center;
    float _radius;
};

class Camera {
public:
    // Field of view in radians
    Camera(Point position, float field_of_view, int width, int height);

    const Point& position() const { return _position; }
    int width() const { return _width; }
    int height() const { return _height; }

    Ray emit_ray(int height_pos, int width_pos) const;

private:
    float pixel_ndc_x(int pos) const;
    float pixel_ndc_y(int pos) const;

    float pixel_screen_x(int x) const;
    float pixel_screen_y(int y) const;

    float x_axis_direction(int x) const;
    float y_axis_direction(int y) const;
    float z_axis_direction() const;

private:
    Point _position;
    float _field_of_view, _aspect_ratio;
    int _width, _height;
};

struct Scene {
    Objects objects;
    Color background;
    Lights lights;
    size_t width;
    size_t height;
};

class Render {
public:
    Render(Scene scene, Camera camera);
    ~Render() = default;

    std::vector<Color> render() const;

private:
    Color trace(const Ray& ray) const;
    Hit intersects(const Ray& ray) const;

private:
    Scene _scene;
    Camera _camera;
};

}