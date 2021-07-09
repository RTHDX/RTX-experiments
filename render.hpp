#pragma once

#include <vector>
#include <memory>
#include <array>
#include <limits>
#include <ostream>

#include <glm.hpp>


namespace render {

using Point  = glm::vec3;
using Color  = glm::vec3;
using Albedo = glm::vec3;
using Vector = glm::vec3;


struct Material {
    Color color;
    Albedo albedo;
    float specular_exponent;

public:
    Material() = default;
    Material(Color color, Albedo albedo, float specular_exponent);
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

class IObject;
struct Hit {
    float t_near;
    float t_far;
    bool  is_hitted;
    Point point;
    Vector normal;
    IObject* object {nullptr};

public:
    Hit(bool is_hitted);
};

std::ostream& operator << (std::ostream& os, const Hit&);

struct Ray {
public:
    Point origin;
    Vector direction;

public:
    Ray(Point origin, Vector direction);

    Point at(float n) const;
};
std::ostream& operator << (std::ostream& out, const Ray& ray);


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
    static constexpr float BIAS = 1e-4;
    static constexpr int MAX_DEPTH = 50;

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
    Color trace(const Ray& ray, int depth) const;
    Hit intersects(const Ray& ray) const;
    bool is_shaded(const Ray& ray, IObject* current) const;

private:
    Scene _scene;
    Camera _camera;
};

}