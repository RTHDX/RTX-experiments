#pragma once

#include <ostream>
#include <vector>
#include <glm.hpp>

#include "BaseRender.hpp"


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

class Camera {
public:
    Camera() = default;
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


class IObject;
struct Hit {
    float t_near    {std::numeric_limits<float>::max()};
    float t_far     {std::numeric_limits<float>::max()};
    Point point     {Point(1.0f, 1.0f, 1.0f)};
    Vector normal   {Point(1.0f, 1.0f, 1.0f)};
    IObject* object {nullptr};

public:
    Hit() = default;
    Hit(IObject* object);

    bool is_hitted() const;
};
std::ostream& operator << (std::ostream& os, const Hit&);


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


struct Material {
    Color color;
    Albedo albedo;
    float specular_exponent;

public:
    Material() = default;
    Material(Color color, Albedo albedo, float specular_exponent);
};


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


class NativeRender final : public BaseRender {
public:
    NativeRender(Scene scene, Camera camera);
    ~NativeRender() override = default;

    void render() override;
    float* frame() override;
    void draw() override;

private:
    Color trace(const Ray& ray, int depth);
    Hit intersects(const Ray& ray);
    bool is_shaded(const Ray& ray, IObject* current);
    void update_pixel(const size_t index, Color&& color);

private:
    Camera _camera;
    Scene _scene;
    std::vector<Color> _frame;
};

}