#pragma once

#include <vector>
#include <memory>
#include <array>
#include <limits>

#include "glm/glm/glm.hpp"


namespace render {

using Point = glm::vec3;
using Color = glm::vec3;
using Vector = glm::vec3;


struct Hit {
    float t_near;
    float t_far;
    bool  is_hitted;
    Point point;
    Vector normal;

public:
    Hit(bool is_hitted)
        : is_hitted(is_hitted)
        , t_near(std::numeric_limits<float>::max())
        , t_far(std::numeric_limits<float>::max())
    {}
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
    IOBject(Color color) : _color(std::move(color)) {}
    virtual ~IOBject() = default;

    const Color& color() const { return _color; }

    virtual Hit hit(const Ray& ray) const = 0;

private:
    Color _color;
};
using Objects = std::vector<std::shared_ptr<IOBject>>;


class Sphere : public IOBject {
public:
    Sphere(Point center, float radius, Color color);

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

private:
    Scene _scene;
    Camera _camera;
};

}