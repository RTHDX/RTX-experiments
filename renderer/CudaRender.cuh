#pragma once

#include <stdio.h>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CudaUtils.cuh"
#include "BaseRender.hpp"


namespace render { namespace cuda {

struct Ray {
    Point origin;
    Vector direction;

public:
    __device__ Ray(Point origin, Vector direction)
        : origin(std::move(origin))
        , direction(std::move(direction))
    {}

    __device__ Point at(float n) const {
        return origin + n * direction;
    }
};

struct Material {
    Color color;
    Albedo albedo;
    float specular_exponent;

public:
    inline Material() = default;
    inline Material(Color color, Albedo albedo, float specular_exponent)
        : color(std::move(color))
        , albedo(std::move(albedo))
        , specular_exponent(specular_exponent)
    {}
};


struct Light {
    Point position;
    float intensity;

public:
    inline Light(Point position, float intensity)
        : position(std::move(position))
        , intensity(intensity)
    {}

    __device__ Vector direction(const Point& point) const {
        return glm::normalize(point - position);
    }

    __device__ float diffuce_factor(const Point& point, const Vector& normal) const {
        float dot = glm::dot(-direction(point), normal);
        return intensity * utils::cuda::max(0.0f, dot);
    }

    __device__ float specular_factor(const Point& point, const Vector& normal, float exp) const {
        float dot = glm::dot(normal, -direction(point));
        return powf(utils::cuda::max(0.0f, dot), exp) * intensity;
    }
};
using Lights = std::vector<Light>;


class Camera {
public:
    inline Camera() = default;
    inline Camera(Point position, float field_of_view, int width, int height)
        : _position(std::move(position))
        , _field_of_view(field_of_view)
        , _aspect_ratio(float(width) / float(height))
        , _width(width)
        , _height(height)
    {}

    __device__ const Point& position() const { return _position; }
    __device__ int width() const { return _width; }
    __device__ int height() const { return _height; }

    __device__ Ray emit_ray(int height_pos, int width_pos) const {
        Vector direction(x_axis_direction(width_pos),
                         y_axis_direction(height_pos),
                         z_axis_direction());
        return Ray{ _position, glm::normalize(direction) };
    }

private:
    __device__ float pixel_ndc_x(int pos) const {
        return (pos + 0.5f) / _width;
    }

    __device__ float pixel_ndc_y(int pos) const {
        return (pos + 0.5f) / _height;
    }

    __device__ float pixel_screen_x(int x) const {
        return 2.0f * pixel_ndc_x(x) - 1.0f;
    }

    __device__ float pixel_screen_y(int y) const {
        return 2.0f * pixel_ndc_y(y) - 1.0f;
    }

    __device__ float x_axis_direction(int x) const {
        return pixel_screen_x(x) * _aspect_ratio * tanf(_field_of_view / 2);
    }

    __device__ float y_axis_direction(int y) const {
        return (pixel_screen_y(y) * tanf(_field_of_view / 2));
    }

    __device__ float z_axis_direction() const {
        return -1.0f;
    }

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
    __device__ Hit() = default;
    __device__ Hit(IObject* object);
    __device__ bool is_hitted() const { return object != nullptr; }
};


class IObject {
public:
    inline IObject(Material material)
        : _material(std::move(material))
    {}
    virtual ~IObject() = default;

    __device__ const Material& material() const { return _material; }
    __device__ virtual Hit hit(const Ray& ray) = 0;

private:
    Material _material;
};
using Object = std::shared_ptr<IObject>;
using Objects = std::vector<Object>;


class Sphere : public IObject {
public:
    inline Sphere(Point center, float radius, Material material)
        : IObject(std::move(material))
        , _center(std::move(center))
        , _radius(radius)
    {}

    __device__ Hit hit(const Ray& ray) override {
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

private:
    Point _center;
    float _radius;
};

class Scene {
    static constexpr float BIAS    = 1e-4;
    static constexpr int MAX_DEPTH = 5;

public:
    Scene() = default;
    Scene(Objects objects, Color background, Lights lights,
          size_t width, size_t height);

    __device__ const Objects& objects() const { return _objects; }
    __device__ const Color& background() const { return _background; }
    __device__ const Lights& lights() const { return _lights; }
    __device__ size_t width() const { return _width; }
    __device__ size_t height() const { return _height; }

private:
    Objects _objects;
    Color _background;
    Lights _lights;
    size_t _width;
    size_t _height;
};

class CudaRender final : public BaseRender {
public:
    CudaRender(Scene scene, Camera camera, int width, int height);
    ~CudaRender() override = default;

    void render() override;
    float* frame() override { return glm::value_ptr(*_frame); }
    void draw() override {}

    __device__ const Scene& scene() const { return _scene; }
    __device__ const Camera& camera() const { return _camera; }

private:
    Scene _scene;
    Camera _camera;
    size_t _len;
    Color* _frame;
    Color* _cuda_frame_ptr;
    CudaRender* _dev_ptr;
};

}}
