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
    ATTRIBS Ray(Point origin, Vector direction)
        : origin(std::move(origin))
        , direction(std::move(direction))
    {}

    ATTRIBS Point at(float n) const {
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

    ATTRIBS Vector direction(const Point& point) const {
        return glm::normalize(point - position);
    }

    ATTRIBS float diffuce_factor(const Point& point, const Vector& normal) const {
        float dot = glm::dot(-direction(point), normal);
        return intensity * utils::cuda::max(0.0f, dot);
    }

    ATTRIBS float specular_factor(const Point& point, const Vector& normal, float exp) const {
        float dot = glm::dot(normal, -direction(point));
        return powf(utils::cuda::max(0.0f, dot), exp) * intensity;
    }
};


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

    ATTRIBS const Point& position() const { return _position; }
    ATTRIBS int width() const { return _width; }
    ATTRIBS int height() const { return _height; }

    ATTRIBS Ray emit_ray(int height_pos, int width_pos) const {
        Vector direction(x_axis_direction(width_pos),
                         y_axis_direction(height_pos),
                         z_axis_direction());
        return Ray{ _position, glm::normalize(direction) };
    }

private:
    ATTRIBS float pixel_ndc_x(int pos) const {
        return (pos + 0.5f) / _width;
    }

    ATTRIBS float pixel_ndc_y(int pos) const {
        return (pos + 0.5f) / _height;
    }

    ATTRIBS float pixel_screen_x(int x) const {
        return 2.0f * pixel_ndc_x(x) - 1.0f;
    }

    ATTRIBS float pixel_screen_y(int y) const {
        return 2.0f * pixel_ndc_y(y) - 1.0f;
    }

    ATTRIBS float x_axis_direction(int x) const {
        return pixel_screen_x(x) * _aspect_ratio * tanf(_field_of_view / 2);
    }

    ATTRIBS float y_axis_direction(int y) const {
        return (pixel_screen_y(y) * tanf(_field_of_view / 2));
    }

    ATTRIBS float z_axis_direction() const {
        return -1.0f;
    }

private:
    Point _position;
    float _field_of_view, _aspect_ratio;
    int _width, _height;
};


class Sphere;
struct Hit {
    float    t_near {utils::cuda::positive_infinite()};
    float    t_far  {utils::cuda::positive_infinite()};
    Point    point  {Point(1.0f, 1.0f, 1.0f)};
    Vector   normal {Point(1.0f, 1.0f, 1.0f)};
    Sphere* object {nullptr};

public:
    ATTRIBS Hit() = default;
    ATTRIBS Hit(Sphere* object) : object(object) {}
    ATTRIBS bool is_hitted() const { return object != nullptr; }
};


class Sphere {
public:
    inline Sphere(Point center, float radius, Material material)
        : _material(std::move(material))
        , _center(std::move(center))
        , _radius(radius)
    {}

    ATTRIBS Hit hit(const Ray& ray) {
        Vector origin_to_center = _center - ray.origin;
        float tca = glm::dot(origin_to_center, ray.direction);
        if (tca < 0) { return Hit(); } // ray direction mismatches

        float d = sqrt(glm::dot(origin_to_center, origin_to_center) - tca * tca);
        if (d > _radius) { return Hit(); } // ray misses sphere

        float thc = sqrt(_radius * _radius - d * d);
        Hit hit(this);
        hit.t_near = (tca - thc);
        hit.t_far = (tca + thc);
        hit.point = (ray.at(hit.t_near));
        hit.normal = (glm::normalize(hit.point - _center));
        return hit;
    }

    ATTRIBS const Point& center() const { return _center; }
    ATTRIBS float radius() const { return _radius; }
    ATTRIBS const Material& material() const { return _material; }

private:
    Material _material;
    Point _center;
    float _radius;
};


class Scene {
    using D_Spheres = utils::cuda::Collection<Sphere>;
    using D_Lights = utils::cuda::Collection<Light>;

    static constexpr float BIAS    = 1e-4;
    static constexpr int MAX_DEPTH = 5;

public:
    Scene() = default;
    Scene(const std::vector<Sphere>& spheres, Color background,
          const std::vector<Light>& lights, size_t width, size_t height);

    __host__ __device__ const D_Spheres& objects() const { return _objects; }
    __host__ __device__ const Color& background() const { return _background; }
    __host__ __device__ const D_Lights& lights() const { return _lights; }
    __host__ __device__ size_t width() const { return _width; }
    __host__ __device__ size_t height() const { return _height; }
    __host__ __device__ float bias() const { return 1e-4; }
    __host__ __device__ float max_depth() const { return 5; }

public:
    utils::cuda::Collection<Sphere> _objects;
    Color _background;
    utils::cuda::Collection<Light> _lights;
    size_t _width;
    size_t _height;
};

class CudaRender final : public BaseRender {
public:
    CudaRender(const Scene& scene, const Camera& camera, int width, int height);
    ~CudaRender() override;

    void render() override;
    float* frame() override;
    void draw() override;

    ATTRIBS const Scene* scene() const { return _scene; }
    ATTRIBS const Camera* camera() const { return _camera; }

private:
    Scene* _scene;
    Camera* _camera;
    size_t _width, _height, _len;
    Color* _frame;
    Color* _cuda_frame_ptr;
};

struct Context {
    Scene* scene   {nullptr};
    Camera* camera {nullptr};
};

__global__ void kernel_render(const Context* ctx, size_t len, Color* frame);
ATTRIBS Hit intersects(const Context* ctx, const Ray& ray);
ATTRIBS Color trace(const Context* ctx, const Ray& ray);
ATTRIBS bool is_shaded(const Context* ctx, const Ray& ray,
                       const Sphere* sphere);

}}
