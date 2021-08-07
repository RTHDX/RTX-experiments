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
    __ATTRIBS__ Ray(Point origin, Vector direction)
        : origin(std::move(origin))
        , direction(std::move(direction))
    {}

    __ATTRIBS__ Point at(float n) const {
        return origin + n * direction;
    }
};

struct Material {
    Color color;
    Albedo albedo;
    float specular_exponent;

public:
    __ATTRIBS__ inline Material() = default;
    __ATTRIBS__ inline Material(Color color, Albedo albedo, float specular_exponent)
        : color(std::move(color))
        , albedo(std::move(albedo))
        , specular_exponent(specular_exponent)
    {}
};


struct Light {
    Point position;
    float intensity;

public:
    __ATTRIBS__ inline Light(Point position, float intensity)
        : position(std::move(position))
        , intensity(intensity)
    {}

    __ATTRIBS__ Vector direction(const Point& point) const {
        return glm::normalize(point - position);
    }

    __ATTRIBS__ float diffuce_factor(const Point& point, const Vector& normal) const {
        float dot = glm::dot(-direction(point), normal);
        return intensity * utils::cuda::max(0.0f, dot);
    }

    __ATTRIBS__ float specular_factor(const Point& point, const Vector& normal, float exp) const {
        float dot = glm::dot(normal, -direction(point));
        return powf(utils::cuda::max(0.0f, dot), exp) * intensity;
    }
};


class Camera {
public:
    __ATTRIBS__ inline Camera() = default;
    __ATTRIBS__ Camera(Point position, float field_of_view, int width, int height);

    __ATTRIBS__ const Point& position() const;
    __ATTRIBS__ int width() const;
    __ATTRIBS__ int height() const;
    __ATTRIBS__ Ray emit_ray(int height_pos, int width_pos) const;
    __ATTRIBS__ Ray emit_world_ray(int height_pos, int width_pos) const;

    __ATTRIBS__ void move_forward() { _z_step -= _speed; }
    __ATTRIBS__ void move_backward() { _z_step += _speed; }
    __ATTRIBS__ void move_right() { _x_step += _speed; }
    __ATTRIBS__ void move_left() { _x_step -= _speed; }
    __ATTRIBS__ void move_up() { _y_step += _speed; }
    __ATTRIBS__ void move_down() { _y_step -= _speed; }

private:
    __ATTRIBS__ float pixel_ndc_x(int pos) const;
    __ATTRIBS__ float pixel_ndc_y(int pos) const;
    __ATTRIBS__ float pixel_screen_x(int x) const;
    __ATTRIBS__ float pixel_screen_y(int y) const;
    __ATTRIBS__ float x_axis_direction(int x) const;
    __ATTRIBS__ float y_axis_direction(int y) const;
    __ATTRIBS__ float z_axis_direction() const;
    __ATTRIBS__ glm::mat4x4 view() const;
    __ATTRIBS__ glm::mat4x4 projection() const;

private:
    Point _position;
    float _field_of_view, _aspect_ratio;
    int _width, _height;

    float _x_step = 0.0, _y_step = 0.0, _z_step = 0.0;
    float _speed = 0.1;
};


class Sphere;
class Hit {
    float    _t_near {utils::cuda::positive_infinite()};
    float    _t_far  {utils::cuda::positive_infinite()};
    Point    _point  {};
    Vector   _normal {};
    Sphere*  _object {nullptr};

public:
    __ATTRIBS__ Hit() = default;
    __ATTRIBS__ Hit(Sphere* object) : _object(object) {}
    __ATTRIBS__ bool is_hitted() const { return _object != nullptr; }

    __ATTRIBS__ float t_near() const { return _t_near; }
    __ATTRIBS__ void t_near(float val) { _t_near = val; }

    __ATTRIBS__ float t_far() const { return _t_far; }
    __ATTRIBS__ void t_far(float val) { _t_far = val; }

    __ATTRIBS__ const Point& point() const { return _point; }
    __ATTRIBS__ void point(const Point& point) { _point = point; }
    
    __ATTRIBS__ const Vector& normal() const { return _normal; }
    __ATTRIBS__ void normal(const Vector& val) { _normal = val; }
    
    __ATTRIBS__ Sphere* object() { return _object; }
};


class Sphere {
public:
    __ATTRIBS__ inline Sphere(Point center, float radius, Material material)
        : _material(std::move(material))
        , _center(std::move(center))
        , _radius(radius)
    {}

    __ATTRIBS__ Hit hit(const Ray& ray) {
        Vector origin_to_center = _center - ray.origin;
        float tca = glm::dot(origin_to_center, ray.direction);
        if (tca < 0) { return Hit(); } // ray direction mismatches

        float d = sqrt(glm::dot(origin_to_center, origin_to_center) - tca * tca);
        if (d > _radius) { return Hit(); } // ray misses sphere

        float thc = sqrt(_radius * _radius - d * d);
        Hit hit(this);
        hit.t_near(tca - thc);
        hit.t_far(tca + thc);
        hit.point(ray.at(hit.t_near()));
        hit.normal(glm::normalize(hit.point() - _center));
        return hit;
    }

    __ATTRIBS__ const Point& center() const { return _center; }
    __ATTRIBS__ float radius() const { return _radius; }
    __ATTRIBS__ const Material& material() const { return _material; }

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
    __ATTRIBS__ Scene() = default;
    __ATTRIBS__ Scene(const std::vector<Sphere>& spheres, Color background,
                      const std::vector<Light>& lights, size_t width, size_t height);

    __ATTRIBS__ const D_Spheres& objects() const { return _objects; }
    __ATTRIBS__ const Color& background() const { return _background; }
    __ATTRIBS__ const D_Lights& lights() const { return _lights; }
    __ATTRIBS__ size_t width() const { return _width; }
    __ATTRIBS__ size_t height() const { return _height; }
    __ATTRIBS__ float bias() const { return 1e-4; }
    __ATTRIBS__ float max_depth() const { return 7; }

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

    __ATTRIBS__ const Scene* scene() const { return _scene; }
    __ATTRIBS__ const Camera* camera() const { return _camera; }
    __ATTRIBS__ Camera* camera() { return _camera; }

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

//__global__ void kernel_render(const Context* ctx, size_t len, Color* frame);
//__ATTRIBS__ Hit intersects(const Context* ctx, const Ray& ray);
//__ATTRIBS__ Color trace(const Context* ctx, const Ray& ray, int depth);
//__ATTRIBS__ bool is_shaded(const Context* ctx, const Ray& ray,
//                       const Sphere* sphere);

}}
