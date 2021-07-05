#include <algorithm>

#include "render.hpp"


namespace render {

Point Ray::at(float n) const {
    return origin + n * direction;
}

Sphere::Sphere(Point center, float radius, Color color)
    : IOBject(std::move(color))
    , _center(std::move(center))
    , _radius(radius)
{}

Hit Sphere::hit(const Ray& ray) const {
    Vector origin_to_center = _center - ray.origin;
    float tca = glm::dot(origin_to_center, ray.direction);
    if (tca < 0) { return Hit(false); } // ray direction mismatches

    float d = sqrt(glm::dot(origin_to_center, origin_to_center) - tca * tca);
    if (d > _radius) { return Hit(false); } // ray misses sphere

    float thc = sqrt(_radius * _radius - d * d);
    Hit hit(true);
    hit.t_near = tca - thc;
    hit.t_far = tca + thc;
    hit.point = ray.at(hit.t_near);
    hit.normal = glm::normalize(hit.point - _center);
    return hit;
}

Camera::Camera(glm::vec3 position, float field_of_view, int width, int height)
    : _position(std::move(position))
    , _field_of_view(field_of_view)
    , _aspect_ratio(float(width) / float(height))
    , _width(width)
    , _height(height)
{}

Ray Camera::emit_ray(int height_pos, int width_pos) const {
    Vector direction(x_axis_direction(width_pos),
                     y_axis_direction(height_pos),
                     z_axis_direction());
    Ray ray;
    ray.origin = _position;
    ray.direction = glm::normalize(direction);
    return ray;
}

float Camera::pixel_ndc_x(int pos) const { return (pos + 0.5f) / _width; }
float Camera::pixel_ndc_y(int pos) const { return (pos + 0.5f) / _height; }

float Camera::pixel_screen_x(int x) const {
    return 2.0f * pixel_ndc_x(x) - 1.0f;
}

float Camera::pixel_screen_y(int y) const {
    return 2.0f * pixel_ndc_y(y) - 1.0f;
}

float Camera::x_axis_direction(int x) const {
    return pixel_screen_x(x) * _aspect_ratio * tanf(_field_of_view / 2);
}

float Camera::y_axis_direction(int y) const {
    return (pixel_screen_y(y) * tanf(_field_of_view / 2));
}

float Camera::z_axis_direction() const { return -1.0f; }

Render::Render(Scene scene, Camera camera)
    : _scene(std::move(scene))
    , _camera(std::move(camera))
{}

std::vector<Color> Render::render() const {
    std::vector<Color> frame(_scene.width * _scene.height);
    for (size_t i = 0; i < _scene.height; ++i) {
        for (size_t j = 0; j < _scene.width; ++j) {
            frame[i * _scene.width + j] = trace(_camera.emit_ray(i, j));
        }
    }
    return frame;
}

Color Render::trace(const Ray& ray) const {
    for (const auto& object : _scene.objects) {
        Hit hit = object->hit(ray);
        if (hit.is_hitted) {
            return object->color();
        }
    }
    return _scene.background;
}

}
