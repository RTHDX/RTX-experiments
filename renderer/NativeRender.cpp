#include <glad/glad.h>
#include <gtx/string_cast.hpp>

#include "Utils.hpp"
#include "NativeRender.hpp"

namespace render {

Camera::Camera(Point position, float field_of_view, int width, int height)
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
    return Ray {_position, glm::normalize(direction)};
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


Hit::Hit(IObject* object) : object(object) {}

bool Hit::is_hitted() const {
    return object != nullptr;
}

std::ostream& operator << (std::ostream& os, const Hit& hit) {
    return os << "<Hit.\n t_near: " << hit.t_near
              << "\n t_far: " << hit.t_far
              << "\n is hitted: " << hit.is_hitted()
              << "\n point: " << glm::to_string(hit.point)
              << "\n normal: " << glm::to_string(hit.normal)
              << "\n object: " << hit.object << ">";
}


Ray::Ray(Point point, Vector direction)
    : origin(std::move(point))
    , direction(std::move(direction))
{}

Point Ray::at(float n) const {
    return origin + n * direction;
}

std::ostream& operator << (std::ostream& out, const Ray& ray) {
    return out << "<Ray. origin: " << glm::to_string(ray.origin)
        << ", direction: " << glm::to_string(ray.direction) << ">";
}


Light::Light(Point position, float intensity)
    : position(std::move(position))
    , intensity(intensity)
{}

Vector Light::direction(const Point& point) const {
    return glm::normalize(point - position);
}

float Light::diffuce_factor(const Point& point, const Vector& normal) const {
    float dot = glm::dot(-direction(point), normal);
    return intensity * std::max(0.0f, dot);
}

float Light::specular_factor(const Point& point, const Vector& normal, float exp) const {
    float dot = glm::dot(normal, -direction(point));
    return powf(std::max(0.0f, dot), exp) * intensity;
}

Material::Material(Color color, Albedo albedo, float specular_exponent)
    : color(std::move(color))
    , albedo(std::move(albedo))
    , specular_exponent(specular_exponent)
{}


Sphere::Sphere(Point center, float radius, Material material)
    : IObject(std::move(material))
    , _center(std::move(center))
    , _radius(radius)
{}

Hit Sphere::hit(const Ray& ray) {
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


static inline Vector reflect(const Vector& income, const Vector& normal) {
    return income - normal * 2.f * glm::dot(income, normal) * normal;
}


NativeRender::NativeRender(Scene scene, Camera camera)
    : BaseRender()
    , _scene(std::move(scene))
    , _camera(std::move(camera))
    , _frame(_scene.width * _scene.height)
{}

void NativeRender::render() {
    const size_t width = _scene.width;
    const size_t height = _scene.height;

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            const size_t index = i * width + j;
            update_pixel(index, trace(_camera.emit_ray(i, j), 0));
        }
    }
}

float* NativeRender::frame() {
    return utils::fast_convert(_frame);
}

void NativeRender::draw() {
    BaseRender::draw();
    glDrawPixels(_scene.width, _scene.height, GL_RGB, GL_FLOAT,
                 frame());
}

Color NativeRender::trace(const Ray& ray, int depth) {
    if (depth == _scene.MAX_DEPTH) { return _scene.background; }

    Hit hit = intersects(ray);
    if (!hit.is_hitted()) { return _scene.background; }

    Ray reflect_ray(hit.point + hit.normal * _scene.BIAS,
                    reflect(ray.direction, hit.normal));
    Color reflect_color = trace(reflect_ray, depth + 1);

    float diffuse_light_intensity = 0.0f,
          specular_light_intensity = 0.0f;
    for (const Light& light : _scene.lights) {
        const auto& point = hit.point;
        const auto& normal = hit.normal;
        float exponent = hit.object->material().specular_exponent;
        Ray shadow_ray(point + normal * _scene.BIAS,
                       -light.direction(point));
        if (is_shaded(shadow_ray, hit.object)) { continue; }
        diffuse_light_intensity += light.diffuce_factor(point, normal);
        specular_light_intensity += light.specular_factor(point, normal,
                                                          exponent);
    }

    const Albedo& albedo = hit.object->material().albedo;
    Color diffuse = hit.object->material().color * diffuse_light_intensity
                    * albedo.x;
    Color specular = Color(1.0f, 1.0f, 1.0f) * specular_light_intensity
                     * albedo.y;
    Color reflection_component = albedo.z * reflect_color;
    Color total = diffuse + specular + reflection_component;
    return total;
}

Hit NativeRender::intersects(const Ray& ray) {
    Hit hit(false);
    float distance = std::numeric_limits<float>::max();
    for (const auto& object : _scene.objects) {
        Hit temp = object->hit(ray);
        if (temp.is_hitted() && distance > temp.t_near) {
            hit = temp;
            distance = temp.t_near;
        }
    }
    return hit;
}

bool NativeRender::is_shaded(const Ray& shadow_ray, IObject* current) {
    for (const Object& object : _scene.objects) {
        if (object.get() == current) continue;
        Hit hit = object->hit(shadow_ray);
        if (hit.is_hitted()) { return true; }
    }
    return false;
}

void NativeRender::update_pixel(const size_t index, Color&& color) {
    assert(index > 0 || index < _frame.size());
    _frame[index] = color;
}

}