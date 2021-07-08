#include <algorithm>

#include "render.hpp"


namespace render {

static inline Vector reflect(const Vector& income, const Vector& normal) {
    return income - normal * 2.f * (income * normal);
}

Material::Material(Color color, Albedo albedo, float specular_exponent)
    : color(std::move(color))
    , albedo(std::move(albedo))
    , specular_exponent(specular_exponent)
{}

Light::Light(Point position, float intensity)
    : position(std::move(position))
    , intensity(intensity)
{}

Vector Light::direction(const Point& point) const {
    return glm::normalize(position - point);
}

float Light::diffuce_factor(const Point& point, const Vector& normal) const {
    float dot = glm::dot(direction(point), normal);
    return intensity * std::max(0.0f, dot);
}

float Light::specular_factor(const Point& point, const Vector& normal, float exp) const {
    //Vector ref = -reflect(-direction(point), normal);
    float dot = glm::dot(normal, direction(point));
    return powf(std::max(0.0f, dot), exp) * intensity;
}

Hit::Hit(bool is_hitted)
    : is_hitted(is_hitted)
    , t_near(std::numeric_limits<float>::max())
    , t_far(std::numeric_limits<float>::max())
    , point(1.0f, 1.0f, 1.0f)
    , normal(1.0f, 1.0f, 1.0f)
    , material()
{}

Point Ray::at(float n) const {
    return origin + n * direction;
}

Sphere::Sphere(Point center, float radius, Material material)
    : IObject(std::move(material))
    , _center(std::move(center))
    , _radius(radius)
{}

Hit Sphere::hit(const Ray& ray) {
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
    hit.material = material();
    hit.object = this;
    return hit;
}

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
    Hit hit = intersects(ray);
    if (!hit.is_hitted) { return _scene.background; }

    float diffuse_light_intensity = 0.0f,
          specular_light_intensity = 0.0f;
    for (const Light& light : _scene.lights) {
        const auto& point = hit.point;
        const auto& normal = hit.normal;
        float exponent = hit.material.specular_exponent;
        Ray shadow_ray;
        shadow_ray.origin = light.position;
        shadow_ray.direction = light.direction(point);
        if (is_shaded(shadow_ray, hit.object)) {
            return Color(0.0, 0.0, 0.0);
        }
        diffuse_light_intensity += light.diffuce_factor(point, normal);
        specular_light_intensity += light.specular_factor(point, normal, exponent);
    }

    const Albedo& albedo = hit.material.albedo;
    Color diffuse = hit.material.color * diffuse_light_intensity * albedo.x;
    Color specular = Color(1.0f, 1.0f, 1.0f) * specular_light_intensity * albedo.y;
    Color total = diffuse + specular;
    return total;
}

Hit Render::intersects(const Ray& ray) const {
    Hit hit(false);
    for (const auto& object : _scene.objects) {
        hit = object->hit(ray);
        if (hit.is_hitted) { return hit; }
    }
    return hit;
}

bool Render::is_shaded(const Ray& shadow_ray, IObject* current) const {
    for (const Object& object : _scene.objects) {
        if (object.get() == current) continue;
        Hit hit = object->hit(shadow_ray);
        if (hit.is_hitted) {
            return true;
        }
    }
    return false;
}

}
