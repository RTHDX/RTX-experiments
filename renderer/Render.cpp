#include <algorithm>
#include <gtx/string_cast.hpp>

#include "utils.hpp"

#include "render.hpp"


namespace render {

static inline Vector reflect(const Vector& income, const Vector& normal) {
    return income - normal * 2.f * glm::dot(income, normal) * normal;
}


IRender::IRender(Scene scene, Camera camera)
    : _scene(std::move(scene))
    , _camera(std::move(camera))
{}


NativeRender::NativeRender(Scene scene, Camera camera)
    : IRender(std::move(scene), std::move(camera))
{}

std::vector<Color> NativeRender::render() const {
    const size_t width = scene().width;
    const size_t height = scene().height;

    std::vector<Color> frame(width * height);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            const size_t index = i * width + j;
            frame[index] = trace(camera().emit_ray(i, j), 0);
        }
    }
    return frame;
}

Color NativeRender::trace(const Ray& ray, int depth) const {
    if (depth == scene().MAX_DEPTH) { return scene().background; }

    Hit hit = intersects(ray);
    if (!hit.is_hitted()) { return scene().background; }

    Ray reflect_ray(hit.point + hit.normal * scene().BIAS,
                    reflect(ray.direction, hit.normal));
    Color reflect_color = trace(reflect_ray, depth + 1);

    float diffuse_light_intensity = 0.0f,
          specular_light_intensity = 0.0f;
    for (const Light& light : scene().lights) {
        const auto& point = hit.point;
        const auto& normal = hit.normal;
        float exponent = hit.object->material().specular_exponent;
        Ray shadow_ray(point + normal * scene().BIAS,
                       -light.direction(point));
        if (is_shaded(shadow_ray, hit.object)) { continue; }
        diffuse_light_intensity += light.diffuce_factor(point, normal);
        specular_light_intensity += light.specular_factor(point, normal,
                                                          exponent);
    }

    const Albedo& albedo = hit.object->material().albedo;
    Color diffuse = hit.object->material().color * diffuse_light_intensity
                    * albedo.x;
    Color specular = utils::WHITE * specular_light_intensity
                     * albedo.y;
    Color reflection_component = albedo.z * reflect_color;
    Color total = diffuse + specular + reflection_component;
    return total;
}

Hit NativeRender::intersects(const Ray& ray) const {
    Hit hit(false);
    float distance = std::numeric_limits<float>::max();
    for (const auto& object : scene().objects) {
        Hit temp = object->hit(ray);
        if (temp.is_hitted() && distance > temp.t_near) {
            hit = temp;
            distance = temp.t_near;
        }
    }
    return hit;
}

bool NativeRender::is_shaded(const Ray& shadow_ray, IObject* current) const {
    for (const Object& object : scene().objects) {
        if (object.get() == current) continue;
        Hit hit = object->hit(shadow_ray);
        if (hit.is_hitted()) { return true; }
    }
    return false;
}

}
