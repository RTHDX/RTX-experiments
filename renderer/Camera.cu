#include "CudaRender.cuh"


namespace render { namespace cuda {

__ATTRIBS__ static inline glm::vec4 convert_to_vec4(const glm::vec3& input) {
    return glm::vec4(input.x, input.y, input.z, 1.0);
}

__ATTRIBS__ static inline glm::vec3 covert_to_vec3(const glm::vec4& input) {
    return glm::vec3(input.x, input.y, input.z);
}


Camera::Camera(Point position, float field_of_view, int width, int height)
    : _position(std::move(position))
    , _field_of_view(field_of_view)
    , _aspect_ratio(float(width) / float(height))
    , _width(width)
    , _height(height)
{}

__ATTRIBS__ Ray Camera::emit_ray(int height_pos, int width_pos) const {
    Vector direction(x_axis_direction(width_pos),
                     y_axis_direction(height_pos),
                     z_axis_direction());

    return Ray{_position, glm::normalize(direction)};
}

__ATTRIBS__ Ray Camera::emit_world_ray(int height_pos, int width_pos) const {
    Vector direction(x_axis_direction(width_pos),
                     y_axis_direction(height_pos),
                     z_axis_direction());

    glm::vec4 ray_origin_world = convert_to_vec4(_position) * projection();
    glm::vec4 ray_position_world = convert_to_vec4(direction) *
                                   projection();

    return Ray{covert_to_vec3(ray_origin_world),
               glm::normalize(covert_to_vec3(ray_position_world))};
}


__ATTRIBS__ const Point& Camera::position() const { return _position; }
__ATTRIBS__ int Camera::width() const { return _width; }
__ATTRIBS__ int Camera::height() const { return _height; }

__ATTRIBS__ float Camera::pixel_ndc_x(int pos) const {
    return (pos + 0.5f) / _width;
}

__ATTRIBS__ float Camera::pixel_ndc_y(int pos) const {
    return (pos + 0.5f) / _height;
}

__ATTRIBS__ float Camera::pixel_screen_x(int x) const {
    return 2.0f * pixel_ndc_x(x) - 1.0f;
}

__ATTRIBS__ float Camera::pixel_screen_y(int y) const {
    return 2.0f * pixel_ndc_y(y) - 1.0f;
}

__ATTRIBS__ float Camera::x_axis_direction(int x) const {
    return pixel_screen_x(x) * _aspect_ratio * tanf(_field_of_view / 2);
}

__ATTRIBS__ float Camera::y_axis_direction(int y) const {
    return (pixel_screen_y(y) * tanf(_field_of_view / 2));
}

__ATTRIBS__ float Camera::z_axis_direction() const {
    return -1.0f;
}

__ATTRIBS__ glm::mat4x4 Camera::view() const {
    return glm::mat4x4{
        {1.0, 0.0, 0.0, _x_step},
        {0.0, 1.0, 0.0, _y_step},
        {0.0, 0.0, 1.0, _z_step},
        {0.0, 0.0, 0.0, 1.0}
    };
}

__ATTRIBS__ glm::mat4x4 Camera::projection() const {
    const glm::mat4x4 one{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };

    return one * view();
}

}}
