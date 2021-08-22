#include <gtc/matrix_access.hpp>

#include "CudaRender.cuh"


namespace render { namespace cuda {

__ATTRIBS__ static inline glm::vec4 convert_to_vec4(const glm::vec3& input) {
    return glm::vec4(input.x, input.y, input.z, 1.0);
}

__ATTRIBS__ static inline glm::vec3 covert_to_vec3(const glm::vec4& input) {
    return glm::vec3(input.x, input.y, input.z);
}


Camera::Camera(Point position, Point look_at, float field_of_view, int width, int height)
    : _position(std::move(position))
    , _look_at(look_at)
    , _field_of_view(field_of_view)
    , _aspect_ratio(float(width) / float(height))
    , _width(width)
    , _height(height)
{}

__ATTRIBS__ Ray Camera::emit_ray(int height_pos, int width_pos) const {
    Vector direction = Vector(width_pos - _width / 2.0,
                             (_height / 2.0 - height_pos),
                             (_height / 2.0) / tanf(_field_of_view / 2.0)) *
                       cam_to_world();

    return Ray(_position, glm::normalize(direction));
}

__ATTRIBS__ void Camera::move_forward() {
    _position.z = _position.z - _speed;
}

__ATTRIBS__ void Camera::move_backward() {
    _position.z = _position.z + _speed;
}

__ATTRIBS__ void Camera::move_right() {
    _position.x = _position.x + _speed;
}

__ATTRIBS__ void Camera::move_left() {
    _position.x = _position.x - _speed;
}

__ATTRIBS__ void Camera::move_up() {
    _position.y = _position.y + _speed;
}

__ATTRIBS__ void Camera::move_down() {
    _position.y = _position.y - _speed;
}

__ATTRIBS__ void Camera::update_position(const Point& point) {
    _position = point;
}

__ATTRIBS__ const Point& Camera::position() const { return _position; }
__ATTRIBS__ int Camera::width() const { return _width; }
__ATTRIBS__ int Camera::height() const { return _height; }

__ATTRIBS__ glm::mat3x3 Camera::cam_to_world() const {
    const Vector _up(0.0, 1.0, 0.0);
    const float EPSILON = 0.0000001;

    Vector forward = glm::normalize(_look_at - _position);
    Vector right = (std::fabs(std::fabs(glm::dot(_up, forward)) - 1.0f) > EPSILON) ?
                   -glm::cross(glm::normalize(_up), forward) :
                   Vector(1.0, 0.0, 0.0);
    Vector up = glm::normalize(glm::cross(forward, right));

    return glm::mat3x3 {
        {right.x, up.x, forward.x},
        {right.y, up.y, forward.y},
        {right.z, up.z, forward.z},
    };
}

__ATTRIBS__ void Camera::dump() const {
    printf("<Camera: location - (%.4f;%.4f;%.4f), WxH - %dx%d, "
                    "aspect ratio - %.3f, field of view - %.3f, "
                    "look at - (%.4f;%.4f%.4f)>\n",
            _position.x, _position.y, _position.z, _width, _height,
            _aspect_ratio, _field_of_view, _look_at.x, _look_at.y,
            _look_at.z);
}

}}
