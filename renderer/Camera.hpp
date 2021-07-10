#pragma once

#include "Aliases.hpp"
#include "Ray.hpp"


namespace render {

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

}