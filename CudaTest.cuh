#pragma once

#include "Utils.hpp"
#include "renderer/CudaRender.cuh"

using namespace utils;
using namespace render;
using namespace render::cuda;

constexpr int DEFAULT_DIM = 20;

static inline std::vector<Sphere> make_spheres() {
    Material material_1(Color(1.0, 0.1, 0.1), Albedo(0.7, 0.09, 0.1), 40.0);
    Material material_2(Color(0.9, 0.9, 0.9), Albedo(0.2, 0.01, 0.5), 30.0);
    Material black_mirror(Color(0.01, 0.01, 0.01), Albedo(0.5, 0.5, 0.1), 50.0);
    Material white_mirror(Color(0.99, 0.99, 0.99), Albedo(0.5, 0.5, 0.3), 50.0);
    Material mint_opacity(Color(0.1, 0.9, 0.5), Albedo(0.3, 0.5, 0.01), 5.0);

    return {
        Sphere(Point(0.0, 0.0, 0.0), 2.0f, material_1),
        Sphere(Point(10.0, 0.0, 0.0), 5.0, black_mirror),
        Sphere(Point(-10.0, 0.0, 0.0), 5.0, white_mirror),
        Sphere(Point(0.0, 0.0, -30.0), 20.0, mint_opacity),
        Sphere(Point(10.0, 20.0, -20.0), 8.0f, white_mirror),
        Sphere(Point(0.0, -100005.0, 0.0), 100000.0, material_2)
    };
}

static inline std::vector<Light> make_lights() {
    return {
        Light(Point(40.0f, 60.0f, 30.0f), 3.0f),
        Light(Point(-40.0f, 60.0f, 30.0f), 3.0f),
    };
}


static inline Scene make_scene(int width = DEFAULT_DIM, int height = DEFAULT_DIM) {
    return Scene(
        make_spheres(), Color(0.001, 0.001, 0.001), make_lights(),
        width, height
    );
}

inline Camera make_camera(int widht = DEFAULT_DIM, int height = DEFAULT_DIM) {
    return Camera(Point(0.0, 0.0, 50.0), to_radian(45),
                  widht, height);
}

inline CudaRender make_render(int width = DEFAULT_DIM, int height = DEFAULT_DIM) {
    return CudaRender(
        make_scene(width, height),
        make_camera(width, height),
        width, height
    );
}


class CameraTest : public testing::Test {
public:
    CameraTest();

    render::cuda::Camera _camera;
};
