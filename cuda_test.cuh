#pragma once

#include "Utils.hpp"
#include "renderer/CudaRender.cuh"

using namespace utils;
using namespace render;
using namespace render::cuda;

constexpr int DEFAULT_DIM = 20;


static inline Objects make_objects() {
    Material material_1(Color(1.0, 1.0, 1.0), Albedo(0.1, 0.1, 0.1), 20.0);
    Material material_2(Color(0.9, 0.9, 0.9), Albedo(0.1, 0.1, 0.1), 30.0);
    return {
        std::make_shared<Sphere>(Point(0.0, 0.0, 0.0), 1.5f, material_1),
        std::make_shared<Sphere>(Point(0.0, -100005.0, 0.0), 100000.0,
                                 material_2)
    };
}

static inline Lights make_lights() {
    return Lights{
        Light(Point(0.0, 40.0, 0.0), 10.0f),
    };
}


static inline Scene make_scene(int width = DEFAULT_DIM,
    int height = DEFAULT_DIM) {
    return Scene(
        make_objects(), Color(1.0, 1.0, 1.0), make_lights(),
        width, height
    );
}

inline Camera make_camera(int widht = DEFAULT_DIM, int height = DEFAULT_DIM) {
    return Camera(Point(0.0, 0.0, 10.0), to_radian(60),
        widht, height);
}

inline CudaRender make_render(int width = DEFAULT_DIM, int height = DEFAULT_DIM) {
    return CudaRender(
        make_scene(width, height),
        make_camera(width, height),
        width, height
    );
}
